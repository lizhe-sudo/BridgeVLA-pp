"""Regression tests for episode-scoped Codex App Server control sessions."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from astra.codex_session import CodexAppServerSession
from astra.errors import InferenceDeadlineExceeded, ModelServiceError, PolicyToolViolation
from astra.execution_feedback import build_execution_feedback
from astra.action_adapter import AstraActionAdapter
from astra.schemas import AstraAction, AstraObservation


FAKE_CODEX = r'''#!/usr/bin/env python3
import json, os, sys, uuid

if sys.argv[1:] == ["--version"]:
    print("codex-cli 0.157.1-fake")
    raise SystemExit(0)

mode = os.environ.get("ASTRA_FAKE_MODE", "normal")
log_path = os.environ["ASTRA_FAKE_LOG"]
thread_id = "thread-" + uuid.uuid4().hex
session_id = "session-" + uuid.uuid4().hex
turn_count = 0

def send(value):
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()

def response(request_id, result=None):
    send({"id": request_id, "result": result or {}})

def notification(method, params):
    send({"method": method, "params": params})

def record(request):
    with open(log_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(request, ensure_ascii=False) + "\n")

for line in sys.stdin:
    request = json.loads(line)
    record(request)
    method = request.get("method")
    params = request.get("params", {})
    request_id = request.get("id")
    if method == "initialize":
        response(request_id, {"serverInfo": {"name": "fake", "version": "1"}})
    elif method == "initialized":
        pass
    elif method == "thread/start":
        thread = {"id": thread_id, "sessionId": session_id, "ephemeral": False}
        if mode == "missing_session_id":
            thread.pop("sessionId")
        response(request_id, {"thread": thread, "instructionSources": [], "model": "gpt-6-luna"})
        notification("thread/started", {"thread": thread})
    elif method == "turn/start":
        turn_count += 1
        turn_id = "turn-" + str(turn_count)
        if mode == "second_wait_start" and turn_count == 2:
            continue
        if mode == "second_missing_id" and turn_count == 2:
            response(request_id, {"turn": {"status": "inProgress"}})
            continue
        if mode == "second_reused_id" and turn_count == 2:
            turn_id = "turn-1"
        response(request_id, {"turn": {"id": turn_id, "status": "inProgress"}})
        if mode == "timeout" or mode == "timeout_then_interrupt":
            continue
        if mode == "stale_old_event" and turn_count == 2:
            stale_item = {"id": "old-assistant", "type": "agentMessage",
                          "phase": "final_answer", "text": "old action"}
            notification("item/completed", {
                "threadId": thread_id, "turnId": "turn-1", "item": stale_item,
            })
            notification("turn/completed", {
                "threadId": thread_id,
                "turn": {"id": "turn-1", "status": "completed", "items": [stale_item]},
            })
        event_thread = "wrong-thread" if (
            mode == "identity_mismatch" or
            (mode == "second_wrong_identity" and turn_count == 2)
        ) else thread_id
        notification("turn/started", {"threadId": event_thread, "turnId": turn_id})
        if mode == "duplicate_started":
            notification("turn/started", {"threadId": thread_id, "turnId": turn_id})
        if mode == "tool":
            notification("item/started", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "item-tool", "type": "commandExecution"},
            })
        if mode in ("context_compaction", "context_compaction_tool"):
            notification("item/started", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "compact-1", "type": "contextCompaction"},
            })
            notification("item/completed", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "compact-1", "type": "contextCompaction"},
            })
        if mode == "context_compaction_tool":
            notification("item/started", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "item-tool", "type": "commandExecution"},
            })
        if mode == "unknown_item":
            notification("item/started", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "future-item", "type": "futureToolItem"},
            })
        if mode == "legacy_context_compaction":
            notification("thread/compacted", {
                "threadId": thread_id, "turnId": turn_id,
            })
        delta_text = "{\"position\":[999,999,999]}"
        if mode == "valid_action_delta":
            delta_text = json.dumps({"position": [0.8, 0.8, 0.8],
                                    "quaternion": [0, 0, 0, 1], "gripper": 0})
        notification("item/agentMessage/delta", {
            "threadId": thread_id, "turnId": turn_id,
            "itemId": "assistant", "delta": delta_text,
        })
        output = json.dumps({
            "position": [0.05 * turn_count, 0.01 * turn_count, 0.2],
            "quaternion": [0, 0, 0, 1], "gripper": 1,
        })
        item = {"id": "assistant", "type": "agentMessage", "text": output}
        if mode != "phase_missing":
            item["phase"] = "final_answer"
        if mode in ("commentary_and_final", "commentary_only"):
            commentary = {
                "id": "commentary", "type": "agentMessage", "phase": "commentary",
                "text": json.dumps({"position": [999, 999, 999],
                                    "quaternion": [0, 0, 0, 1], "gripper": 1}),
            }
            notification("item/completed", {
                "threadId": thread_id, "turnId": turn_id, "item": commentary,
            })
            if mode == "commentary_only":
                item = commentary
        if mode == "commentary_and_final":
            # The assistant's interim text is valid action-shaped JSON, but
            # only the final_answer phase is eligible for execution.
            pass
        if mode == "conflicting_final":
            conflicting = dict(item, id="assistant-conflict",
                               text=json.dumps({"position": [0.91, 0.92, 0.93],
                                                "quaternion": [0, 0, 0, 1],
                                                "gripper": 0}))
            notification("item/completed", {
                "threadId": thread_id, "turnId": turn_id, "item": conflicting,
            })
        notification("item/completed", {
            "threadId": thread_id, "turnId": turn_id, "item": item,
        })
        status = "failed" if mode == "turn_failed" else "completed"
        snapshot_item = dict(item)
        if mode == "conflicting_item_id":
            snapshot_item["text"] = output + " conflicting snapshot"
        if mode == "conflicting_phase_item":
            snapshot_item["phase"] = "commentary"
        snapshot_items = [snapshot_item]
        if mode in ("commentary_and_final", "commentary_only"):
            snapshot_items.insert(0, commentary)
        if mode == "conflicting_final":
            snapshot_items.append(conflicting)
        completed = {"id": turn_id, "status": status, "items": snapshot_items}
        notification("turn/completed", {"threadId": thread_id, "turn": completed})
        if mode == "duplicate_completed":
            notification("turn/completed", {"threadId": thread_id, "turn": completed})
    elif method == "turn/interrupt":
        response(request_id, {})
        notification("turn/completed", {
            "threadId": thread_id,
            "turn": {"id": params.get("turnId"), "status": "interrupted", "items": []},
        })
    else:
        response(request_id, {})
'''


class FakeAppServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.executable = self.root / "fake codex 控制端"
        self.executable.write_text(FAKE_CODEX, encoding="utf-8")
        self.executable.chmod(0o755)
        self.cwd = self.root / "stable episode cwd"
        self.cwd.mkdir()
        self.log_path = self.root / "protocol.jsonl"
        self.env = mock.patch.dict(os.environ, {
            "ASTRA_FAKE_LOG": str(self.log_path), "ASTRA_FAKE_MODE": "normal",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.temp_dir.cleanup)

    def protocol(self):
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    def new_session(self, timeout=2.0):
        session = CodexAppServerSession(
            self.executable, self.cwd, "gpt-6-luna", "max", timeout,
        )
        session.create_thread("Only use explicit control inputs; no tools.")
        return session

    def images(self):
        result = {}
        for name in ("front", "left_shoulder", "right_shoulder", "wrist"):
            path = self.root / f"{name}.png"
            path.write_bytes(b"fake image bytes")
            result[name] = path
        return result

    def schema(self):
        return {
            "type": "object", "required": ["position", "quaternion", "gripper"],
            "properties": {}, "additionalProperties": False,
        }

    def test_one_native_thread_serves_three_turns_with_current_images_and_schema(self):
        session = self.new_session()
        try:
            returned = []
            for step in range(3):
                text, metadata = session.run_turn(
                    f"current step {step}", self.images(), self.schema(),
                )
                returned.append(json.loads(text))
                self.assertEqual(metadata["thread_id"], session.thread_id)
                self.assertEqual(metadata["session_id"], session.session_id)
                self.assertEqual(metadata["turn_id"], f"turn-{step + 1}")
            requests = self.protocol()
            starts = [row for row in requests if row.get("method") == "thread/start"]
            turns = [row for row in requests if row.get("method") == "turn/start"]
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(turns), 3)
            self.assertFalse(starts[0]["params"]["ephemeral"])
            self.assertNotIn("--last", starts[0].get("argv", []))
            self.assertTrue(all(row["params"]["threadId"] == session.thread_id
                                for row in turns))
            for index, request in enumerate(turns):
                self.assertEqual(request["params"]["effort"], "max")
                self.assertEqual(request["params"]["model"], "gpt-6-luna")
                self.assertEqual(request["params"]["outputSchema"], self.schema())
                self.assertEqual(request["params"]["input"][0]["text"],
                                 f"current step {index}")
                self.assertEqual(
                    [item["path"].split("/")[-1]
                     for item in request["params"]["input"][1:]],
                    ["front.png", "left_shoulder.png", "right_shoulder.png", "wrist.png"],
                )
            self.assertNotEqual(returned[0]["position"], [999, 999, 999])
            self.assertEqual(session.turn_count, 3)
            self.assertEqual(session.rpc_request_count, 5)
            self.assertEqual(session.turn_records[0]["status"], "completed")
        finally:
            session.close()
        self.assertTrue(session.process_group_exit_confirmed)

    def test_episode_sessions_get_distinct_ids_and_missing_id_fails_before_turn(self):
        first = self.new_session()
        second = CodexAppServerSession(
            self.executable, self.cwd, "gpt-6-luna", "max", 2,
        )
        try:
            second_identity = second.create_thread("new episode only")
            self.assertNotEqual(first.session_id, second_identity["session_id"])
            self.assertNotEqual(first.thread_id, second_identity["thread_id"])
        finally:
            first.close()
            second.close()

        self.log_path.unlink()
        with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "missing_session_id"}):
            missing = CodexAppServerSession(
                self.executable, self.cwd, "gpt-6-luna", "max", 2,
            )
            with self.assertRaisesRegex(ModelServiceError, "thread.sessionId"):
                missing.create_thread("new episode")
            self.assertEqual(missing.turn_count, 0)
            self.assertEqual(
                [row["method"] for row in self.protocol()],
                ["initialize", "initialized", "thread/start"],
            )
            missing.close(force=True)

    def test_identity_tool_and_failed_turns_never_yield_an_action(self):
        for mode, error_type in (
            ("identity_mismatch", ModelServiceError),
            ("tool", PolicyToolViolation),
            ("turn_failed", ModelServiceError),
            ("duplicate_started", ModelServiceError),
        ):
            with self.subTest(mode=mode):
                self.log_path.unlink(missing_ok=True)
                with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": mode}):
                    session = self.new_session()
                    try:
                        with self.assertRaises(error_type):
                            session.run_turn("step", self.images(), self.schema())
                        self.assertTrue(session.failed)
                        if mode == "identity_mismatch":
                            identity_event = next(
                                event for event in
                                session.last_turn_metadata["event_evidence"]
                                if event.get("method") == "turn/started"
                            )
                            self.assertEqual(identity_event["thread_id"], "wrong-thread")
                    finally:
                        session.close(force=True)

    def test_timeout_requests_interrupt_and_fails_closed(self):
        with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "timeout_then_interrupt"}):
            session = CodexAppServerSession(
                self.executable, self.cwd, "gpt-6-luna", "max", 0.1,
            )
            session.create_thread("timeout test")
            try:
                with self.assertRaises(InferenceDeadlineExceeded):
                    session.run_turn("step", self.images(), self.schema())
                self.assertTrue(session.failed)
                self.assertTrue(session.last_turn_metadata["turn_end_confirmed"])
                methods = [row["method"] for row in self.protocol()]
                self.assertIn("turn/interrupt", methods)
                self.assertEqual(session.turn_count, 1)
            finally:
                session.close(force=True)
            self.assertTrue(session.process_group_exit_confirmed)

    def test_turn_start_failures_never_rebind_or_mutate_previous_turn(self):
        cases = (
            ("before_write", "normal"),
            ("waiting_response", "second_wait_start"),
            ("missing_turn_id", "second_missing_id"),
            ("wrong_identity", "second_wrong_identity"),
            ("reused_turn_id", "second_reused_id"),
        )
        for failure, mode in cases:
            with self.subTest(failure=failure):
                self.log_path.unlink(missing_ok=True)
                with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": mode}):
                    session = self.new_session(timeout=0.15 if mode == "second_wait_start" else 2)
                    try:
                        _, first = session.run_turn(
                            "first", self.images(), self.schema(),
                            request_identity={"step_id": 0, "action_id": "action-0",
                                              "observation_id": "obs-0"},
                        )
                        first_record = json.loads(json.dumps(session.turn_records[0]))
                        if failure == "before_write":
                            original_send = session._send
                            count = {"turn_start": 1}

                            def fail_before_second_write(message):
                                if message.get("method") == "turn/start":
                                    count["turn_start"] += 1
                                    if count["turn_start"] == 2:
                                        raise ModelServiceError(
                                            "simulated turn/start write failure",
                                            error_code="app_server_write_failed",
                                        )
                                return original_send(message)

                            session._send = fail_before_second_write
                        with self.assertRaises((ModelServiceError,
                                                InferenceDeadlineExceeded)):
                            session.run_turn(
                                "second", self.images(), self.schema(),
                                request_identity={"step_id": 1, "action_id": "action-1",
                                                  "observation_id": "obs-1"},
                            )
                        self.assertEqual(session.turn_records[0], first_record)
                        second = session.turn_records[1]
                        self.assertNotEqual(second["control_request_id"],
                                            first["control_request_id"])
                        self.assertEqual(second["request_identity"]["action_id"],
                                         "action-1")
                        self.assertNotEqual(second.get("turn_id"), first.get("turn_id"))
                        self.assertIn(second.get("turn_end_confirmed"), (False, None))
                        self.assertEqual(
                            second.get("turn_start_sent"),
                            failure != "before_write",
                        )
                        self.assertTrue(second.get("process_group_exit_confirmed"))
                        starts = [row for row in self.protocol()
                                  if row.get("method") == "turn/start"]
                        self.assertEqual(len(starts), 1 if failure == "before_write" else 2)
                        if failure == "waiting_response":
                            self.assertNotIn(
                                "turn/interrupt",
                                [row.get("method") for row in self.protocol()],
                            )
                    finally:
                        session.close(force=True)

    def test_one_absolute_deadline_covers_start_response_and_completion(self):
        class FakeClock:
            now = 50.0

            def __call__(self):
                return self.now

        clock = FakeClock()
        session = CodexAppServerSession(
            self.executable, self.cwd, "gpt-6-luna", "max", 10,
            monotonic=clock,
        )
        session.thread_id = "thread-test"
        session.session_id = "session-test"

        def fake_request(method, params, timeout=None, deadline=None):
            self.assertEqual(method, "turn/start")
            self.assertEqual(deadline, 60.0)
            clock.now = 56.0  # Starting the turn consumed 6 of 10 seconds.
            return {"turn": {"id": "turn-test", "status": "inProgress"}}

        def fake_consume(turn_id, deadline):
            self.assertEqual(turn_id, "turn-test")
            self.assertEqual(deadline, 60.0)
            self.assertEqual(deadline - clock.now, 4.0)
            return ('{"position":[0,0,0],"quaternion":[0,0,0,1],"gripper":1}',
                    [], [], True)

        actual_consume = session._consume_notifications
        session._request = fake_request
        session._consume_notifications = fake_consume
        _, metadata = session.run_turn("one request", self.images(), self.schema())
        self.assertEqual(metadata["latency_seconds"], 6.0)
        self.assertEqual(session.turn_count, 1)
        session._notifications = [{
            "method": "turn/completed",
            "params": {"threadId": "thread-test",
                       "turn": {"id": "turn-test", "status": "completed"}},
        }]
        clock.now = 60.0
        session._consume_notifications = actual_consume
        with self.assertRaises(InferenceDeadlineExceeded):
            session._consume_notifications("turn-test", deadline=60.0)

    def test_final_message_selection_phase_deduplication_and_stale_events(self):
        cases = (
            ("commentary_and_final", 0.05),
            ("phase_missing", 0.05),
            ("valid_action_delta", 0.05),
        )
        for mode, expected_x in cases:
            with self.subTest(mode=mode):
                self.log_path.unlink(missing_ok=True)
                with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": mode}):
                    session = self.new_session()
                    try:
                        text, metadata = session.run_turn(
                            "step", self.images(), self.schema()
                        )
                        self.assertEqual(json.loads(text)["position"][0], expected_x)
                        self.assertEqual(len(session.turn_records), 1)
                        self.assertEqual(metadata["final_message_candidate_count"], 1)
                        self.assertEqual(metadata["final_message_item_ids"], ["assistant"])
                        if mode == "phase_missing":
                            self.assertEqual(
                                metadata["phase_compatibility"],
                                "single_completed_unknown_phase_message_no_commentary",
                            )
                        if mode == "commentary_and_final":
                            self.assertEqual(
                                session.last_turn_metadata["turn_end_state"], "completed"
                            )
                    finally:
                        session.close(force=True)

        for mode, error_code in (
            ("commentary_only", "missing_final_message"),
            ("conflicting_final", "conflicting_final_messages"),
            ("conflicting_item_id", "conflicting_agent_message_item"),
            ("conflicting_phase_item", "conflicting_agent_message_item"),
            ("unknown_item", "tool_event_detected"),
        ):
            with self.subTest(rejected_mode=mode):
                self.log_path.unlink(missing_ok=True)
                with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": mode}):
                    session = self.new_session()
                    try:
                        with self.assertRaises((ModelServiceError,
                                                PolicyToolViolation)) as raised:
                            session.run_turn("step", self.images(), self.schema())
                        self.assertEqual(raised.exception.error_code, error_code)
                        self.assertTrue(session.failed)
                    finally:
                        session.close(force=True)

        self.log_path.unlink(missing_ok=True)
        with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "duplicate_completed"}):
            session = self.new_session()
            try:
                # If both completion notifications are already buffered the
                # protocol is rejected; if the duplicate arrives later, one
                # run_turn still returns at most one action and the stale event
                # cannot become a second action on the next turn.
                returned = []
                try:
                    returned.append(session.run_turn(
                        "first", self.images(), self.schema()
                    )[0])
                except ModelServiceError:
                    pass
                self.assertLessEqual(len(returned), 1)
                self.assertEqual(len(session.turn_records), 1)
                self.assertEqual(session.turn_count, 1)
            finally:
                session.close(force=True)

        self.log_path.unlink(missing_ok=True)
        with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "stale_old_event"}):
            session = self.new_session()
            try:
                session.run_turn("first", self.images(), self.schema())
                text, _ = session.run_turn("second", self.images(), self.schema())
                self.assertEqual(json.loads(text)["position"][0], 0.1)
                self.assertTrue(any(event.get("stale_duplicate")
                                    for event in session.last_turn_metadata["event_evidence"]))
            finally:
                session.close(force=True)

    def test_context_compaction_is_safe_but_tools_remain_rejected(self):
        from astra.codex_policy import CodexAstraPolicy

        for mode in ("context_compaction", "legacy_context_compaction"):
            self.log_path.unlink(missing_ok=True)
            with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": mode}):
                session = self.new_session()
                try:
                    text, _ = session.run_turn("step", self.images(), self.schema())
                    self.assertEqual(json.loads(text)["position"][0], 0.05)
                    self.assertEqual(len(session.context_compression_events),
                                     2 if mode == "context_compaction" else 1)
                    self.assertTrue(all(event.get("thread_id") == session.thread_id
                                        for event in session.context_compression_events))
                    self.assertTrue(all(event.get("turn_id") == "turn-1"
                                        for event in session.context_compression_events))
                    if mode == "context_compaction":
                        safe_events = [
                            json.loads(line) for line in
                            CodexAstraPolicy._safe_event_log(
                                CodexAstraPolicy._event_evidence_text(
                                    session.last_turn_metadata["event_evidence"]
                                )
                            ).splitlines()
                        ]
                        context_event = next(
                            row for row in safe_events
                            if row.get("item_type") == "contextCompaction"
                        )
                        self.assertEqual(context_event["thread_id"], session.thread_id)
                        self.assertEqual(context_event["turn_id"], "turn-1")
                        self.assertEqual(context_event["item_id"], "compact-1")
                        self.assertEqual(context_event["location"], "item_lifecycle")
                finally:
                    session.close(force=True)
        self.log_path.unlink(missing_ok=True)
        with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "context_compaction_tool"}):
            session = self.new_session()
            try:
                with self.assertRaises(PolicyToolViolation):
                    session.run_turn("step", self.images(), self.schema())
                self.assertEqual(len(session.context_compression_events), 2)
                self.assertEqual(session.turn_count, 1)
            finally:
                session.close(force=True)

    def test_real_action_adapter_diagnostics_keep_requested_submitted_and_measured_quaternions(self):
        action = AstraAction(
            position=[0.13, 0.24, 0.35], quaternion=[0.0, 0.0, 0.0, 2.0],
            gripper=1,
        )
        adapter = AstraActionAdapter("fixed0")
        final_action = adapter.adapt(action)
        diagnostics = adapter.last_diagnostics
        execution = {
            **diagnostics,
            "step_id": 3,
            "eef_pose_before": [0, 0, 0.2, 0, 0, 0, 1],
            "actual_eef_pose": [0.1, 0.2, 0.3, 0, 0, 1, 0],
            "effective_planner_target": [0.13, 0.24, 0.35],
            "effective_target_source": "action_adapter",
            "environment_step_returned": True,
            "planner_returned": True,
        }
        feedback = build_execution_feedback(execution, "action-3", "obs-3", "obs-4")
        self.assertEqual(final_action[3:7], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(feedback["requested_target_pose"]["quaternion_xyzw"],
                         [0.0, 0.0, 0.0, 2.0])
        self.assertEqual(feedback["validated_action_submitted"]["quaternion_xyzw"],
                         [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(feedback["actual_orientation_before_xyzw"], [0, 0, 0, 1])
        self.assertEqual(feedback["actual_orientation_after_xyzw"], [0, 0, 1, 0])
        broken = {**execution, "raw_policy_action": {
            "position": action.position, "quaternion": action.quaternion,
            "gripper": action.gripper,
        }}
        with self.assertRaisesRegex(ValueError, "raw_policy_action.quaternion_xyzw"):
            build_execution_feedback(broken, "action-3", "obs-3", "obs-4")

    def test_actual_policy_appends_measured_feedback_in_one_episode_thread(self):
        from astra.codex_policy import CodexAstraPolicy

        which_patch = mock.patch(
            "astra.codex_policy.shutil.which", return_value=str(self.executable)
        )
        which_patch.start()
        self.addCleanup(which_patch.stop)
        policy = CodexAstraPolicy(
            model="gpt-6-luna", reasoning_effort="max", timeout=2,
            work_root=str(self.root / "policy work"), collision_mode="fixed0",
        )
        policy.set_evaluation_context("eval-x", 1, "open_drawer", 0, 25)
        policy.reset("Open the drawer named by this instruction.")
        episode_dir = policy._episode_dir
        measured_pose = [0.0, 0.0, 0.2, 0, 0, 0, 1]
        environment_action_count = 0
        try:
            for step in range(3):
                images = {name: np.full((8, 8, 3), step, dtype=np.uint8)
                          for name in CodexAstraPolicy.IMAGE_FIELDS}
                obs = AstraObservation(
                    "Open the drawer named by this instruction.", images,
                    measured_pose, True,
                )
                action = policy.act(obs)
                metadata = dict(policy.last_metadata)
                self.assertEqual(metadata["app_server_request_count"],
                                 3 if step == 0 else 1)
                self.assertEqual(metadata["thread_id"], policy._session.thread_id)
                if step:
                    self.assertEqual(
                        metadata["thread_id"],
                        json.loads((episode_dir / "session_manifest.json").read_text())["thread_id"],
                    )
                # This 5 cm action is legal and remains unchanged by prompt-only guidance.
                self.assertAlmostEqual(action.position[0], 0.05 * (step + 1))
                action_adapter = AstraActionAdapter("fixed0")
                final_action = action_adapter.adapt(action)
                diagnostics = action_adapter.last_diagnostics
                actual_after = [0.01 * (step + 1), 0.002, 0.2, 0, 0, 0, 1]
                execution = {
                    **diagnostics,
                    "step_id": step,
                    "eef_pose_before": measured_pose,
                    "actual_eef_pose": actual_after,
                    "effective_planner_target": diagnostics["validated_action"]["position"],
                    "effective_target_source": "computed_by_existing_adapter",
                    "gripper_before": True, "gripper_after": True,
                    "environment_step_returned": True, "planner_returned": True,
                    "reward": 100, "object_pose": [99, 99, 99],
                }
                feedback = build_execution_feedback(
                    execution, metadata["action_id"], metadata["observation_id"],
                    policy.observation_id_for_step(step + 1),
                )
                self.assertNotIn("reward", feedback)
                self.assertNotIn("object_pose", feedback)
                self.assertIsNotNone(
                    feedback["requested_target_pose"]["quaternion_xyzw"]
                )
                self.assertEqual(feedback["requested_target_pose"]["quaternion_xyzw"],
                                 action.quaternion)
                self.assertNotEqual(
                    feedback["requested_target_pose"]["position"], actual_after[:3]
                )
                if step == 0:
                    with self.assertRaises(ModelServiceError):
                        policy.record_execution_feedback({**feedback, "action_id": "wrong"})
                policy.record_execution_feedback(feedback)
                environment_action_count += 1
                if step == 0:
                    with self.assertRaisesRegex(ModelServiceError, "already recorded"):
                        policy.record_execution_feedback(feedback)
                measured_pose = actual_after

            turn_records = list(policy._session.turn_records)
            policy.end_episode("fake_three_step_validation")
            inputs = [row for row in self.protocol()
                      if row.get("method") == "turn/start"]
            starts = [row for row in self.protocol()
                      if row.get("method") == "thread/start"]
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(inputs), 3)
            self.assertEqual(environment_action_count, len(inputs))
            self.assertEqual(len({row["params"]["threadId"] for row in inputs}), 1)
            self.assertEqual(len(turn_records), 3)
            self.assertEqual(len({row["turn_id"] for row in turn_records}), 3)
            self.assertEqual(len({row["control_request_id"] for row in turn_records}), 3)
            self.assertEqual([row["step_id"] for row in turn_records], [0, 1, 2])
            self.assertTrue(all(row["status"] == "completed" for row in turn_records))
            self.assertTrue(all(row["thread_id"] == inputs[0]["params"]["threadId"]
                                for row in turn_records))
            prompts = [row["params"]["input"][0]["text"] for row in inputs]
            self.assertIn("there is no action-outcome experience", prompts[0])
            self.assertNotIn("Measured feedback for the immediately previous action", prompts[0])
            self.assertIn("Measured feedback for the immediately previous action", prompts[1])
            self.assertIn("action000", prompts[1])
            self.assertIn("Measured feedback for the immediately previous action", prompts[2])
            self.assertIn("action001", prompts[2])
            self.assertIn("observation_id=eval-x:r1:open_drawer:ep0:obs002", prompts[2])
            self.assertIn('"quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]', prompts[1])
            self.assertNotIn("reward", prompts[1])
            control_rows = [json.loads(line) for line in
                            (episode_dir / "control_messages.jsonl").read_text().splitlines()]
            feedback_rows = [row for row in control_rows
                             if row.get("event") == "execution_feedback"]
            self.assertEqual([row["step_id"] for row in feedback_rows], [0, 1, 2])
            self.assertEqual(len({row["thread_id"] for row in feedback_rows}), 1)
            self.assertEqual(feedback_rows[1]["eef_pose_after"],
                             [0.02, 0.002, 0.2, 0, 0, 0, 1])
            first_thread_id = inputs[0]["params"]["threadId"]
            first_session_id = policy._session_manifest_path.read_text()

            # Reset begins an isolated new episode: app state is cleared and
            # the next accepted action must belong to a freshly created thread.
            policy.set_evaluation_context("eval-x", 1, "open_drawer", 1, 25)
            policy.reset("A different demonstration episode.")
            self.assertIsNone(policy._pending_feedback)
            self.assertIsNone(policy._last_action_binding)
            self.assertIsNone(policy._session)
            next_images = {name: np.zeros((8, 8, 3), dtype=np.uint8)
                           for name in CodexAstraPolicy.IMAGE_FIELDS}
            next_action = policy.act(AstraObservation(
                "A different demonstration episode.", next_images,
                [0, 0, 0, 0, 0, 0, 1], True,
            ))
            self.assertEqual(next_action.position[0], 0.05)
            next_thread = policy.last_metadata["thread_id"]
            self.assertNotEqual(next_thread, first_thread_id)
            self.assertNotEqual(
                policy.last_metadata["control_session_id"],
                json.loads(first_session_id)["control_session_id"],
            )
            self.assertEqual(len([row for row in self.protocol()
                                  if row.get("method") == "thread/start"]), 2)
            policy.end_episode("second_fake_episode_complete")
            self.assertIsNone(policy._control_work_dir)
        finally:
            policy.close()

    def test_binding_a_failed_new_request_cannot_reuse_previous_turn_id(self):
        from astra.codex_policy import CodexAstraPolicy

        class Session:
            thread_id = "thread-1"
            session_id = "session-1"
            turn_records = [{
                "control_request_id": "request-old", "step_id": 0,
                "action_id": "action-old", "observation_id": "obs-old",
                "turn_id": "turn-old", "status": "completed",
            }]

        policy = object.__new__(CodexAstraPolicy)
        policy._session = Session()
        previous = json.loads(json.dumps(policy._session.turn_records[0]))
        policy._bind_session_turn(
            1, "action-new", "obs-new", {
                "control_request_id": "request-new",
                "request_identity": {
                    "step_id": 1, "action_id": "action-new",
                    "observation_id": "obs-new",
                },
                "turn_id": None,
                "turn_end_confirmed": False,
                "turn_end_state": "unknown",
            }, "rejected",
        )
        self.assertEqual(policy._session.turn_records[0], previous)
        self.assertIsNone(policy._session.turn_records[1]["turn_id"])
        self.assertEqual(policy._session.turn_records[1]["action_id"], "action-new")
        self.assertEqual(policy._session.turn_records[1]["status"], "rejected")

    def test_policy_session_error_does_not_create_a_replacement_thread(self):
        from astra.codex_policy import CodexAstraPolicy

        which_patch = mock.patch(
            "astra.codex_policy.shutil.which", return_value=str(self.executable)
        )
        which_patch.start()
        self.addCleanup(which_patch.stop)
        policy = CodexAstraPolicy(
            model="gpt-6-luna", reasoning_effort="max", timeout=2,
            work_root=str(self.root / "failed policy"), collision_mode="fixed0",
        )
        policy.set_evaluation_context("eval-failure", 1, "open_drawer", 0, 25)
        policy.reset("Open the drawer.")
        images = {name: np.zeros((8, 8, 3), dtype=np.uint8)
                  for name in CodexAstraPolicy.IMAGE_FIELDS}
        observation = AstraObservation(
            "Open the drawer.", images, [0, 0, 0, 0, 0, 0, 1], True,
        )
        try:
            with mock.patch.dict(os.environ, {"ASTRA_FAKE_MODE": "turn_failed"}):
                with self.assertRaises(ModelServiceError):
                    policy.act(observation)
            requests = self.protocol()
            self.assertEqual(len([row for row in requests
                                  if row.get("method") == "thread/start"]), 1)
            self.assertEqual(len([row for row in requests
                                  if row.get("method") == "turn/start"]), 1)
            self.assertTrue(policy._session.failed)
            self.assertFalse(Path(policy.last_metadata["accepted_action_path"]).exists())
        finally:
            policy.close()


if __name__ == "__main__":
    unittest.main()
