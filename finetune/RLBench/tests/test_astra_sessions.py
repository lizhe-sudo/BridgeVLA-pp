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
from astra.schemas import AstraObservation


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
        response(request_id, {"turn": {"id": turn_id, "status": "inProgress"}})
        if mode == "timeout" or mode == "timeout_then_interrupt":
            continue
        event_thread = "wrong-thread" if mode == "identity_mismatch" else thread_id
        notification("turn/started", {"threadId": event_thread, "turnId": turn_id})
        if mode == "duplicate_started":
            notification("turn/started", {"threadId": thread_id, "turnId": turn_id})
        if mode == "tool":
            notification("item/started", {
                "threadId": thread_id, "turnId": turn_id,
                "item": {"id": "item-tool", "type": "commandExecution"},
            })
        notification("item/agentMessage/delta", {
            "threadId": thread_id, "turnId": turn_id,
            "itemId": "assistant", "delta": "{\"position\":[999,999,999]}",
        })
        output = json.dumps({
            "position": [0.05 * turn_count, 0.01 * turn_count, 0.2],
            "quaternion": [0, 0, 0, 1], "gripper": 1,
        })
        item = {"id": "assistant", "type": "agentMessage", "text": output}
        notification("item/completed", {
            "threadId": thread_id, "turnId": turn_id, "item": item,
        })
        status = "failed" if mode == "turn_failed" else "completed"
        completed = {"id": turn_id, "status": status, "items": [item]}
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
                actual_after = [0.01 * (step + 1), 0.002, 0.2, 0, 0, 0, 1]
                execution = {
                    "step_id": step,
                    "raw_policy_action": {
                        "position": action.position,
                        "quaternion": action.quaternion,
                        "gripper": action.gripper,
                    },
                    "validated_action": {
                        "position": action.position,
                        "quaternion_xyzw": action.quaternion,
                        "gripper": action.gripper,
                    },
                    "eef_pose_before": measured_pose,
                    "actual_eef_pose": actual_after,
                    "effective_planner_target": action.position,
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
                self.assertNotEqual(
                    feedback["requested_target_pose"]["position"], actual_after[:3]
                )
                if step == 0:
                    with self.assertRaises(ModelServiceError):
                        policy.record_execution_feedback({**feedback, "action_id": "wrong"})
                policy.record_execution_feedback(feedback)
                if step == 0:
                    with self.assertRaisesRegex(ModelServiceError, "already recorded"):
                        policy.record_execution_feedback(feedback)
                measured_pose = actual_after

            policy.end_episode("fake_three_step_validation")
            inputs = [row for row in self.protocol()
                      if row.get("method") == "turn/start"]
            starts = [row for row in self.protocol()
                      if row.get("method") == "thread/start"]
            self.assertEqual(len(starts), 1)
            self.assertEqual(len(inputs), 3)
            self.assertEqual(len({row["params"]["threadId"] for row in inputs}), 1)
            prompts = [row["params"]["input"][0]["text"] for row in inputs]
            self.assertIn("there is no action-outcome experience", prompts[0])
            self.assertNotIn("Measured feedback for the immediately previous action", prompts[0])
            self.assertIn("Measured feedback for the immediately previous action", prompts[1])
            self.assertIn("action000", prompts[1])
            self.assertIn("Measured feedback for the immediately previous action", prompts[2])
            self.assertIn("action001", prompts[2])
            self.assertIn("observation_id=eval-x:r1:open_drawer:ep0:obs002", prompts[2])
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
