"""Episode-scoped Codex App Server control sessions.

The adapter deliberately supports one thread and one in-flight turn at a time.
It accepts only completed assistant messages, validates thread/turn identity,
and fails closed if App Server starts a tool item or loses the current turn.
"""

import json
import os
import selectors
import signal
import subprocess
import time
import uuid
from pathlib import Path

from .errors import (
    InferenceDeadlineExceeded,
    ModelServiceError,
    PolicyToolViolation,
    sanitize_diagnostic,
)


SAFE_ITEM_TYPES = {
    "agentMessage", "reasoning", "plan", "userMessage",
    "developerMessage", "systemMessage", "contextCompaction",
}
SAFE_ITEM_DELTA_TYPES = {"agentMessage", "reasoning", "plan"}


class CodexAppServerSession:
    """One App Server process and one newly created native thread."""

    def __init__(self, executable, cwd, model, reasoning_effort, timeout,
                 process_factory=subprocess.Popen, monotonic=time.monotonic):
        self.executable = str(executable)
        self.cwd = Path(cwd).resolve()
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout = float(timeout)
        self.process_factory = process_factory
        self.monotonic = monotonic
        self.process = None
        self.selector = None
        self._stdout_buffer = bytearray()
        self._next_request_id = 1
        self._notifications = []
        self._stderr_file = None
        self.thread_id = None
        self.session_id = None
        self.resolved_model = None
        self.app_server_info = None
        self.instruction_sources = None
        self.initialization_latency_seconds = None
        self.failed = False
        self.closed = False
        self.process_group_exit_confirmed = None
        self.turn_count = 0
        self.rpc_request_count = 0
        self.turn_records = []
        self.context_compression_events = []
        self.last_turn_metadata = {}

    @staticmethod
    def _rpc_error(error, default_code):
        message = error.get("message") if isinstance(error, dict) else None
        safe = sanitize_diagnostic(message if isinstance(message, str) else "")
        return ModelServiceError(
            safe["text"] or "Codex App Server request failed",
            error_code=default_code,
        )

    def _start(self):
        if self.process is not None:
            return
        if not self.cwd.is_dir():
            raise ModelServiceError("episode control working directory is missing",
                                    error_code="control_cwd_missing")
        try:
            self.process = self.process_factory(
                [self.executable, "app-server", "--stdio"],
                cwd=str(self.cwd), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0, start_new_session=True,
            )
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
        except OSError as exc:
            self.failed = True
            raise ModelServiceError("failed to start Codex App Server",
                                    error_code="app_server_start_failed") from exc

    def _read_line(self, deadline):
        while True:
            if self.monotonic() >= deadline:
                raise InferenceDeadlineExceeded(
                    "Codex App Server request exceeded its configured deadline",
                    error_code="inference_deadline_exceeded",
                )
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[:newline + 1]
                try:
                    value = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ModelServiceError(
                        "Codex App Server emitted an invalid JSON-RPC message",
                        error_code="app_server_invalid_message",
                    ) from exc
                if not isinstance(value, dict):
                    raise ModelServiceError(
                        "Codex App Server emitted a non-object JSON-RPC message",
                        error_code="app_server_invalid_message",
                    )
                return value

            if self.process is None or self.process.poll() is not None:
                raise ModelServiceError(
                    "Codex App Server exited before completing the request",
                    error_code="app_server_exited",
                )
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                raise InferenceDeadlineExceeded(
                    "Codex App Server request exceeded its configured deadline",
                    error_code="inference_deadline_exceeded",
                )
            ready = self.selector.select(remaining)
            if not ready:
                raise InferenceDeadlineExceeded(
                    "Codex App Server request exceeded its configured deadline",
                    error_code="inference_deadline_exceeded",
                )
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise ModelServiceError(
                    "Codex App Server closed its output stream",
                    error_code="app_server_stream_closed",
                )
            self._stdout_buffer.extend(chunk)

    def _send(self, message):
        if self.process is None or self.process.poll() is not None:
            raise ModelServiceError("Codex App Server is not running",
                                    error_code="app_server_not_running")
        try:
            wire = (json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                    + "\n").encode("utf-8")
            self.process.stdin.write(wire)
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            self.failed = True
            raise ModelServiceError("failed to write to Codex App Server",
                                    error_code="app_server_write_failed") from exc

    def _request(self, method, params, timeout=None, deadline=None):
        if deadline is None:
            deadline = self.monotonic() + (
                self.timeout if timeout is None else float(timeout)
            )
        if self.monotonic() >= deadline:
            raise InferenceDeadlineExceeded(
                "Codex App Server request exceeded its configured deadline",
                error_code="inference_deadline_exceeded",
            )
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send({"method": method, "id": request_id, "params": params})
        if method == "turn/start":
            self.last_turn_metadata["app_server_rpc_request_id"] = request_id
            self.last_turn_metadata["turn_start_sent"] = True
        self.rpc_request_count += 1
        if self.monotonic() >= deadline:
            raise InferenceDeadlineExceeded(
                "Codex App Server request exceeded its configured deadline",
                error_code="inference_deadline_exceeded",
            )
        while True:
            if self.monotonic() >= deadline:
                raise InferenceDeadlineExceeded(
                    "Codex App Server request exceeded its configured deadline",
                    error_code="inference_deadline_exceeded",
                )
            message = self._read_line(deadline)
            if message.get("id") == request_id:
                if self.monotonic() >= deadline:
                    raise InferenceDeadlineExceeded(
                        "Codex App Server request exceeded its configured deadline",
                        error_code="inference_deadline_exceeded",
                    )
                if "error" in message:
                    raise self._rpc_error(message["error"],
                                          "app_server_request_failed")
                if "result" not in message or not isinstance(message["result"], dict):
                    raise ModelServiceError(
                        "Codex App Server returned a malformed request result",
                        error_code="app_server_invalid_response",
                    )
                return message["result"]
            if "method" in message and "id" not in message:
                self._notifications.append(message)
                continue
            # A response for a different outstanding request cannot belong to
            # this strictly serial client; do not guess which request it is for.
            raise ModelServiceError(
                "Codex App Server response identity did not match the active request",
                error_code="app_server_response_mismatch",
            )

    def _notify(self, method, params):
        self._send({"method": method, "params": params})

    def _initialize(self):
        self._start()
        result = self._request("initialize", {
            "clientInfo": {
                "name": "astra_rlbench_eval",
                "title": "Astra RLBench episode controller",
                "version": "1",
            },
            "capabilities": {"experimentalApi": False},
        })
        server_info = result.get("serverInfo")
        if isinstance(server_info, dict):
            self.app_server_info = {
                key: value for key, value in server_info.items()
                if key in ("name", "version") and isinstance(value, str)
            }
        self._notify("initialized", {})

    def create_thread(self, developer_instructions):
        if self.failed or self.closed:
            raise ModelServiceError("Codex control session is not usable",
                                    error_code="control_session_unusable")
        initialized_at = self.monotonic()
        try:
            self._initialize()
            result = self._request("thread/start", {
                "model": self.model,
                "allowProviderModelFallback": False,
                "cwd": str(self.cwd),
                "runtimeWorkspaceRoots": [],
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "ephemeral": False,
                "developerInstructions": str(developer_instructions),
                "dynamicTools": [],
                "environments": [],
                "serviceName": "astra_rlbench_eval",
            })
        finally:
            self.initialization_latency_seconds = max(
                0.0, self.monotonic() - initialized_at
            )
        thread = result.get("thread")
        if not isinstance(thread, dict):
            self.failed = True
            raise ModelServiceError("thread/start did not return a thread",
                                    error_code="thread_identity_missing")
        thread_id = thread.get("id")
        session_id = thread.get("sessionId")
        if (not isinstance(thread_id, str) or not thread_id.strip()
                or not isinstance(session_id, str) or not session_id.strip()):
            self.failed = True
            raise ModelServiceError(
                "thread/start did not return both thread.id and thread.sessionId",
                error_code="thread_identity_missing",
            )
        instruction_sources = result.get("instructionSources")
        if not isinstance(instruction_sources, list):
            self.failed = True
            raise ModelServiceError("thread instruction sources were missing or malformed",
                                    error_code="thread_context_unverified")
        if instruction_sources:
            self.failed = True
            raise ModelServiceError(
                "Codex control working directory loaded instruction files",
                error_code="inherited_codex_context",
            )
        self.thread_id = thread_id
        self.session_id = session_id
        model = result.get("model")
        if isinstance(model, str) and model.strip():
            self.resolved_model = model.strip()
        self.instruction_sources = instruction_sources
        return {
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "resolved_model": self.resolved_model,
            "app_server_info": self.app_server_info,
            "instruction_sources": list(instruction_sources),
            "native_session_id": thread.get("sessionId"),
            "ephemeral": thread.get("ephemeral"),
            "thread_start_response_model": model,
            "initialization_latency_seconds": self.initialization_latency_seconds,
        }

    @staticmethod
    def _item_type(message):
        params = message.get("params")
        item = params.get("item") if isinstance(params, dict) else None
        return item.get("type") if isinstance(item, dict) else None

    @staticmethod
    def _notification_identity(message):
        params = message.get("params")
        if not isinstance(params, dict):
            return None, None
        turn = params.get("turn")
        turn_id = turn.get("id") if isinstance(turn, dict) else params.get("turnId")
        return params.get("threadId"), turn_id

    @staticmethod
    def _event_evidence(message, item=None, stale_duplicate=False):
        params = message.get("params")
        params = params if isinstance(params, dict) else {}
        thread_id, turn_id = CodexAppServerSession._notification_identity(message)
        if thread_id is None and isinstance(params.get("thread"), dict):
            thread_id = params["thread"].get("id")
        if item is None:
            item = params.get("item")
        evidence = {"method": message.get("method")}
        for key, value in (("thread_id", thread_id), ("turn_id", turn_id)):
            if isinstance(value, str):
                evidence[key] = value
        if isinstance(item, dict):
            item_id = item.get("id")
            item_type = item.get("type")
            phase = item.get("phase")
            if isinstance(item_id, str):
                evidence["item_id"] = item_id
            if isinstance(item_type, str):
                evidence["item_type"] = item_type
            if phase is None or isinstance(phase, str):
                evidence["phase"] = phase
            if item_type == "contextCompaction":
                evidence["location"] = (
                    "turn_completed_snapshot"
                    if message.get("method") == "turn/completed"
                    else "item_lifecycle"
                )
        else:
            params_item_id = params.get("itemId")
            if isinstance(params_item_id, str):
                evidence["item_id"] = params_item_id
            if message.get("method") == "thread/compacted":
                evidence["location"] = "legacy_thread_notification"
        if stale_duplicate:
            evidence["stale_duplicate"] = True
        return evidence

    def _append_context_compaction(self, message, item=None, location=None,
                                   stale_duplicate=False):
        evidence = self._event_evidence(message, item=item)
        event = {
            "protocol_event": evidence.get("method"),
            "location": location or (
                "item_lifecycle" if evidence.get("item_type") == "contextCompaction"
                else "legacy_thread_notification"
            ),
        }
        for source, destination in (
            ("thread_id", "thread_id"), ("turn_id", "turn_id"),
            ("item_id", "item_id"),
        ):
            value = evidence.get(source)
            if isinstance(value, str):
                event[destination] = value
        if stale_duplicate:
            event["stale_duplicate"] = True
        self.context_compression_events.append(event)

    def _consume_notifications(self, expected_turn_id, deadline):
        seen_turn_started = False
        completed_turn = None
        completed_messages = {}
        tool_events = []
        event_evidence = []
        pending = self._notifications
        self._notifications = []

        def save_progress():
            self.last_turn_metadata["event_evidence"] = list(event_evidence)
            self.last_turn_metadata["tool_call_events"] = list(tool_events)
            self.last_turn_metadata["tool_call_detected"] = bool(tool_events)

        def fail(message, error_code):
            save_progress()
            self.failed = True
            raise ModelServiceError(message, error_code=error_code)

        def capture_completed_agent(item, source):
            if not isinstance(item, dict):
                fail("completed App Server agent message was malformed",
                     "app_server_turn_protocol_error")
            item_id = item.get("id")
            text = item.get("text")
            phase = item.get("phase")
            if not isinstance(item_id, str) or not item_id.strip():
                fail("completed App Server agent message had no item id",
                     "app_server_turn_protocol_error")
            if not isinstance(text, str):
                fail("completed App Server agent message had no text",
                     "app_server_turn_protocol_error")
            if phase not in (None, "commentary", "final_answer"):
                fail("App Server agent message had an unknown phase",
                     "app_server_message_phase_unknown")
            prior = completed_messages.get(item_id)
            if prior is not None and (prior["text"], prior["phase"]) != (text, phase):
                fail("App Server reused an agent message id with conflicting content or phase",
                     "conflicting_agent_message_item")
            if prior is None:
                completed_messages[item_id] = {
                    "id": item_id, "text": text, "phase": phase,
                    "sources": [source],
                }
            else:
                prior["sources"].append(source)

        def selected_final(messages):
            item_ids = sorted(row["id"] for row in messages)
            self.last_turn_metadata["final_message_item_ids"] = item_ids
            self.last_turn_metadata["final_message_candidate_count"] = len(messages)
            return messages[0]["text"]

        def prior_completed_turn_ids():
            return {
                row.get("turn_id") for row in self.turn_records
                if row.get("turn_id") and row.get("status") == "completed"
            }

        def is_stale_turn_event(message):
            thread_id, turn_id = self._notification_identity(message)
            return (
                thread_id == self.thread_id
                and turn_id != expected_turn_id
                and turn_id in prior_completed_turn_ids()
            )

        while True:
            if self.monotonic() >= deadline:
                self._notifications = pending + self._notifications
                save_progress()
                raise InferenceDeadlineExceeded(
                    "Codex App Server control turn exceeded its configured deadline",
                    error_code="inference_deadline_exceeded",
                )
            if not pending:
                try:
                    pending.append(self._read_line(deadline))
                except Exception:
                    self._notifications = pending + self._notifications
                    save_progress()
                    raise
            message = pending.pop(0)
            method = message.get("method")
            if not isinstance(method, str):
                event_evidence.append({"method": "unknown"})
                fail("unexpected App Server message during turn",
                     "app_server_turn_protocol_error")
            params = message.get("params", {})
            if not isinstance(params, dict):
                event_evidence.append(self._event_evidence(message))
                fail("App Server notification parameters were malformed",
                     "app_server_turn_protocol_error")
            if method in ("turn/started", "turn/completed") or method.startswith("item/"):
                if is_stale_turn_event(message):
                    stale_item = params.get("item")
                    if (method.startswith("item/") and isinstance(stale_item, dict)
                            and stale_item.get("type") == "contextCompaction"):
                        self._append_context_compaction(
                            message, item=stale_item, location="stale_turn_item",
                            stale_duplicate=True,
                        )
                    event_evidence.append(self._event_evidence(
                        message, stale_duplicate=True
                    ))
                    continue
            if method in ("thread/started", "thread/updated"):
                event_thread = params.get("thread", {}).get("id") if isinstance(
                    params.get("thread"), dict) else None
                if event_thread and event_thread != self.thread_id:
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server thread identity changed", "thread_identity_mismatch")
                event_evidence.append(self._event_evidence(message))
                continue
            if method == "thread/compacted":
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id:
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server context compaction thread identity did not match",
                         "thread_identity_mismatch")
                if event_turn != expected_turn_id:
                    if event_turn in prior_completed_turn_ids():
                        event_evidence.append(self._event_evidence(
                            message, stale_duplicate=True
                        ))
                        self.context_compression_events.append({
                            "protocol_event": method, "location": "legacy_thread_notification",
                            "thread_id": event_thread, "turn_id": event_turn,
                            "stale_duplicate": True,
                        })
                        continue
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server context compaction turn identity did not match",
                         "turn_identity_mismatch")
                event_evidence.append(self._event_evidence(message))
                self._append_context_compaction(message, location="legacy_thread_notification")
                continue
            if method == "turn/started":
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server turn identity did not match", "turn_identity_mismatch")
                if seen_turn_started:
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server duplicated turn/started", "duplicate_turn_event")
                seen_turn_started = True
                event_evidence.append(self._event_evidence(message))
                continue
            if method.startswith("item/"):
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server item identity did not match", "turn_identity_mismatch")
                item_type = self._item_type(message)
                item = params.get("item")
                event = self._event_evidence(message, item=item)
                event_type = item_type
                if event_type is None and method not in ("item/started", "item/completed"):
                    parts = method.split("/")
                    event_type = parts[1] if len(parts) > 1 else None
                    if isinstance(event_type, str):
                        event["item_type"] = event_type
                event_evidence.append(event)
                if event_type == "contextCompaction":
                    if (item_type == "contextCompaction"
                            and method in ("item/started", "item/completed")):
                        self._append_context_compaction(message, item=item)
                    else:
                        tool_events.append(event)
                elif event_type not in SAFE_ITEM_TYPES:
                    # Only a protocol-explicit allowlist is safe. Known tools
                    # and future/unknown item types are retained as evidence
                    # and rejected once this turn completes.
                    tool_events.append(event)
                elif method not in ("item/started", "item/completed"):
                    delta_kind = method.split("/")[-1]
                    if event_type not in SAFE_ITEM_DELTA_TYPES or delta_kind not in {
                            "delta", "summaryTextDelta", "summaryPartAdded", "textDelta"}:
                        tool_events.append(event)
                if method == "item/completed" and item_type == "agentMessage":
                    capture_completed_agent(item, "item/completed")
                continue
            if method == "turn/completed":
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    # Duplicate completion events from older turns are never
                    # candidates for this action; future/unknown turns fail closed.
                    if event_thread == self.thread_id and event_turn in {
                            row.get("turn_id") for row in self.turn_records}:
                        event_evidence.append(self._event_evidence(
                            message, stale_duplicate=True
                        ))
                        continue
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server completion identity did not match",
                         "turn_identity_mismatch")
                turn = params.get("turn")
                if not isinstance(turn, dict):
                    event_evidence.append(self._event_evidence(message))
                    fail("App Server completion had no turn object",
                         "app_server_turn_protocol_error")
                if completed_turn is not None:
                    # Repeated completion for this same turn is evidence of an
                    # ambiguous response. Never parse/execute it twice.
                    fail("App Server duplicated turn/completed", "duplicate_turn_event")
                completed_turn = turn
                turn_status = turn.get("status")
                if turn_status not in ("completed", "interrupted", "failed"):
                    event_evidence.append(self._event_evidence(message))
                    self.last_turn_metadata["turn_end_confirmed"] = False
                    self.last_turn_metadata["turn_end_state"] = "unknown"
                    fail("Codex control turn completion had an unknown status",
                         "control_turn_status_unknown")
                self.last_turn_metadata["turn_end_confirmed"] = True
                self.last_turn_metadata["turn_end_state"] = turn_status
                if turn_status != "completed":
                    event_evidence.append(self._event_evidence(message))
                    fail("Codex control turn did not complete successfully",
                         "control_turn_not_completed")
                event_evidence.append(self._event_evidence(message))
                items = turn.get("items")
                if isinstance(items, list):
                    for item in items:
                        if not isinstance(item, dict):
                            tool_events.append({
                                "method": "turn/completed_snapshot",
                                "thread_id": self.thread_id,
                                "turn_id": expected_turn_id,
                                "item_type": None,
                            })
                            continue
                        snapshot_type = item.get("type")
                        if snapshot_type == "agentMessage":
                            capture_completed_agent(item, "turn/completed_snapshot")
                        elif snapshot_type == "contextCompaction":
                            self._append_context_compaction(
                                message, item=item,
                                location="turn_completed_snapshot",
                            )
                        elif snapshot_type not in SAFE_ITEM_TYPES:
                            tool_events.append(self._event_evidence(
                                message, item=item
                            ))
                break
            if "id" in message:
                event_evidence.append(self._event_evidence(message))
                fail("unexpected App Server request during a control turn",
                     "unexpected_app_server_request")
            event_evidence.append(self._event_evidence(message))

        # Check already-buffered notifications for a duplicate completion for
        # this turn before accepting its one final message.
        still_pending = []
        for message in pending + self._notifications:
            if (message.get("method") == "turn/completed"
                    and self._notification_identity(message) ==
                    (self.thread_id, expected_turn_id)):
                event_evidence.append(self._event_evidence(message))
                fail("App Server duplicated turn/completed", "duplicate_turn_event")
            still_pending.append(message)
        self._notifications = still_pending

        if tool_events:
            self.failed = True
            return None, tool_events, event_evidence, seen_turn_started
        if not seen_turn_started:
            fail("App Server omitted turn/started identity evidence",
                 "turn_start_unconfirmed")

        explicit_final = [row for row in completed_messages.values()
                          if row["phase"] == "final_answer"]
        if explicit_final:
            unique_texts = {row["text"] for row in explicit_final}
            if len(unique_texts) > 1:
                fail("completed App Server turn had conflicting final messages",
                     "conflicting_final_messages")
            return (selected_final(explicit_final), tool_events, event_evidence,
                    seen_turn_started)

        # The installed schema marks phase as optional/null for providers that
        # do not emit it consistently. Compatibility is limited to one unique
        # completed agentMessage, and only when no explicit commentary exists;
        # this fallback is applied after the matching turn/completed event.
        unknown_phase = [row for row in completed_messages.values()
                         if row["phase"] is None]
        has_commentary = any(row["phase"] == "commentary"
                             for row in completed_messages.values())
        if len(unknown_phase) == 1 and not has_commentary:
            self.last_turn_metadata["phase_compatibility"] = (
                "single_completed_unknown_phase_message_no_commentary"
            )
            return (selected_final(unknown_phase), tool_events, event_evidence,
                    seen_turn_started)
        if not completed_messages or (not explicit_final and not unknown_phase):
            fail("completed App Server turn had no final assistant message",
                 "missing_final_message")
        fail("App Server final message phase was ambiguous",
             "ambiguous_final_message_phase")

    def _interrupt_after_timeout(self, turn_id):
        cleanup_started = self.monotonic()
        confirmed = False
        interrupt_acknowledged = False

        def consume_end_event(message):
            nonlocal confirmed
            if message.get("method") != "turn/completed":
                return False
            thread_id, completed_id = self._notification_identity(message)
            if thread_id != self.thread_id or completed_id != turn_id:
                return False
            turn = message.get("params", {}).get("turn")
            status = turn.get("status") if isinstance(turn, dict) else None
            if status in ("interrupted", "failed", "completed"):
                confirmed = True
                self.last_turn_metadata["turn_end_state"] = status
            return True

        queued = self._notifications
        self._notifications = []
        for message in queued:
            if not consume_end_event(message):
                self._notifications.append(message)
        if not confirmed:
            try:
                self._request("turn/interrupt", {
                    "threadId": self.thread_id, "turnId": turn_id,
                }, timeout=2.0)
                interrupt_acknowledged = True
            except Exception:
                # An RPC acknowledgement is useful evidence, but only the matching
                # turn/completed notification confirms that the remote turn ended.
                pass
        queued = self._notifications
        self._notifications = []
        for message in queued:
            if not consume_end_event(message):
                self._notifications.append(message)
        deadline = self.monotonic() + 2.0
        while not confirmed and self.monotonic() < deadline:
            try:
                message = self._read_line(deadline)
            except Exception:
                break
            if not consume_end_event(message) and message.get("method"):
                self._notifications.append(message)
        self.last_turn_metadata["turn_end_confirmed"] = confirmed
        self.last_turn_metadata["turn_end_state"] = (
            self.last_turn_metadata.get("turn_end_state") if confirmed else "unknown"
        )
        self.last_turn_metadata["interrupt_acknowledged"] = interrupt_acknowledged
        self.last_turn_metadata["interrupt_completion_event"] = (
            {"thread_id": self.thread_id, "turn_id": turn_id,
             "status": self.last_turn_metadata.get("turn_end_state")}
            if confirmed else None
        )
        self.last_turn_metadata["interrupt_cleanup_seconds"] = max(
            0.0, self.monotonic() - cleanup_started
        )
        if not confirmed:
            self.failed = True
            process_cleanup_started = self.monotonic()
            self.close(force=True)
            self.last_turn_metadata["process_cleanup_seconds"] = max(
                0.0, self.monotonic() - process_cleanup_started
            )
        self.last_turn_metadata["process_group_exit_confirmed"] = (
            self.process_group_exit_confirmed
        )

    def _record_turn(self, status):
        request_id = self.last_turn_metadata.get("control_request_id")
        if any(row.get("control_request_id") == request_id for row in self.turn_records):
            return
        self.turn_records.append({
            **self.last_turn_metadata,
            "status": status,
            "tool_call_detected": bool(
                self.last_turn_metadata.get("tool_call_detected", False)
            ),
        })

    def run_turn(self, text, image_paths, output_schema, request_identity=None):
        started = self.monotonic()
        deadline = started + self.timeout
        control_request_id = uuid.uuid4().hex
        identity = {
            key: (request_identity or {}).get(key)
            for key in ("step_id", "action_id", "observation_id")
        }
        # Replace metadata before validation or I/O so an early failure cannot
        # inherit a completed prior turn's IDs, events, or latency.
        self.last_turn_metadata = {
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "turn_id": None,
            "control_request_id": control_request_id,
            "request_identity": identity,
            "turn_start_attempted": False,
            "turn_start_sent": False,
            "app_server_rpc_request_id": None,
            "turn_end_confirmed": False,
            "turn_end_state": "unknown",
            "turn_started_confirmed": False,
            "tool_call_detected": False,
            "event_evidence": [],
            "tool_call_events": [],
            "phase_compatibility": None,
        }
        turn_start_submitted = False
        try:
            if self.failed or self.closed or not self.thread_id:
                raise ModelServiceError("Codex control thread is not usable",
                                        error_code="control_session_unusable")
            if not isinstance(image_paths, dict) or set(image_paths) != {
                    "front", "left_shoulder", "right_shoulder", "wrist"}:
                raise ModelServiceError("control turn requires exactly four named images",
                                        error_code="invalid_control_images")
            input_items = [{"type": "text", "text": str(text)}]
            for camera in ("front", "left_shoulder", "right_shoulder", "wrist"):
                path = Path(image_paths[camera]).resolve()
                if not path.is_file():
                    raise ModelServiceError(f"control image is missing: {camera}",
                                            error_code="control_image_missing")
                input_items.append({"type": "localImage", "path": str(path)})

            self.last_turn_metadata["turn_start_attempted"] = True
            turn_start_submitted = True
            result = self._request("turn/start", {
                "threadId": self.thread_id,
                "input": input_items,
                "cwd": str(self.cwd),
                "model": self.model,
                "effort": self.reasoning_effort,
                "approvalPolicy": "never",
                "outputSchema": output_schema,
            }, deadline=deadline)
            turn = result.get("turn")
            turn_id = turn.get("id") if isinstance(turn, dict) else None
            if not isinstance(turn_id, str) or not turn_id.strip():
                raise ModelServiceError("turn/start did not return a turn id",
                                        error_code="turn_identity_missing")
            if any(row.get("turn_id") == turn_id for row in self.turn_records):
                raise ModelServiceError("turn/start reused a prior turn id",
                                        error_code="turn_identity_reused")
            self.last_turn_metadata.update({
                "turn_id": turn_id,
                "turn_start_status": turn.get("status"),
            })
            self.turn_count += 1
            final_text, tool_events, events, turn_started = self._consume_notifications(
                turn_id, deadline
            )
            self.last_turn_metadata.update({
                "turn_end_confirmed": True,
                "turn_end_state": "completed",
                "turn_started_confirmed": turn_started,
                "tool_call_detected": bool(tool_events),
                "tool_call_events": tool_events,
                "event_evidence": events,
                "latency_seconds": max(0.0, self.monotonic() - started),
            })
            self._record_turn("rejected" if tool_events else "completed")
            if tool_events:
                raise PolicyToolViolation(
                    "Codex control turn used an App Server tool or unknown item type; "
                    "the episode session is closed",
                    error_code="tool_event_detected",
                )
            return final_text, dict(self.last_turn_metadata)
        except InferenceDeadlineExceeded as exc:
            turn_elapsed = max(0.0, self.monotonic() - started)
            if self.last_turn_metadata.get("turn_id"):
                self._interrupt_after_timeout(self.last_turn_metadata["turn_id"])
            else:
                self.last_turn_metadata["turn_end_confirmed"] = False
                self.last_turn_metadata["turn_end_state"] = "unknown"
                if turn_start_submitted:
                    process_cleanup_started = self.monotonic()
                    self.close(force=True)
                    self.last_turn_metadata["process_cleanup_seconds"] = max(
                        0.0, self.monotonic() - process_cleanup_started
                    )
                    self.last_turn_metadata["process_group_exit_confirmed"] = (
                        self.process_group_exit_confirmed
                    )
            self.failed = True
            self.last_turn_metadata.update({
                "status": "timeout",
                "latency_seconds": turn_elapsed,
            })
            self._record_turn("timeout")
            raise InferenceDeadlineExceeded(
                "Codex App Server control turn exceeded its configured deadline",
                error_code="inference_deadline_exceeded",
            ) from exc
        except Exception:
            self.failed = True
            turn_elapsed = max(0.0, self.monotonic() - started)
            if turn_start_submitted and not self.last_turn_metadata.get("turn_end_confirmed"):
                # Until an end event is confirmed, the local controller will
                # never accept late output from this process/session.
                self.last_turn_metadata["turn_end_state"] = "unknown"
                process_cleanup_started = self.monotonic()
                self.close(force=True)
                self.last_turn_metadata["process_cleanup_seconds"] = max(
                    0.0, self.monotonic() - process_cleanup_started
                )
                self.last_turn_metadata["process_group_exit_confirmed"] = (
                    self.process_group_exit_confirmed
                )
            self.last_turn_metadata.setdefault("status", "failed")
            self.last_turn_metadata.setdefault("latency_seconds", turn_elapsed)
            self._record_turn(self.last_turn_metadata["status"])
            raise

    def close(self, force=False):
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is None:
            self.process_group_exit_confirmed = True
            return
        if not force and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=2.0)
            except Exception:
                force = True

        def group_exists():
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        def signal_group(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            except Exception:
                pass

        if force or group_exists():
            signal_group(signal.SIGTERM)
            deadline = time.monotonic() + 2.0
            while group_exists() and time.monotonic() < deadline:
                process.poll()  # reap the owned leader before checking its group
                time.sleep(0.05)
            if group_exists():
                signal_group(signal.SIGKILL)
                deadline = time.monotonic() + 2.0
                while group_exists() and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(0.05)
        try:
            process.wait(timeout=2.0)
        except Exception:
            pass
        self.process_group_exit_confirmed = not group_exists()
        try:
            if self.selector is not None:
                self.selector.close()
        except Exception:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
