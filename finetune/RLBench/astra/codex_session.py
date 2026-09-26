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
from pathlib import Path

from .errors import (
    InferenceDeadlineExceeded,
    ModelServiceError,
    PolicyToolViolation,
    sanitize_diagnostic,
)


SAFE_ITEM_TYPES = {
    "agentMessage", "reasoning", "plan", "userMessage",
    "developerMessage", "systemMessage",
}


class CodexAppServerSession:
    """One App Server process and one newly created native thread."""

    def __init__(self, executable, cwd, model, reasoning_effort, timeout,
                 process_factory=subprocess.Popen):
        self.executable = str(executable)
        self.cwd = Path(cwd).resolve()
        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout = float(timeout)
        self.process_factory = process_factory
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
            remaining = deadline - time.monotonic()
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

    def _request(self, method, params, timeout=None):
        request_id = self._next_request_id
        self._next_request_id += 1
        self._send({"method": method, "id": request_id, "params": params})
        self.rpc_request_count += 1
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        while True:
            message = self._read_line(deadline)
            if message.get("id") == request_id:
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

    def _consume_notifications(self, expected_turn_id, deadline):
        seen_turn_started = False
        completed_turn = None
        assistant_messages = []
        tool_events = []
        event_evidence = []
        pending = self._notifications
        self._notifications = []
        while True:
            if not pending:
                pending.append(self._read_line(deadline))
            message = pending.pop(0)
            method = message.get("method")
            if not isinstance(method, str):
                self.failed = True
                raise ModelServiceError("unexpected App Server message during turn",
                                        error_code="app_server_turn_protocol_error")
            params = message.get("params", {})
            if not isinstance(params, dict):
                self.failed = True
                raise ModelServiceError("App Server notification parameters were malformed",
                                        error_code="app_server_turn_protocol_error")
            if "compact" in method.lower() or "context" in method.lower():
                self.context_compression_events.append({
                    "method": method, "turn_id": expected_turn_id,
                })
            event = {"method": method}
            if method in ("thread/started", "thread/updated"):
                event_thread = params.get("thread", {}).get("id") if isinstance(
                    params.get("thread"), dict) else None
                if event_thread and event_thread != self.thread_id:
                    self.failed = True
                    raise ModelServiceError("App Server thread identity changed",
                                            error_code="thread_identity_mismatch")
                event_evidence.append(event)
                continue
            if method == "turn/started":
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    self.failed = True
                    raise ModelServiceError("App Server turn identity did not match",
                                            error_code="turn_identity_mismatch")
                if seen_turn_started:
                    self.failed = True
                    raise ModelServiceError("App Server duplicated turn/started",
                                            error_code="duplicate_turn_event")
                seen_turn_started = True
                event_evidence.append(event)
                continue
            if method.startswith("item/"):
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    self.failed = True
                    raise ModelServiceError("App Server item identity did not match",
                                            error_code="turn_identity_mismatch")
                item_type = self._item_type(message)
                event["item_type"] = item_type if isinstance(item_type, str) else None
                event_evidence.append(event)
                method_lower = method.lower()
                method_names_tool = any(token in method_lower for token in (
                    "commandexecution", "filechange", "mcptool", "websearch",
                    "functioncall", "toolcall", "shellcommand",
                ))
                if ((item_type is not None and item_type not in SAFE_ITEM_TYPES)
                        or method_names_tool):
                    tool_events.append(event)
                if method == "item/completed" and item_type == "agentMessage":
                    item = params.get("item")
                    text = item.get("text") if isinstance(item, dict) else None
                    if isinstance(text, str):
                        assistant_messages.append(text)
                continue
            if method == "turn/completed":
                event_thread, event_turn = self._notification_identity(message)
                if event_thread != self.thread_id or event_turn != expected_turn_id:
                    # Duplicate completion events from older turns are never
                    # candidates for this action; future/unknown turns fail closed.
                    if event_thread == self.thread_id and event_turn in {
                            row.get("turn_id") for row in self.turn_records}:
                        event_evidence.append({"method": method, "stale_duplicate": True})
                        continue
                    self.failed = True
                    raise ModelServiceError("App Server completion identity did not match",
                                            error_code="turn_identity_mismatch")
                turn = params.get("turn")
                if not isinstance(turn, dict):
                    self.failed = True
                    raise ModelServiceError("App Server completion had no turn object",
                                            error_code="app_server_turn_protocol_error")
                if completed_turn is not None:
                    # Repeated completion for this same turn is evidence of an
                    # ambiguous response. Never parse/execute it twice.
                    self.failed = True
                    raise ModelServiceError("App Server duplicated turn/completed",
                                            error_code="duplicate_turn_event")
                completed_turn = turn
                items = turn.get("items")
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and item.get("type") == "agentMessage":
                            text = item.get("text")
                            if isinstance(text, str):
                                assistant_messages.append(text)
                if turn.get("status") != "completed":
                    self.failed = True
                    raise ModelServiceError(
                        "Codex control turn did not complete successfully",
                        error_code="control_turn_not_completed",
                    )
                break
            if "id" in message:
                self.failed = True
                raise ModelServiceError(
                    "unexpected App Server request during a control turn",
                    error_code="unexpected_app_server_request",
                )
            event_evidence.append(event)

        # Check already-buffered notifications for a duplicate completion for
        # this turn before accepting its one final message.
        still_pending = []
        for message in pending + self._notifications:
            if (message.get("method") == "turn/completed"
                    and self._notification_identity(message) ==
                    (self.thread_id, expected_turn_id)):
                self.failed = True
                raise ModelServiceError("App Server duplicated turn/completed",
                                        error_code="duplicate_turn_event")
            still_pending.append(message)
        self._notifications = still_pending

        if tool_events:
            self.failed = True
            return None, tool_events, event_evidence, seen_turn_started
        if not seen_turn_started:
            self.failed = True
            raise ModelServiceError("App Server omitted turn/started identity evidence",
                                    error_code="turn_start_unconfirmed")
        if not assistant_messages:
            self.failed = True
            raise ModelServiceError("completed App Server turn had no final assistant message",
                                    error_code="missing_final_message")
        return assistant_messages[-1], tool_events, event_evidence, seen_turn_started

    def _interrupt_after_timeout(self, turn_id):
        confirmed = False
        try:
            self._request("turn/interrupt", {
                "threadId": self.thread_id, "turnId": turn_id,
            }, timeout=2.0)
            queued = self._notifications
            self._notifications = []
            for message in queued:
                if message.get("method") == "turn/completed":
                    thread_id, completed_id = self._notification_identity(message)
                    if thread_id == self.thread_id and completed_id == turn_id:
                        status = message.get("params", {}).get("turn", {}).get("status")
                        confirmed = status in ("interrupted", "failed", "completed")
                        if confirmed:
                            break
                self._notifications.append(message)
            deadline = time.monotonic() + 2.0
            while not confirmed and time.monotonic() < deadline:
                message = self._read_line(deadline)
                if message.get("method") == "turn/completed":
                    thread_id, completed_id = self._notification_identity(message)
                    if thread_id == self.thread_id and completed_id == turn_id:
                        status = message.get("params", {}).get("turn", {}).get("status")
                        confirmed = status in ("interrupted", "failed", "completed")
                        break
                elif message.get("method"):
                    self._notifications.append(message)
        except Exception:
            confirmed = False
        self.last_turn_metadata["turn_end_confirmed"] = confirmed
        if not confirmed:
            self.failed = True
            self.close(force=True)

    def run_turn(self, text, image_paths, output_schema):
        if self.failed or self.closed or not self.thread_id:
            raise ModelServiceError("Codex control thread is not usable",
                                    error_code="control_session_unusable")
        if set(image_paths) != {"front", "left_shoulder", "right_shoulder", "wrist"}:
            self.failed = True
            raise ModelServiceError("control turn requires exactly four named images",
                                    error_code="invalid_control_images")
        input_items = [{"type": "text", "text": str(text)}]
        for camera in ("front", "left_shoulder", "right_shoulder", "wrist"):
            path = Path(image_paths[camera]).resolve()
            if not path.is_file():
                self.failed = True
                raise ModelServiceError(f"control image is missing: {camera}",
                                        error_code="control_image_missing")
            input_items.append({"type": "localImage", "path": str(path)})

        started = time.monotonic()
        result = self._request("turn/start", {
            "threadId": self.thread_id,
            "input": input_items,
            "cwd": str(self.cwd),
            "model": self.model,
            "effort": self.reasoning_effort,
            "approvalPolicy": "never",
            "outputSchema": output_schema,
        })
        turn = result.get("turn")
        turn_id = turn.get("id") if isinstance(turn, dict) else None
        if not isinstance(turn_id, str) or not turn_id.strip():
            self.failed = True
            raise ModelServiceError("turn/start did not return a turn id",
                                    error_code="turn_identity_missing")
        self.turn_count += 1
        self.last_turn_metadata = {
            "thread_id": self.thread_id,
            "session_id": self.session_id,
            "turn_id": turn_id,
            "turn_start_status": turn.get("status"),
            "turn_end_confirmed": False,
        }
        try:
            final_text, tool_events, events, turn_started = self._consume_notifications(
                turn_id, time.monotonic() + self.timeout
            )
        except InferenceDeadlineExceeded:
            self._interrupt_after_timeout(turn_id)
            self.failed = True
            self.turn_records.append({
                **self.last_turn_metadata,
                "status": "timeout",
                "tool_call_detected": False,
                "latency_seconds": time.monotonic() - started,
            })
            raise InferenceDeadlineExceeded(
                "Codex App Server control turn exceeded its configured timeout",
                error_code="inference_deadline_exceeded",
            )
        except Exception:
            self.failed = True
            raise
        self.last_turn_metadata["turn_end_confirmed"] = True
        self.last_turn_metadata["turn_started_confirmed"] = turn_started
        self.last_turn_metadata["tool_call_detected"] = bool(tool_events)
        self.last_turn_metadata["tool_call_events"] = tool_events
        self.last_turn_metadata["event_evidence"] = events
        self.last_turn_metadata["latency_seconds"] = time.monotonic() - started
        self.turn_records.append({
            **{key: value for key, value in self.last_turn_metadata.items()
               if key not in ("event_evidence", "tool_call_events")},
            "status": "completed",
            "tool_call_detected": bool(tool_events),
        })
        if tool_events:
            raise PolicyToolViolation(
                "Codex control turn used an App Server tool; the episode session is closed",
                error_code="tool_event_detected",
            )
        return final_text, dict(self.last_turn_metadata)

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
