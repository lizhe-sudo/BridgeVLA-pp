"""ChatGPT-authenticated Codex App Server policy for Astra RLBench evaluation."""

import json
import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .errors import (
    AstraEvaluationError,
    CoreArtifactWriteError,
    InferenceDeadlineExceeded,
    InvalidPolicyOutput,
    ModelServiceError,
    PolicyToolViolation,
    SimulatorInfrastructureError,
    MAX_REJECTED_OUTPUT_BYTES,
    safe_exception_record,
    sanitize_diagnostic,
)
from .codex_session import CodexAppServerSession
from .policy import AstraPolicy
from .schemas import AstraAction, AstraObservation

REPO_ROOT = Path(__file__).resolve().parents[3]


class CodexAstraPolicyError(InvalidPolicyOutput):
    """Backward-compatible name for invalid Codex policy output."""


class CodexAstraPolicy(AstraPolicy):
    """Run one native Codex App Server thread for each task episode."""

    IMAGE_FIELDS = ("front", "left_shoulder", "right_shoulder", "wrist")
    SAFE_ITEM_TYPES = {
        "agentMessage",
        "reasoning",
        "plan",
        "userMessage",
        "developerMessage",
        "systemMessage",
        "contextCompaction",
    }
    TOOL_EVENT_TYPES = {
        "command_execution",
        "file_change",
        "mcp_tool_call",
        "web_search_call",
        "tool_call",
        "function_call",
        "shell",
        "shell_command",
    }
    CONTEXT_FILES = ("AGENTS.md", "AGENTS.override.md", ".codex/config.toml")

    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "max",
        timeout: float = 180.0,
        work_root: Optional[str] = None,
        collision_mode: str = "fixed0",
    ):
        if not model:
            raise ValueError("Codex model must be non-empty")
        if not reasoning_effort:
            raise ValueError("Codex reasoning effort must be non-empty")
        if not math.isfinite(float(timeout)) or float(timeout) <= 0:
            raise ValueError("Codex timeout must be positive and finite")

        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        if collision_mode not in ("fixed0", "fixed1", "predict"):
            raise ValueError("collision_mode must be fixed0, fixed1, or predict")
        self.collision_mode = collision_mode
        self.timeout = float(timeout)
        if work_root is None:
            raise ValueError(
                "work_root must be the current run's policy_work directory"
            )
        self.work_root = Path(work_root).expanduser().absolute()
        try:
            self.work_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CoreArtifactWriteError(
                "failed to create persistent Codex policy work directory"
            ) from exc
        self.last_metadata = None
        self._instruction = ""
        self._step_index = 0
        self._episode_dir = None
        self._evaluation_context = {}
        self._active_metadata = None
        self._session = None
        self._control_work_dir = None
        self._session_manifest_path = None
        self._control_messages_path = None
        self._pending_feedback = None
        self._last_action_binding = None
        self._feedback_recorded_action_id = None
        self._session_started_at = None
        self._session_termination_reason = None
        self._transient_final_text = None
        self.codex_cli_version_probe_invocation_count = 0
        self.codex_cli_version = self._read_cli_version()

    def _read_cli_version(self):
        codex = shutil.which("codex")
        if codex is None:
            return None
        self.codex_cli_version_probe_invocation_count = 1
        try:
            result = subprocess.run(
                [codex, "--version"], capture_output=True, text=True,
                timeout=5, check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    def set_evaluation_context(self, evaluation_id, repeat_id, task, episode_id,
                               waypoint_budget=None):
        self._evaluation_context = {
            "evaluation_id": evaluation_id,
            "repeat_id": repeat_id,
            "task": task,
            "episode_id": episode_id,
            "waypoint_budget": waypoint_budget,
        }

    def reset(self, instruction: Optional[str] = None) -> None:
        if self._session is not None:
            self.end_episode("episode_reset_without_explicit_close")
        self._instruction = str(instruction or "")
        self._step_index = 0
        self._pending_feedback = None
        self._last_action_binding = None
        self._feedback_recorded_action_id = None
        self._session = None
        self._session_termination_reason = None
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self._episode_dir = self.work_root / f"{run_id}_{uuid.uuid4().hex[:8]}"
        try:
            self._episode_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise CoreArtifactWriteError(
                "failed to create persistent Codex episode directory"
            ) from exc
        control_root = Path(tempfile.gettempdir()) / "astra_codex_control"
        try:
            control_root.mkdir(parents=True, exist_ok=True)
            self._control_work_dir = Path(tempfile.mkdtemp(
                prefix="episode_", dir=str(control_root)
            )).resolve()
        except OSError as exc:
            raise CoreArtifactWriteError(
                "failed to create isolated Codex control working directory"
            ) from exc
        self._session_manifest_path = self._episode_dir / "session_manifest.json"
        self._control_messages_path = self._episode_dir / "control_messages.jsonl"
        self._session_started_at = None
        self._write_session_manifest("not_created")
        self.last_metadata = None

    def _session_manifest(self, status):
        return {
            "episode_identity": {
                key: self._evaluation_context.get(key)
                for key in ("evaluation_id", "repeat_id", "task", "episode_id")
            },
            "control_session_id": (
                self._session.session_id if self._session is not None else None
            ),
            "thread_id": self._session.thread_id if self._session is not None else None,
            "backend": "codex_app_server",
            "codex_cli_version": self.codex_cli_version,
            "model_requested": self.model,
            "resolved_model": (
                self._session.resolved_model if self._session is not None else None
            ),
            "app_server_info": (
                self._session.app_server_info if self._session is not None else None
            ),
            "reasoning_effort": self.reasoning_effort,
            "created_at": self._session_started_at,
            "initialization_latency_seconds": (
                self._session.initialization_latency_seconds
                if self._session is not None else None
            ),
            "status": status,
            "session_creation_count": int(
                self._session is not None and self._session.thread_id is not None
            ),
            "session_resume_count": 0,
            "app_server_rpc_request_count": (
                self._session.rpc_request_count if self._session is not None else 0
            ),
            "control_turn_count": (
                self._session.turn_count if self._session is not None else 0
            ),
            "process_group_exit_confirmed": (
                self._session.process_group_exit_confirmed
                if self._session is not None else None
            ),
            "native_storage": "Codex App Server managed storage; not copied into application logs",
            "native_storage_root": str(
                Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex")).expanduser()
            ),
            "working_directory": str(self._control_work_dir) if self._control_work_dir else None,
            "instruction_sources": (
                self._session.instruction_sources if self._session is not None else None
            ),
            "turns": list(self._session.turn_records) if self._session is not None else [],
            "context_compression_events": (
                list(self._session.context_compression_events)
                if self._session is not None and self._session.context_compression_events
                else None
            ),
            "termination_reason": self._session_termination_reason,
        }

    def _write_session_manifest(self, status):
        if self._session_manifest_path is None:
            return
        self._write_json(self._session_manifest_path, self._session_manifest(status))

    def _append_control_record(self, record):
        if self._control_messages_path is None:
            raise CoreArtifactWriteError("control message log is not initialized")
        try:
            with self._control_messages_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False,
                                        allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            raise CoreArtifactWriteError("failed to persist control message record") from exc

    def _bind_session_turn(self, step_id, action_id, observation_id,
                           turn_metadata, status):
        if self._session is None:
            return
        turn_id = turn_metadata.get("turn_id")
        request_identity = turn_metadata.get("request_identity") or {}
        expected_identity = {
            "step_id": step_id,
            "action_id": action_id,
            "observation_id": observation_id,
        }
        if request_identity != expected_identity:
            raise ModelServiceError(
                "Codex turn metadata did not match the active application action",
                error_code="control_request_identity_mismatch",
            )
        control_request_id = turn_metadata.get("control_request_id")
        if not isinstance(control_request_id, str) or not control_request_id:
            raise ModelServiceError(
                "Codex turn metadata had no application request identity",
                error_code="control_request_identity_missing",
            )
        binding = {
            "control_request_id": control_request_id,
            "step_id": step_id,
            "observation_id": observation_id,
            "action_id": action_id,
            "turn_id": turn_id,
            "status": status,
        }
        matches = [row for row in self._session.turn_records
                   if row.get("control_request_id") == control_request_id]
        if len(matches) > 1:
            raise ModelServiceError(
                "Codex control request identity was recorded more than once",
                error_code="duplicate_control_request_record",
            )
        if not matches:
            # This request failed before the session could persist a turn
            # record. Keep the failure attached to its own null-ID request.
            self._session.turn_records.append({
                **binding,
                "thread_id": self._session.thread_id,
                "session_id": self._session.session_id,
                "turn_end_confirmed": False,
                "turn_end_state": "unknown",
                "request_identity": expected_identity,
            })
            return
        record = matches[0]
        if record.get("turn_id") != turn_id:
            raise ModelServiceError(
                "Codex control request record turn id changed during binding",
                error_code="control_turn_record_mismatch",
            )
        record.update(binding)

    def end_episode(self, termination_reason="episode_finished"):
        """Release this episode's App Server process and finalize its manifest."""
        self._session_termination_reason = str(termination_reason)
        if self._session is not None:
            self._session.close()
            status = "failed" if self._session.failed else "closed"
            self._write_session_manifest(status)
            self._session = None
        else:
            self._write_session_manifest("not_created" if not self._step_index else "failed")
        if self._control_work_dir is not None:
            try:
                self._control_work_dir.rmdir()
            except OSError:
                # The App Server may create harmless cwd files; retain the
                # stable episode directory rather than deleting unknown data.
                pass
            self._control_work_dir = None

    @staticmethod
    def _schema():
        return CodexAstraPolicy._schema_for_mode("fixed0")

    @staticmethod
    def _schema_for_mode(collision_mode):
        properties = {
            "position": {
                "type": "array", "items": {"type": "number"},
                "minItems": 3, "maxItems": 3,
            },
            "quaternion": {
                "type": "array", "items": {"type": "number"},
                "minItems": 4, "maxItems": 4,
            },
            "gripper": {"type": "integer", "enum": [0, 1]},
        }
        required = ["position", "quaternion", "gripper"]
        if collision_mode == "predict":
            properties["ignore_collisions"] = {"type": "integer", "enum": [0, 1]}
            required.append("ignore_collisions")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    @staticmethod
    def _image_as_uint8(image, name):
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise SimulatorInfrastructureError(
                f"{name} RGB image must be HxWx3, got {array.shape}"
            )
        if array.dtype == np.uint8:
            return array
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise SimulatorInfrastructureError(f"{name} RGB image is non-finite")
            if array.size and float(array.max()) <= 1.0 and float(array.min()) >= 0.0:
                array = array * 255.0
            elif array.size and (float(array.min()) < 0.0 or float(array.max()) > 255.0):
                raise SimulatorInfrastructureError(
                    f"{name} RGB image values must be in [0,1] or [0,255]"
                )
            return np.rint(array).clip(0, 255).astype(np.uint8)
        if np.issubdtype(array.dtype, np.integer):
            return array.clip(0, 255).astype(np.uint8)
        raise SimulatorInfrastructureError(
            f"{name} RGB image has unsupported dtype {array.dtype}"
        )

    @staticmethod
    def _number_list(value, length, field, error_type=InvalidPolicyOutput):
        if not isinstance(value, list) or len(value) != length:
            raise error_type(
                f"{field} must contain exactly {length} numbers",
                error_code=f"invalid_{field}_shape",
            )
        result = []
        for component in value:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise error_type(
                    f"{field} must contain only numbers",
                    error_code=f"invalid_{field}_type",
                )
            component = float(component)
            if not math.isfinite(component):
                raise error_type(
                    f"{field} contains a non-finite number",
                    error_code=f"non_finite_{field}",
                )
            result.append(component)
        return result

    @classmethod
    def _tool_events(cls, event_text):
        if not event_text.strip():
            raise ModelServiceError(
                "Codex event stream is empty", error_code="empty_event_stream"
            )
        found = []
        for line_number, line in enumerate(event_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ModelServiceError(
                    f"Codex emitted invalid JSON event on line {line_number}",
                    error_code="invalid_event_json",
                ) from exc
            if not isinstance(event, dict):
                raise ModelServiceError(
                    f"Codex event on line {line_number} is not a JSON object"
                )
            event_type = cls._safe_event_type(event.get("type", ""))
            if event_type in cls.TOOL_EVENT_TYPES or "tool_call" in event_type:
                found.append({"line": line_number, "event_type": event_type})
            item = event.get("item")
            if isinstance(item, dict):
                item_type = cls._safe_event_type(item.get("type", ""))
                if item_type in cls.TOOL_EVENT_TYPES or "tool_call" in item_type:
                    found.append({"line": line_number, "item_type": item_type})
                elif item_type not in cls.SAFE_ITEM_TYPES:
                    found.append({"line": line_number, "item_type": item_type})
        return found

    @classmethod
    def _safe_event_log(cls, event_text):
        """Persist protocol identity and event types, never arbitrary payloads."""
        records = []
        lines = event_text.splitlines()
        max_lines = 512
        for line_number, line in enumerate(lines[:max_lines], 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                records.append({"line": line_number, "parse_error": True})
                continue
            if not isinstance(event, dict):
                records.append({"line": line_number, "non_object_event": True})
                continue
            record = {
                "line": line_number,
                "type": cls._safe_event_type(event.get("type", "unknown")),
            }
            for key in ("thread_id", "turn_id", "item_id", "location"):
                value = event.get(key)
                if isinstance(value, str) and len(value) <= 256:
                    record[key] = value
            phase = event.get("phase")
            if phase in (None, "commentary", "final_answer"):
                record["phase"] = phase
            if event.get("stale_duplicate") is True:
                record["stale_duplicate"] = True
            item = event.get("item")
            if isinstance(item, dict):
                record["item_type"] = cls._safe_event_type(item.get("type", "unknown"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                record["usage"] = {
                    key: value for key, value in usage.items()
                    if key in ("input_tokens", "output_tokens", "reasoning_output_tokens")
                    and isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0
                }
            records.append(record)
        if len(lines) > max_lines:
            records.append({"truncated": True, "omitted_event_count": len(lines) - max_lines})
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)

    @staticmethod
    def _safe_event_type(value):
        value = str(value or "")
        if len(value) > 80 or any(
            not (char.isalnum() or char in "_./-") for char in value
        ):
            return "unrecognized"
        return value or "unknown"

    @staticmethod
    def _event_evidence_text(events):
        lines = []
        for event in events:
            if not isinstance(event, dict):
                continue
            row = {"type": event.get("method", "unknown")}
            for key in ("thread_id", "turn_id", "item_id", "location"):
                value = event.get(key)
                if isinstance(value, str):
                    row[key] = value
            if isinstance(event.get("item_type"), str):
                row["item"] = {"type": event["item_type"]}
            if event.get("phase") in (None, "commentary", "final_answer"):
                row["phase"] = event.get("phase")
            if event.get("stale_duplicate") is True:
                row["stale_duplicate"] = True
            lines.append(json.dumps(row, ensure_ascii=False))
        return "".join(line + "\n" for line in lines)

    @staticmethod
    def _token_usage(event_text):
        usage = None
        for line in event_text.splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
        return usage

    @staticmethod
    def _write_json(path, value):
        temp_path = path.with_name(path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise CoreArtifactWriteError(
                f"failed to persist policy artifact {path.name}"
            ) from exc

    @staticmethod
    def _write_text_atomic(path, text):
        temp_path = path.with_name(path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise CoreArtifactWriteError(
                f"failed to persist policy artifact {path.name}"
            ) from exc

    @classmethod
    def _write_event_evidence(cls, path, event_text, metadata):
        safe_text = cls._safe_event_log(event_text)
        encoded = safe_text.encode("utf-8")
        truncated = len(encoded) > 65536
        if truncated:
            safe_text = encoded[:65536].decode("utf-8", errors="ignore")
        cls._write_text_atomic(path, safe_text)
        metadata["event_evidence_truncated"] = truncated
        metadata["event_line_count"] = len(event_text.splitlines())

    @classmethod
    def _write_stderr_evidence(cls, path, stderr_text, metadata):
        evidence = sanitize_diagnostic(stderr_text)
        cls._write_text_atomic(path, evidence["text"])
        metadata["stderr_evidence"] = {
            "present": bool(stderr_text),
            "original_chars": evidence["original_chars"],
            "truncated": evidence["truncated"],
            "redaction_failed": evidence["redaction_failed"],
            "saved_chars": len(evidence["text"]),
        }
        metadata["stderr_summary"] = evidence["text"]

    def _save_metadata(self, step_dir, metadata):
        if self._active_metadata is not None:
            self._active_metadata.update(metadata)
            metadata = self._active_metadata
        self.last_metadata = metadata
        self._write_json(step_dir / "metadata.json", metadata)

    @classmethod
    def _rejected_output_evidence(cls, raw_path, destination, reason_code):
        """Save allowlisted action fields; never persist arbitrary model text."""
        evidence = {
            "status": "rejected",
            "reason_code": str(reason_code),
            "raw_content_saved": False,
            "byte_length": None,
            "truncated": False,
            "parseable_json": False,
        }
        try:
            raw_file = Path(raw_path)
            byte_length = raw_file.stat().st_size
            with raw_file.open("rb") as stream:
                raw = stream.read(MAX_REJECTED_OUTPUT_BYTES + 1)
        except FileNotFoundError:
            evidence["status"] = "no_output_file"
            cls._write_json(destination, evidence)
            return evidence
        except Exception:
            evidence["status"] = "output_unreadable"
            cls._write_json(destination, evidence)
            return evidence

        evidence["byte_length"] = byte_length
        if byte_length <= MAX_REJECTED_OUTPUT_BYTES:
            evidence["sha256"] = hashlib.sha256(raw).hexdigest()
        truncated = byte_length > MAX_REJECTED_OUTPUT_BYTES
        evidence["truncated"] = truncated
        try:
            decoded = raw[:MAX_REJECTED_OUTPUT_BYTES].decode("utf-8")
            action = json.loads(decoded) if not truncated else None
        except Exception:
            action = None
        if isinstance(action, dict):
            safe_fields = {}
            for field in ("position", "quaternion", "gripper", "ignore_collisions"):
                if field not in action:
                    continue
                value = action[field]
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    safe_fields[field] = value if math.isfinite(float(value)) else str(value)
                elif isinstance(value, list) and len(value) <= 8:
                    safe_fields[field] = [
                        item if isinstance(item, (int, float)) and not isinstance(item, bool)
                        and (not isinstance(item, float) or math.isfinite(item))
                        else {"type": type(item).__name__}
                        for item in value
                    ]
                else:
                    safe_fields[field] = {"type": type(value).__name__}
            evidence["parseable_json"] = True
            evidence["action_fields"] = safe_fields
            evidence["unexpected_field_count"] = sum(
                field not in ("position", "quaternion", "gripper", "ignore_collisions")
                for field in action
            )
        cls._write_json(destination, evidence)
        return evidence

    def _make_prompt(self, observation, pose, step_id):
        instruction = str(observation.instruction or self._instruction)
        gripper_text = "open" if observation.gripper_open else "closed"
        identity = self._evaluation_context
        remaining = max(0, int(identity.get("waypoint_budget") or 25)
                        - int(step_id))
        observation_id = self._observation_id(step_id)
        camera_lines = "\n".join(
            f"- {camera}: observation_id={observation_id}"
            for camera in self.IMAGE_FIELDS
        )
        current_state = (
            f"Task instruction: {instruction}\n"
            f"Current step_id: {int(step_id)}\n"
            f"Remaining target-action budget: {remaining}\n"
            f"Current observation_id: {observation_id}\n"
            f"Measured EEF pose XYZ meters: {json.dumps(pose[:3])}\n"
            f"Measured EEF quaternion XYZW: {json.dumps(pose[3:7])}\n"
            f"Measured gripper state: {gripper_text}\n"
            "The four newly attached RGB images are the CURRENT observation, in this order:\n"
            f"{camera_lines}\n"
        )
        if int(step_id) == 0:
            return (
                "You control the end-effector of a Franka Panda robot in RLBench.\n"
                "This conversation belongs to exactly one task episode.\n\n"
                "At each turn, output exactly one next absolute end-effector target. "
                "The evaluator executes that target and then provides the next observation "
                "and measured execution feedback in this same conversation.\n\n"
                "Use the CURRENT MEASURED pose as the starting point. A previously "
                "requested target is not evidence that the robot reached it. When older "
                "assumptions conflict with new measurements, use the new measurements.\n\n"
                "Positions are absolute world-frame coordinates in meters. Orientations "
                "are absolute unit quaternions in XYZW order. The pose refers to the "
                "RLBench arm-tip reference used by the existing action mode. The arm "
                "moves first; the gripper command is applied after the arm motion. "
                "Gripper command: 0 closes, 1 opens.\n"
                f"{self._collision_prompt()}\n\n"
                "This is the initial state; there is no action-outcome experience from "
                "this episode. Prefer a small exploratory or approach movement in a "
                "direction with visible clearance. Aim for an initial translation of "
                "approximately 1 cm or less. Prefer to keep the current orientation and "
                "gripper state during this initial position-to-image exploration. This "
                "is behavioral guidance, not a guarantee of collision-free motion; do "
                "not assume that moving upward or along a fixed world axis is always safe.\n\n"
                "On later turns, use the conversation history together with measured "
                "execution feedback and newest images to assess how requested movements "
                "actually affected the robot. When direction, scale, or clearance is "
                "uncertain, continue with small movements. When recent movements "
                "reasonably track requests and there is visible clearance, you may choose "
                "a somewhat larger next movement. Reduce the movement size again near "
                "the drawer, handle, cabinet surfaces, or narrow contact regions. Avoid "
                "combining a large translation, a large rotation, and a gripper-state "
                "change during an uncertain approach.\n\n"
                "If execution returned but the target was not reached, do not treat that "
                "as successful motion. Inspect the feedback and current images. Do not "
                "simply request a larger motion farther in the same direction. A "
                "discrepancy may suggest obstruction or tracking error, but does not "
                "prove a collision. The wrist camera moves with the robot; account for "
                "that motion when comparing images over time.\n\n"
                "Follow the task instruction to select the correct drawer. Do not assume "
                "a fixed drawer level, handle coordinate, or pulling axis. Choose all "
                "exploratory, approach, repositioning, and manipulation targets yourself. "
                "The evaluator does not insert intermediate movements or recovery skills. "
                "Every submitted target consumes one action from the remaining budget.\n\n"
                "Use only the task instruction, images, measured robot state, your own "
                "prior control messages, and actual execution feedback supplied in this "
                "conversation. Do not inspect files, source code, demonstrations, or "
                "external information. Do not use tools.\n\n"
                f"{current_state}\n"
                "Return only the structured action required by the output schema."
            )

        if self._pending_feedback is None:
            raise ModelServiceError(
                "a later control turn has no completed-action feedback",
                error_code="missing_execution_feedback",
            )
        return (
            "Continue the SAME task episode and control conversation.\n\n"
            f"Current step: {int(step_id)}\n"
            f"Remaining target-action budget: {remaining}\n\n"
            "Measured feedback for the immediately previous action:\n"
            f"{json.dumps(self._pending_feedback, ensure_ascii=False, allow_nan=False)}\n\n"
            f"{current_state}\n"
            "Use the actual outcome of the previous action and the newest observation. "
            "Keep movements small while uncertain, and reduce the motion near contact "
            "or after an unexpected execution result.\n"
            "Output exactly one next absolute target in the required schema."
        )

    def _observation_id(self, step_id):
        identity = self._evaluation_context
        return (f"{identity.get('evaluation_id', 'evaluation')}:"
                f"r{identity.get('repeat_id', 1)}:{identity.get('task', 'task')}:"
                f"ep{identity.get('episode_id', 0)}:obs{int(step_id):03d}")

    def observation_id_for_step(self, step_id):
        return self._observation_id(step_id)

    def _action_id(self, step_id):
        identity = self._evaluation_context
        return (f"{identity.get('evaluation_id', 'evaluation')}:"
                f"r{identity.get('repeat_id', 1)}:{identity.get('task', 'task')}:"
                f"ep{identity.get('episode_id', 0)}:action{int(step_id):03d}")

    def record_execution_feedback(self, feedback):
        """Persist one completed, allowlisted action result for the next turn."""
        if not isinstance(feedback, dict):
            raise ValueError("execution feedback must be a dictionary")
        expected = self._last_action_binding
        if expected is None:
            raise ModelServiceError("there is no pending action to acknowledge",
                                    error_code="feedback_without_action")
        if (feedback.get("step_id") != expected["step_id"]
                or feedback.get("action_id") != expected["action_id"]):
            raise ModelServiceError("execution feedback does not match the last action",
                                    error_code="feedback_identity_mismatch")
        if self._feedback_recorded_action_id == expected["action_id"]:
            raise ModelServiceError("execution feedback was already recorded for this action",
                                    error_code="duplicate_execution_feedback")
        self._pending_feedback = dict(feedback)
        self._feedback_recorded_action_id = expected["action_id"]
        self._append_control_record({
            "event": "execution_feedback",
            "evaluation_id": self._evaluation_context.get("evaluation_id"),
            "episode_id": self._evaluation_context.get("episode_id"),
            "thread_id": self._session.thread_id if self._session else None,
            "turn_id": expected["turn_id"],
            **self._pending_feedback,
        })

    @staticmethod
    def _safe_action_evidence(raw_text, destination, reason_code):
        encoded = raw_text.encode("utf-8", errors="replace")
        evidence = {
            "status": "rejected", "reason_code": reason_code,
            "raw_content_saved": False, "byte_length": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "parseable_json": False,
        }
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            evidence["parseable_json"] = True
            safe_fields = {}
            for field in ("position", "quaternion", "gripper", "ignore_collisions"):
                value = parsed.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    safe_fields[field] = value if not isinstance(value, float) or math.isfinite(value) else str(value)
                elif isinstance(value, list) and len(value) <= 8:
                    safe_fields[field] = [
                        item if isinstance(item, (int, float)) and not isinstance(item, bool)
                        and (not isinstance(item, float) or math.isfinite(item))
                        else {"type": type(item).__name__}
                        for item in value
                    ]
                elif field in parsed:
                    safe_fields[field] = {"type": type(value).__name__}
            evidence["action_fields"] = safe_fields
            evidence["unexpected_field_count"] = sum(
                field not in ("position", "quaternion", "gripper", "ignore_collisions")
                for field in parsed
            )
        CodexAstraPolicy._write_json(destination, evidence)
        return evidence

    def _collision_prompt(self):
        if self.collision_mode == "predict":
            return (
                "Also output ignore_collisions as an integer: 0 keeps collision "
                "checking enabled; 1 asks the existing planner to ignore collisions."
            )
        fixed_value = 0 if self.collision_mode == "fixed0" else 1
        return (
            f"Collision mode is {self.collision_mode}: ignore_collisions is fixed "
            f"to {fixed_value} by the evaluator. Do not output this field."
        )

    def act(self, observation: AstraObservation) -> AstraAction:
        self._transient_final_text = None
        self.last_metadata = {
            **self._evaluation_context,
            "step_id": self._step_index,
            "policy_type": "codex_app_server",
            "requested_model": self.model,
            "resolved_model": None,
            "codex_cli_version": self.codex_cli_version,
            "codex_cli_version_probe_invocation_count": self.codex_cli_version_probe_invocation_count,
            "reasoning_effort": self.reasoning_effort,
            "collision_mode": self.collision_mode,
            "decision_made": True,
            "cli_invocation_count": 0,
            "app_server_request_count": 0,
            "underlying_model_request_count": None,
            "latency_seconds": None,
            "token_usage": None,
            "tool_call_detected": None,
        }
        self._active_metadata = self.last_metadata
        try:
            return self._act_impl(observation)
        except Exception as exc:
            self._active_metadata.update(safe_exception_record(exc))
            self._active_metadata["error"] = self._active_metadata["error_summary"]
            self._active_metadata["raw_structured_output"] = None
            step_dir = self._active_metadata.get("step_dir")
            rejected_path = self._active_metadata.get("rejected_output_path")
            final_text = self._transient_final_text
            if step_dir and rejected_path and isinstance(exc, InvalidPolicyOutput):
                try:
                    evidence = self._safe_action_evidence(
                        final_text or "", rejected_path,
                        self._active_metadata["error_code"],
                    )
                    self._active_metadata["rejected_output_path"] = str(rejected_path)
                    self._active_metadata["rejected_output_evidence"] = evidence
                except Exception:
                    pass
            if self._session is not None and isinstance(
                    exc, (InvalidPolicyOutput, PolicyToolViolation)):
                self._session.failed = True
                self._write_session_manifest("failed")
            if (step_dir and self._active_metadata.get("turn_id")
                    and self._active_metadata.get("action_id")):
                try:
                    self._append_control_record({
                        "event": "turn_result",
                        "thread_id": self._active_metadata.get("thread_id"),
                        "session_id": self._active_metadata.get("control_session_id"),
                        "step_id": self._active_metadata.get("step_id"),
                        "observation_id": self._active_metadata.get("observation_id"),
                        "action_id": self._active_metadata.get("action_id"),
                        "turn_id": self._active_metadata.get("turn_id"),
                        "status": "rejected",
                        "error_code": self._active_metadata["error_code"],
                    })
                except Exception:
                    pass
            accepted_path = self._active_metadata.get("accepted_action_path")
            if accepted_path:
                try:
                    Path(accepted_path).unlink(missing_ok=True)
                except Exception:
                    pass
            if step_dir is not None:
                self._save_metadata(Path(step_dir), self._active_metadata)
            self._transient_final_text = None
            self.last_metadata = self._active_metadata
            raise
        finally:
            self._active_metadata = None

    @classmethod
    def _inherited_context_files(cls, working_directory):
        """Find project instruction/config files visible from the control cwd."""
        cwd = Path(working_directory).resolve()
        found = []
        for directory in (cwd, *cwd.parents):
            for relative_path in cls.CONTEXT_FILES:
                candidate = directory / relative_path
                if candidate.is_file():
                    found.append(str(candidate))
        return found

    def _act_impl(self, observation: AstraObservation) -> AstraAction:
        if self._episode_dir is None:
            self.reset(getattr(observation, "instruction", ""))

        step_id = self._step_index
        self._step_index += 1
        step_dir = self._episode_dir / f"step_{step_id:03d}"
        try:
            step_dir.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            raise CoreArtifactWriteError("failed to create policy step directory") from exc

        if self._active_metadata is not None:
            self._active_metadata["step_id"] = step_id
            self._active_metadata["step_dir"] = str(step_dir)

        pose = self._number_list(
            list(observation.eef_pose), 7, "eef_pose",
            error_type=SimulatorInfrastructureError,
        )
        observation_id = self._observation_id(step_id)
        action_id = self._action_id(step_id)
        image_paths = {}
        for camera in self.IMAGE_FIELDS:
            if camera not in observation.images:
                raise SimulatorInfrastructureError(f"Missing {camera} RGB image")
            image = self._image_as_uint8(observation.images[camera], camera)
            image_path = step_dir / f"{camera}.png"
            try:
                Image.fromarray(image).save(str(image_path), format="PNG")
            except Exception as exc:
                raise CoreArtifactWriteError(
                    f"failed to persist {camera} policy input image"
                ) from exc
            image_paths[camera] = str(image_path)

        prompt = self._make_prompt(observation, pose, step_id)
        prompt_path = step_dir / "prompt.txt"
        schema_path = step_dir / "action_schema.json"
        accepted_action_path = step_dir / "accepted_action.json"
        rejected_output_path = step_dir / "rejected_output.json"
        events_path = step_dir / "codex_events.jsonl"
        try:
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            self._write_json(schema_path, self._schema_for_mode(self.collision_mode))
        except Exception as exc:
            raise CoreArtifactWriteError("failed to persist control request") from exc

        codex = shutil.which("codex")
        if codex is None:
            raise ModelServiceError("Codex executable was not found in PATH",
                                    error_code="codex_not_found")
        if self._control_work_dir is None:
            raise CoreArtifactWriteError("episode control directory was not initialized")
        cwd_path = Path(self._control_work_dir).resolve()
        metadata = {
            **(self._active_metadata or {}),
            "model": self.model,
            "requested_model": self.model,
            "resolved_model": self._session.resolved_model if self._session else None,
            "policy_type": "codex_app_server",
            "codex_cli_version": self.codex_cli_version,
            "app_server_info": self._session.app_server_info if self._session else None,
            "reasoning_effort": self.reasoning_effort,
            "latency_seconds": None,
            "inference_deadline_seconds": self.timeout,
            "deadline_type": "app_server_turn_wall_clock",
            "raw_structured_output": None,
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "image_paths": image_paths,
            "observation_id": observation_id,
            "action_id": action_id,
            "thread_id": self._session.thread_id if self._session else None,
            "control_session_id": self._session.session_id if self._session else None,
            "event_jsonl_path": str(events_path),
            "action_schema_path": str(schema_path),
            "action_json_path": str(accepted_action_path),
            "accepted_action_path": str(accepted_action_path),
            "rejected_output_path": str(rejected_output_path),
            "token_usage": None,
            "tool_call_detected": False,
            "tool_call_events": [],
            "codex_working_directory": str(cwd_path),
            "codex_context_files_checked": True,
            "cli_invocation_count": 0,
            "app_server_request_count": 0,
            "underlying_model_request_count": None,
            "collision_mode": self.collision_mode,
            "final_message_text": None,
        }
        if self._active_metadata is not None:
            self._active_metadata.update(metadata)
            metadata = self._active_metadata

        try:
            cwd_path.relative_to(REPO_ROOT)
        except ValueError:
            pass
        else:
            raise ModelServiceError("Codex working directory must be outside the repository",
                                    error_code="unsafe_codex_working_directory")
        inherited_context = self._inherited_context_files(cwd_path)
        metadata["codex_inherited_context_files"] = inherited_context
        if inherited_context:
            raise ModelServiceError(
                "Codex control working directory inherits project instructions/configuration",
                error_code="inherited_codex_context",
            )
        if os.listdir(cwd_path):
            raise ModelServiceError("Codex control working directory is not empty",
                                    error_code="nonempty_codex_working_directory")

        if self._session is None:
            self._session = CodexAppServerSession(
                codex, cwd_path, self.model, self.reasoning_effort, self.timeout,
            )
            self._session_started_at = datetime.now(timezone.utc).isoformat()
            developer_instructions = (
                "You are a controller for one RLBench robot task episode. Follow only "
                "the control instructions and observations supplied in this thread. "
                "Do not use tools, inspect files, or access external information. "
                "Return only the requested structured action."
            )
            thread_identity = self._session.create_thread(developer_instructions)
            metadata.update({
                "thread_id": thread_identity["thread_id"],
                "control_session_id": thread_identity["session_id"],
                "resolved_model": thread_identity["resolved_model"],
                "app_server_info": thread_identity["app_server_info"],
                "app_server_initialization_latency_seconds": (
                    thread_identity["initialization_latency_seconds"]
                ),
                "thread_instruction_sources": thread_identity["instruction_sources"],
                "app_server_session_create_count": 1,
                "app_server_process_count": 1,
            })
            # The real ID binding is durable before the first target can return.
            self._write_session_manifest("active")
        elif self._session.failed or self._session.closed:
            raise ModelServiceError(
                "Codex episode thread is unusable; refusing to create a replacement",
                error_code="control_session_unusable",
            )
        else:
            metadata.update({
                "thread_id": self._session.thread_id,
                "control_session_id": self._session.session_id,
                "resolved_model": self._session.resolved_model,
                "app_server_session_create_count": 0,
                "app_server_process_count": 0,
            })

        request = {
            "event": "turn_input",
            "evaluation_id": self._evaluation_context.get("evaluation_id"),
            "repeat_id": self._evaluation_context.get("repeat_id"),
            "task": self._evaluation_context.get("task"),
            "episode_id": self._evaluation_context.get("episode_id"),
            "thread_id": self._session.thread_id,
            "session_id": self._session.session_id,
            "step_id": step_id,
            "observation_id": observation_id,
            "action_id": action_id,
            "message_text": prompt,
            "images": [
                {"camera_name": camera, "observation_id": observation_id,
                 "path": image_paths[camera]}
                for camera in self.IMAGE_FIELDS
            ],
            "output_schema": self._schema_for_mode(self.collision_mode),
            "status": "submitted",
        }
        self._append_control_record(request)
        started = time.monotonic()
        request_count_before = (
            0 if metadata.get("app_server_session_create_count")
            else self._session.rpc_request_count
        )
        try:
            final_text, turn_metadata = self._session.run_turn(
                prompt, image_paths, self._schema_for_mode(self.collision_mode),
                request_identity={
                    "step_id": step_id,
                    "action_id": action_id,
                    "observation_id": observation_id,
                },
            )
            self._bind_session_turn(
                step_id, action_id, observation_id, turn_metadata, "completed"
            )
        except Exception as exc:
            turn_metadata = dict(self._session.last_turn_metadata)
            self._bind_session_turn(
                step_id, action_id, observation_id, turn_metadata, "rejected"
            )
            metadata.update({
                "thread_id": self._session.thread_id,
                "control_session_id": self._session.session_id,
                "turn_id": turn_metadata.get("turn_id"),
                "turn_end_confirmed": turn_metadata.get("turn_end_confirmed"),
                "tool_call_detected": turn_metadata.get("tool_call_detected", False),
                "tool_call_events": turn_metadata.get("tool_call_events", []),
                "app_server_request_count": (
                    self._session.rpc_request_count - request_count_before
                ),
                "latency_seconds": (
                    turn_metadata.get("latency_seconds")
                    if turn_metadata.get("latency_seconds") is not None
                    else time.monotonic() - started
                ),
            })
            event_text = self._event_evidence_text(
                turn_metadata.get("event_evidence", [])
            )
            self._write_event_evidence(events_path, event_text, metadata)
            self._write_session_manifest("failed")
            self._append_control_record({
                "event": "turn_result", "thread_id": self._session.thread_id,
                "session_id": self._session.session_id,
                "step_id": step_id, "observation_id": observation_id,
                "action_id": action_id, "turn_id": turn_metadata.get("turn_id"),
                "status": "rejected", "error_code": getattr(exc, "error_code", "unknown"),
                "turn_end_confirmed": turn_metadata.get("turn_end_confirmed"),
                "tool_call_detected": turn_metadata.get("tool_call_detected", False),
            })
            self._save_metadata(step_dir, metadata)
            raise

        metadata.update({
            "thread_id": turn_metadata["thread_id"],
            "control_session_id": turn_metadata["session_id"],
            "turn_id": turn_metadata["turn_id"],
            "turn_end_confirmed": turn_metadata["turn_end_confirmed"],
            "turn_started_confirmed": turn_metadata["turn_started_confirmed"],
            "tool_call_detected": turn_metadata["tool_call_detected"],
            "tool_call_events": turn_metadata["tool_call_events"],
            "app_server_request_count": (
                self._session.rpc_request_count - request_count_before
            ),
            "latency_seconds": turn_metadata["latency_seconds"],
            "resolved_model": self._session.resolved_model,
            "model_output_byte_length": len(final_text.encode("utf-8")),
        })
        self._transient_final_text = final_text
        event_text = self._event_evidence_text(
            turn_metadata.get("event_evidence", [])
        )
        self._write_event_evidence(events_path, event_text, metadata)
        self._write_session_manifest("active")

        try:
            raw_bytes = final_text.encode("utf-8")
            if len(raw_bytes) > MAX_REJECTED_OUTPUT_BYTES:
                raise InvalidPolicyOutput("structured action output exceeded the size limit",
                                          error_code="action_output_too_large")
            action_data = json.loads(final_text)
        except InvalidPolicyOutput:
            raise
        except (UnicodeEncodeError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidPolicyOutput("Codex final message was not valid JSON",
                                      error_code="invalid_action_json") from exc
        if not isinstance(action_data, dict):
            raise InvalidPolicyOutput("Codex action must be a JSON object",
                                      error_code="invalid_action_structure")
        expected_keys = {"position", "quaternion", "gripper"}
        if self.collision_mode == "predict":
            expected_keys.add("ignore_collisions")
        if set(action_data) != expected_keys:
            raise InvalidPolicyOutput("Codex action contains missing or unexpected fields",
                                      error_code="invalid_action_fields")

        position = self._number_list(action_data["position"], 3, "position")
        quaternion = self._number_list(action_data["quaternion"], 4, "quaternion")
        quaternion_norm = math.hypot(*quaternion)
        if quaternion_norm <= 1e-8:
            raise InvalidPolicyOutput("Codex quaternion norm is too small",
                                      error_code="invalid_quaternion_norm")
        gripper = action_data["gripper"]
        if isinstance(gripper, bool) or not isinstance(gripper, int) or gripper not in (0, 1):
            raise InvalidPolicyOutput("Codex gripper must be integer 0 or 1",
                                      error_code="invalid_gripper")
        if self.collision_mode == "predict":
            collision = action_data["ignore_collisions"]
            if isinstance(collision, bool) or not isinstance(collision, int) or collision not in (0, 1):
                raise InvalidPolicyOutput("ignore_collisions must be integer 0 or 1",
                                          error_code="invalid_collision_flag")

        metadata["quaternion_norm"] = quaternion_norm
        metadata["parsed_policy_action"] = {
            "position": position, "quaternion": quaternion,
            "gripper": gripper,
            "ignore_collisions": action_data.get("ignore_collisions"),
            "quaternion_norm": quaternion_norm,
            "normalization_applied": False,
        }
        safe_action = {
            "position": position, "quaternion": quaternion, "gripper": gripper,
            **({"ignore_collisions": action_data["ignore_collisions"]}
               if self.collision_mode == "predict" else {}),
        }
        self._write_json(accepted_action_path, safe_action)
        metadata["accepted_action_path"] = str(accepted_action_path)
        metadata["action_json_path"] = str(accepted_action_path)
        self._save_metadata(step_dir, metadata)
        self._append_control_record({
            "event": "turn_result", "thread_id": self._session.thread_id,
            "session_id": self._session.session_id,
            "step_id": step_id, "observation_id": observation_id,
            "action_id": action_id, "turn_id": turn_metadata["turn_id"],
            "final_structured_output": safe_action,
            "accepted": True,
        })
        self._last_action_binding = {
            "step_id": step_id, "action_id": action_id,
            "turn_id": turn_metadata["turn_id"],
        }
        self._feedback_recorded_action_id = None
        self._pending_feedback = None
        self._transient_final_text = None
        self._write_session_manifest("active")
        return AstraAction(
            position=position, quaternion=quaternion, gripper=gripper,
            ignore_collisions=action_data.get("ignore_collisions"),
        )

    def close(self):
        if self._session is not None:
            self.end_episode("policy_closed")

    @staticmethod
    def _resolved_model(event_text):
        """Return a model name only from an explicit CLI response field."""
        try:
            events = [json.loads(line) for line in event_text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            return None
        for event in reversed(events):
            if not isinstance(event, dict):
                continue
            value = event.get("resolved_model")
            if (isinstance(value, str) and value.strip()
                    and len(value) <= 128
                    and all(char.isalnum() or char in "._/-" for char in value)):
                return value.strip()
            if event.get("type") in ("model.resolved", "response.completed"):
                for source in (event.get("response"), event.get("result")):
                    if isinstance(source, dict):
                        value = source.get("model")
                        if (isinstance(value, str) and value.strip()
                                and len(value) <= 128
                                and all(char.isalnum() or char in "._/-" for char in value)):
                            return value.strip()
        return None
