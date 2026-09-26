"""ChatGPT-authenticated Codex CLI policy for Astra RLBench evaluation."""

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
from .policy import AstraPolicy
from .schemas import AstraAction, AstraObservation

REPO_ROOT = Path(__file__).resolve().parents[3]


class CodexAstraPolicyError(InvalidPolicyOutput):
    """Backward-compatible name for invalid Codex policy output."""


class CodexAstraPolicy(AstraPolicy):
    """Call Codex CLI once per observation and return one absolute EEF action.

    Each inference is ephemeral, uses an empty temporary working directory,
    and receives only the four RGB views plus the current robot state. Tool
    events invalidate the inference before an action can reach the evaluator.
    """

    IMAGE_FIELDS = ("front", "left_shoulder", "right_shoulder", "wrist")
    SAFE_ITEM_TYPES = {
        "agent_message",
        "reasoning",
        "plan",
        "user_message",
        "developer_message",
        "system_message",
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

    def set_evaluation_context(self, evaluation_id, repeat_id, task, episode_id):
        self._evaluation_context = {
            "evaluation_id": evaluation_id,
            "repeat_id": repeat_id,
            "task": task,
            "episode_id": episode_id,
        }

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = str(instruction or "")
        self._step_index = 0
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self._episode_dir = self.work_root / f"{run_id}_{uuid.uuid4().hex[:8]}"
        try:
            self._episode_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise CoreArtifactWriteError(
                "failed to create persistent Codex episode directory"
            ) from exc
        self.last_metadata = None

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
        """Persist event types and numeric usage, never arbitrary payloads."""
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
            not (char.isalnum() or char in "_.-") for char in value
        ):
            return "unrecognized"
        return value or "unknown"

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

    @staticmethod
    def _build_command(codex, cwd, model, reasoning_effort, schema_path,
                       output_path, image_paths, prompt):
        command = [
            str(codex), "exec", "-C", str(cwd),
            "--skip-git-repo-check", "--ignore-user-config",
            "-m", str(model),
            "-c", f'model_reasoning_effort="{reasoning_effort}"',
            "-c", 'approval_policy="never"',
            "--ephemeral", "--sandbox", "read-only", "--color", "never",
            "--json",
        ]
        for camera in CodexAstraPolicy.IMAGE_FIELDS:
            command.extend(("-i", str(image_paths[camera])))
        command.extend(("--output-schema", str(schema_path)))
        command.extend(("--output-last-message", str(output_path), str(prompt)))
        return command

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

    def _make_prompt(self, observation, pose):
        instruction = str(observation.instruction or self._instruction)
        gripper_text = "open" if observation.gripper_open else "closed"
        return (
            "You control the end-effector of a Franka Panda robot in RLBench.\n\n"
            f"Task:\n{instruction}\n\n"
            "The four attached RGB images are, in order:\n"
            "1. front camera\n"
            "2. left shoulder camera\n"
            "3. right shoulder camera\n"
            "4. wrist camera\n\n"
            "Current end-effector pose in the RLBench world frame:\n\n"
            f"position:\n{json.dumps(pose[:3])}\n\n"
            "quaternion_xyzw:\n"
            f"{json.dumps(pose[3:7])}\n\n"
            f"Current gripper:\n{gripper_text}\n\n"
            "Choose exactly ONE next end-effector waypoint that makes progress "
            "toward the task.\n\n"
            "The target position must be an ABSOLUTE position in the RLBench "
            "world coordinate frame.\n"
            "Position unit: meters.\n\n"
            "The pose target refers to the RLBench arm tip reference point "
            "(the same tip used by the existing pose action mode).\n\n"
            "The target orientation must be an ABSOLUTE unit quaternion in this "
            "order:\n\n"
            "[qx, qy, qz, qw]\n\n"
            "Gripper command:\n0 = close\n1 = open\n\n"
            "The existing action mode executes the arm motion first and applies "
            "the gripper command after the arm stops.\n\n"
            + self._collision_prompt() + "\n\n"
            "Use only the task description, attached images, and current robot "
            "state. Do not use tools, shell commands, filesystem inspection, "
            "source code, demonstrations, or external information.\n\n"
            "Return only the structured action required by the output schema."
        )

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
        self.last_metadata = {
            **self._evaluation_context,
            "step_id": self._step_index,
            "policy_type": "codex_cli",
            "requested_model": self.model,
            "resolved_model": None,
            "codex_cli_version": self.codex_cli_version,
            "codex_cli_version_probe_invocation_count": self.codex_cli_version_probe_invocation_count,
            "reasoning_effort": self.reasoning_effort,
            "collision_mode": self.collision_mode,
            "decision_made": True,
            "cli_invocation_count": 0,
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
            raw_path = self._active_metadata.get("raw_model_output_path")
            step_dir = self._active_metadata.get("step_dir")
            if raw_path and step_dir:
                rejected_path = Path(step_dir) / "rejected_output.json"
                try:
                    evidence = self._rejected_output_evidence(
                        raw_path, rejected_path, self._active_metadata["error_code"]
                    )
                    self._active_metadata["rejected_output_path"] = str(rejected_path)
                    self._active_metadata["rejected_output_evidence"] = evidence
                finally:
                    try:
                        Path(raw_path).unlink(missing_ok=True)
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
            self.last_metadata = self._active_metadata
            raise
        finally:
            self._active_metadata = None

    @classmethod
    def _inherited_context_files(cls, working_directory):
        """Find project instruction/config files visible from the CLI cwd."""
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

        step_dir = self._episode_dir / f"step_{self._step_index:03d}"
        self._step_index += 1
        try:
            step_dir.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            raise CoreArtifactWriteError("failed to create policy step directory") from exc
        if self._active_metadata is not None:
            self._active_metadata["step_id"] = self._step_index - 1
            self._active_metadata["step_dir"] = str(step_dir)

        pose = self._number_list(
            list(observation.eef_pose), 7, "eef_pose",
            error_type=SimulatorInfrastructureError,
        )
        image_paths = {}
        for key in self.IMAGE_FIELDS:
            if key not in observation.images:
                raise SimulatorInfrastructureError(f"Missing {key} RGB image")
            image = self._image_as_uint8(observation.images[key], key)
            image_path = step_dir / f"{key}.png"
            try:
                Image.fromarray(image).save(str(image_path), format="PNG")
            except Exception as exc:
                raise CoreArtifactWriteError(
                    f"failed to persist {key} policy input image"
                ) from exc
            image_paths[key] = str(image_path)

        prompt = self._make_prompt(observation, pose)
        prompt_path = step_dir / "prompt.txt"
        try:
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
        except Exception as exc:
            raise CoreArtifactWriteError("failed to persist policy prompt") from exc
        schema_path = step_dir / "action_schema.json"
        self._write_json(schema_path, self._schema_for_mode(self.collision_mode))
        raw_output_path = step_dir / "raw_model_output.json"
        accepted_action_path = step_dir / "accepted_action.json"
        rejected_output_path = step_dir / "rejected_output.json"
        events_path = step_dir / "codex_events.jsonl"
        stderr_path = step_dir / "codex_stderr.log"

        codex = shutil.which("codex")
        if codex is None:
            raise ModelServiceError(
                "Codex CLI executable was not found in PATH",
                error_code="codex_cli_not_found",
            )

        metadata = {
            **(self._active_metadata or {}),
            "model": self.model,
            "requested_model": self.model,
            "resolved_model": None,
            "policy_type": "codex_cli",
            "codex_cli_version": self.codex_cli_version,
            "codex_cli_version_probe_invocation_count": self.codex_cli_version_probe_invocation_count,
            "reasoning_effort": self.reasoning_effort,
            "latency_seconds": None,
            "inference_deadline_seconds": self.timeout,
            "deadline_type": "inference_wall_clock",
            "codex_return_code": None,
            "raw_structured_output": None,
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "image_paths": image_paths,
            "event_jsonl_path": str(events_path),
            "action_schema_path": str(schema_path),
            "action_json_path": str(accepted_action_path),
            "accepted_action_path": str(accepted_action_path),
            "raw_model_output_path": str(raw_output_path),
            "rejected_output_path": str(rejected_output_path),
            "token_usage": None,
            "tool_call_detected": False,
            "tool_call_events": [],
            "codex_working_directory": None,
            "cli_invocation_count": 0,
            "underlying_model_request_count": None,
            "collision_mode": self.collision_mode,
        }
        if self._active_metadata is not None:
            self._active_metadata.update(metadata)
            metadata = self._active_metadata

        # Keep the CLI cwd outside this checkout so repository/ancestor files
        # cannot silently become project instructions or agent context.
        cwd_root = Path(tempfile.gettempdir()) / "astra_codex_cwd"
        cwd_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="rlbench_codex_cwd_", dir=str(cwd_root)
        ) as cwd:
            metadata["codex_working_directory"] = cwd
            cwd_path = Path(cwd).resolve()
            try:
                cwd_path.relative_to(REPO_ROOT)
            except ValueError:
                pass
            else:
                raise ModelServiceError(
                    "Codex working directory must be outside the repository",
                    error_code="unsafe_codex_working_directory",
                )
            inherited_context = self._inherited_context_files(cwd_path)
            metadata["codex_context_files_checked"] = True
            metadata["codex_inherited_context_files"] = inherited_context
            if inherited_context:
                raise ModelServiceError(
                    "Codex working directory inherits project instructions/configuration; "
                    "refusing to invoke Codex",
                    error_code="inherited_codex_context",
                )
            if os.listdir(cwd):
                raise ModelServiceError(
                    "Codex working directory is not empty",
                    error_code="nonempty_codex_working_directory",
                )
            command = self._build_command(
                codex, cwd, self.model, self.reasoning_effort, schema_path,
                raw_output_path, image_paths, prompt,
            )
            started = time.monotonic()
            metadata["cli_invocation_count"] = 1
            metadata["tool_isolation"] = (
                "read-only sandbox with post-hoc event rejection; no verified CLI flag disables tools"
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )
                event_text = completed.stdout or ""
                stderr_text = completed.stderr or ""
                metadata["codex_return_code"] = completed.returncode
                metadata["codex_stderr_nonempty"] = bool(stderr_text)
            except OSError as exc:
                metadata["latency_seconds"] = time.monotonic() - started
                self._write_event_evidence(events_path, "", metadata)
                self._write_stderr_evidence(stderr_path, "", metadata)
                raise ModelServiceError(
                    "failed to start Codex CLI process",
                    error_code="codex_process_start_failed",
                ) from exc
            except subprocess.TimeoutExpired as exc:
                event_text = exc.stdout or ""
                stderr_text = exc.stderr or ""
                if isinstance(event_text, bytes):
                    event_text = event_text.decode("utf-8", errors="replace")
                if isinstance(stderr_text, bytes):
                    stderr_text = stderr_text.decode("utf-8", errors="replace")
                metadata["codex_stderr_nonempty"] = bool(stderr_text)
                metadata["latency_seconds"] = time.monotonic() - started
                metadata["raw_structured_output"] = None
                self._write_event_evidence(events_path, event_text, metadata)
                self._write_stderr_evidence(stderr_path, stderr_text, metadata)
                try:
                    metadata["tool_call_events"] = self._tool_events(event_text)
                    metadata["tool_call_detected"] = bool(metadata["tool_call_events"])
                    metadata["token_usage"] = self._token_usage(event_text)
                except AstraEvaluationError as event_error:
                    metadata.update(safe_exception_record(event_error))
                self._save_metadata(step_dir, metadata)
                raise InferenceDeadlineExceeded(
                    f"configured Codex inference deadline of {self.timeout:g} seconds exceeded",
                    error_code="inference_deadline_exceeded",
                ) from exc

        metadata["latency_seconds"] = time.monotonic() - started
        self._write_event_evidence(events_path, event_text, metadata)
        self._write_stderr_evidence(stderr_path, stderr_text, metadata)
        metadata["raw_structured_output"] = None
        try:
            metadata["tool_call_events"] = self._tool_events(event_text)
            metadata["tool_call_detected"] = bool(metadata["tool_call_events"])
            metadata["token_usage"] = self._token_usage(event_text)
            metadata["resolved_model"] = self._resolved_model(event_text)
        except AstraEvaluationError as exc:
            metadata.update(safe_exception_record(exc))
            self._save_metadata(step_dir, metadata)
            raise

        self._save_metadata(step_dir, metadata)
        if metadata["tool_call_detected"]:
            raise PolicyToolViolation(
                "Codex inference used a tool; refusing to return its action",
                error_code="tool_event_detected",
            )
        if completed.returncode != 0:
            raise ModelServiceError(
                f"Codex CLI exited with code {completed.returncode}; sanitized stderr is recorded",
                error_code="codex_cli_nonzero_exit",
            )
        if not raw_output_path.is_file():
            raise InvalidPolicyOutput(
                "Codex did not create a structured action output",
                error_code="missing_action_output",
            )

        try:
            raw_bytes = raw_output_path.read_bytes()
            metadata["model_output_byte_length"] = len(raw_bytes)
            if len(raw_bytes) > MAX_REJECTED_OUTPUT_BYTES:
                raise InvalidPolicyOutput(
                    "structured action output exceeded the size limit",
                    error_code="action_output_too_large",
                )
            action_data = json.loads(raw_bytes.decode("utf-8"))
        except InvalidPolicyOutput:
            raise
        except (OSError, UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
            raise InvalidPolicyOutput(
                "Codex action output was not valid JSON",
                error_code="invalid_action_json",
            ) from exc
        if not isinstance(action_data, dict):
            raise InvalidPolicyOutput(
                "Codex action must be a JSON object",
                error_code="invalid_action_structure",
            )
        expected_keys = {"position", "quaternion", "gripper"}
        if self.collision_mode == "predict":
            expected_keys.add("ignore_collisions")
        if set(action_data) != expected_keys:
            raise InvalidPolicyOutput(
                "Codex action contains missing or unexpected fields",
                error_code="invalid_action_fields",
            )
        position = self._number_list(action_data["position"], 3, "position")
        quaternion = self._number_list(action_data["quaternion"], 4, "quaternion")
        raw_quaternion_norm = math.hypot(*quaternion)
        if raw_quaternion_norm <= 1e-8:
            raise InvalidPolicyOutput(
                "Codex quaternion norm is too small",
                error_code="invalid_quaternion_norm",
            )
        gripper = action_data["gripper"]
        if isinstance(gripper, bool) or not isinstance(gripper, int) or gripper not in (0, 1):
            raise InvalidPolicyOutput(
                "Codex gripper must be integer 0 or 1",
                error_code="invalid_gripper",
            )
        if self.collision_mode == "predict":
            collision = action_data["ignore_collisions"]
            if isinstance(collision, bool) or not isinstance(collision, int) or collision not in (0, 1):
                raise InvalidPolicyOutput(
                    "ignore_collisions must be integer 0 or 1 in predict mode",
                    error_code="invalid_collision_flag",
                )

        metadata["quaternion_norm"] = raw_quaternion_norm
        metadata["parsed_policy_action"] = {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper,
            "ignore_collisions": action_data.get("ignore_collisions"),
            "quaternion_norm": raw_quaternion_norm,
            "normalization_applied": False,
        }
        self._write_json(accepted_action_path, {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper,
            **({"ignore_collisions": action_data["ignore_collisions"]}
               if self.collision_mode == "predict" else {}),
        })
        metadata["accepted_action_path"] = str(accepted_action_path)
        metadata["action_json_path"] = str(accepted_action_path)
        self._save_metadata(step_dir, metadata)
        try:
            raw_output_path.unlink(missing_ok=True)
        except Exception:
            pass
        return AstraAction(
            position=position,
            quaternion=quaternion,
            gripper=gripper,
            ignore_collisions=action_data.get("ignore_collisions"),
        )

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
