"""ChatGPT-authenticated Codex CLI policy for Astra RLBench evaluation."""

import json
import math
import os
import re
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

from .policy import AstraPolicy
from .schemas import AstraAction, AstraObservation

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK_ROOT = REPO_ROOT / "tmp" / "rlbench_codex_policy"


class CodexAstraPolicyError(RuntimeError):
    """Raised when Codex cannot produce a safe, valid structured action."""


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
        if float(timeout) <= 0:
            raise ValueError("Codex timeout must be positive")

        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        if collision_mode not in ("fixed0", "fixed1", "predict"):
            raise ValueError("collision_mode must be fixed0, fixed1, or predict")
        self.collision_mode = collision_mode
        self.timeout = float(timeout)
        self.work_root = (
            Path(work_root).expanduser().absolute()
            if work_root is not None
            else DEFAULT_WORK_ROOT
        )
        self.work_root.mkdir(parents=True, exist_ok=True)
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
        self._episode_dir.mkdir(parents=True, exist_ok=False)
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
            raise CodexAstraPolicyError(
                f"{name} RGB image must be HxWx3, got {array.shape}"
            )
        if array.dtype == np.uint8:
            return array
        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise CodexAstraPolicyError(f"{name} RGB image is non-finite")
            if array.size and float(array.max()) <= 1.0 and float(array.min()) >= 0.0:
                array = array * 255.0
            elif array.size and (float(array.min()) < 0.0 or float(array.max()) > 255.0):
                raise CodexAstraPolicyError(
                    f"{name} RGB image values must be in [0,1] or [0,255]"
                )
            return np.rint(array).clip(0, 255).astype(np.uint8)
        if np.issubdtype(array.dtype, np.integer):
            return array.clip(0, 255).astype(np.uint8)
        raise CodexAstraPolicyError(
            f"{name} RGB image has unsupported dtype {array.dtype}"
        )

    @staticmethod
    def _number_list(value, length, field):
        if not isinstance(value, list) or len(value) != length:
            raise CodexAstraPolicyError(
                f"{field} must contain exactly {length} numbers"
            )
        result = []
        for component in value:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise CodexAstraPolicyError(f"{field} must contain only numbers")
            component = float(component)
            if not math.isfinite(component):
                raise CodexAstraPolicyError(f"{field} contains a non-finite number")
            result.append(component)
        return result

    @classmethod
    def _tool_events(cls, event_text):
        if not event_text.strip():
            raise CodexAstraPolicyError("Codex event stream is empty")
        found = []
        for line_number, line in enumerate(event_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CodexAstraPolicyError(
                    f"Codex emitted invalid JSON event on line {line_number}: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise CodexAstraPolicyError(
                    f"Codex event on line {line_number} is not a JSON object"
                )
            event_type = str(event.get("type", ""))
            if event_type in cls.TOOL_EVENT_TYPES or "tool_call" in event_type:
                found.append({"line": line_number, "event_type": event_type})
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type in cls.TOOL_EVENT_TYPES or "tool_call" in item_type:
                    found.append({"line": line_number, "item_type": item_type})
                elif item_type not in cls.SAFE_ITEM_TYPES:
                    found.append({"line": line_number, "item_type": item_type})
        return found

    @classmethod
    def _safe_event_log(cls, event_text):
        """Persist event types and numeric usage, never arbitrary payloads."""
        records = []
        for line_number, line in enumerate(event_text.splitlines(), 1):
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
            record = {"line": line_number, "type": str(event.get("type", "unknown"))}
            item = event.get("item")
            if isinstance(item, dict):
                record["item_type"] = str(item.get("type", "unknown"))
            usage = event.get("usage")
            if isinstance(usage, dict):
                record["usage"] = {
                    key: value for key, value in usage.items()
                    if key in ("input_tokens", "output_tokens", "reasoning_output_tokens")
                    and isinstance(value, int) and not isinstance(value, bool)
                    and value >= 0
                }
            records.append(record)
        return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)

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
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _save_metadata(self, step_dir, metadata):
        if self._active_metadata is not None:
            self._active_metadata.update(metadata)
            metadata = self._active_metadata
        self.last_metadata = metadata
        self._write_json(step_dir / "metadata.json", metadata)

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
            self._active_metadata["error_type"] = type(exc).__name__
            self._active_metadata["error"] = self._redact_text(str(exc))
            self._active_metadata["raw_structured_output"] = None
            action_path = self._active_metadata.get("action_json_path")
            if action_path:
                try:
                    Path(action_path).unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            step_dir = self._active_metadata.get("step_dir")
            if step_dir is not None:
                try:
                    self._save_metadata(Path(step_dir), self._active_metadata)
                except OSError:
                    pass
            self.last_metadata = self._active_metadata
            raise
        finally:
            self._active_metadata = None

    @staticmethod
    def _redact_text(value):
        value = str(value or "")
        value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+", r"\1[REDACTED]", value)
        value = re.sub(
            r"(?i)((?:api[_-]?key|access[_-]?token|secret|password)\s*[=:]\s*)[^\s,;]+",
            r"\1[REDACTED]", value,
        )
        value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", value)
        return value

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
        step_dir.mkdir(parents=True, exist_ok=False)
        if self._active_metadata is not None:
            self._active_metadata["step_id"] = self._step_index - 1
            self._active_metadata["step_dir"] = str(step_dir)

        pose = self._number_list(list(observation.eef_pose), 7, "eef_pose")
        image_paths = {}
        for key in self.IMAGE_FIELDS:
            if key not in observation.images:
                raise CodexAstraPolicyError(f"Missing {key} RGB image")
            image = self._image_as_uint8(observation.images[key], key)
            image_path = step_dir / f"{key}.png"
            Image.fromarray(image).save(str(image_path), format="PNG")
            image_paths[key] = str(image_path)

        prompt = self._make_prompt(observation, pose)
        prompt_path = step_dir / "prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        schema_path = step_dir / "action_schema.json"
        self._write_json(schema_path, self._schema_for_mode(self.collision_mode))
        action_path = step_dir / "action.json"
        events_path = step_dir / "codex_events.jsonl"
        stderr_path = step_dir / "codex_stderr.log"

        codex = shutil.which("codex")
        if codex is None:
            raise CodexAstraPolicyError("Codex CLI executable was not found in PATH")

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
            "codex_return_code": None,
            "raw_structured_output": None,
            "prompt": prompt,
            "prompt_path": str(prompt_path),
            "image_paths": image_paths,
            "event_jsonl_path": str(events_path),
            "action_json_path": str(action_path),
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
                raise CodexAstraPolicyError(
                    "Codex working directory must be outside the repository"
                )
            inherited_context = self._inherited_context_files(cwd_path)
            metadata["codex_context_files_checked"] = True
            metadata["codex_inherited_context_files"] = inherited_context
            if inherited_context:
                raise CodexAstraPolicyError(
                    "Codex working directory inherits project instructions/configuration; "
                    "refusing to invoke Codex"
                )
            if os.listdir(cwd):
                raise CodexAstraPolicyError("Codex working directory is not empty")
            command = [
                codex,
                "exec",
                "-C", cwd,
                "--skip-git-repo-check",
                "--ignore-user-config",
                "-m", self.model,
                "-c", f'model_reasoning_effort="{self.reasoning_effort}"',
                "-c", 'approval_policy="never"',
                "--ephemeral",
                "--sandbox", "read-only",
                "--color", "never",
                "--json",
                "-i",
                *[image_paths[key] for key in self.IMAGE_FIELDS],
                "--output-schema", str(schema_path),
                "--output-last-message", str(action_path),
                prompt,
            ]
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
                events_path.write_text(self._safe_event_log(event_text), encoding="utf-8")
                stderr_path.write_text("", encoding="utf-8")
                try:
                    metadata["tool_call_events"] = self._tool_events(event_text)
                    metadata["tool_call_detected"] = bool(metadata["tool_call_events"])
                    metadata["token_usage"] = self._token_usage(event_text)
                except CodexAstraPolicyError as event_error:
                    metadata["event_parse_error"] = str(event_error)
                self._save_metadata(step_dir, metadata)
                raise CodexAstraPolicyError(
                    f"Codex inference timed out after {self.timeout:g} seconds"
                ) from exc

        metadata["latency_seconds"] = time.monotonic() - started
        events_path.write_text(self._safe_event_log(event_text), encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        metadata["raw_structured_output"] = None
        try:
            metadata["tool_call_events"] = self._tool_events(event_text)
            metadata["tool_call_detected"] = bool(metadata["tool_call_events"])
            metadata["token_usage"] = self._token_usage(event_text)
            metadata["resolved_model"] = self._resolved_model(event_text)
        except CodexAstraPolicyError as exc:
            metadata["event_parse_error"] = str(exc)
            self._save_metadata(step_dir, metadata)
            raise

        self._save_metadata(step_dir, metadata)
        if metadata["tool_call_detected"]:
            raise CodexAstraPolicyError(
                "Codex inference used a tool; refusing to return its action"
            )
        if completed.returncode != 0:
            metadata["codex_stderr_nonempty"] = bool(stderr_text)
            raise CodexAstraPolicyError(
                f"Codex CLI exited with code {completed.returncode}; stderr content was omitted"
            )
        if not action_path.is_file():
            raise CodexAstraPolicyError("Codex did not create action.json")

        try:
            action_data = json.loads(action_path.read_text(encoding="utf-8"))
        except (TypeError, json.JSONDecodeError) as exc:
            raise CodexAstraPolicyError(f"Codex action JSON is invalid: {exc}") from exc
        if not isinstance(action_data, dict):
            raise CodexAstraPolicyError("Codex action must be a JSON object")
        expected_keys = {"position", "quaternion", "gripper"}
        if self.collision_mode == "predict":
            expected_keys.add("ignore_collisions")
        if set(action_data) != expected_keys:
            raise CodexAstraPolicyError(
                "Codex action must contain only position, quaternion, and gripper"
            )
        position = self._number_list(action_data["position"], 3, "position")
        quaternion = self._number_list(action_data["quaternion"], 4, "quaternion")
        raw_quaternion_norm = math.hypot(*quaternion)
        if raw_quaternion_norm <= 1e-8:
            raise CodexAstraPolicyError("Codex quaternion norm is too small")
        gripper = action_data["gripper"]
        if isinstance(gripper, bool) or not isinstance(gripper, int) or gripper not in (0, 1):
            raise CodexAstraPolicyError("Codex gripper must be integer 0 or 1")
        if self.collision_mode == "predict":
            collision = action_data["ignore_collisions"]
            if isinstance(collision, bool) or not isinstance(collision, int) or collision not in (0, 1):
                raise CodexAstraPolicyError(
                    "ignore_collisions must be integer 0 or 1 in predict mode"
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
        self._write_json(action_path, {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper,
            **({"ignore_collisions": action_data["ignore_collisions"]}
               if self.collision_mode == "predict" else {}),
        })
        self._save_metadata(step_dir, metadata)
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
            if isinstance(value, str) and value.strip():
                return value.strip()
            if event.get("type") in ("model.resolved", "response.completed"):
                for source in (event.get("response"), event.get("result")):
                    if isinstance(source, dict):
                        value = source.get("model")
                        if isinstance(value, str) and value.strip():
                            return value.strip()
        return None
