"""ChatGPT-authenticated Codex CLI policy for Astra RLBench evaluation."""

import json
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

from .policy import AstraPolicy
from .schemas import AstraAction, AstraObservation


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

    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "max",
        timeout: float = 180.0,
        work_root: str = "/tmp/rlbench_codex_policy",
    ):
        if not model:
            raise ValueError("Codex model must be non-empty")
        if not reasoning_effort:
            raise ValueError("Codex reasoning effort must be non-empty")
        if float(timeout) <= 0:
            raise ValueError("Codex timeout must be positive")

        self.model = str(model)
        self.reasoning_effort = str(reasoning_effort)
        self.timeout = float(timeout)
        self.work_root = Path(work_root).expanduser().absolute()
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.last_metadata = None
        self._instruction = ""
        self._step_index = 0
        self._episode_dir = None

    def reset(self, instruction: Optional[str] = None) -> None:
        self._instruction = str(instruction or "")
        self._step_index = 0
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self._episode_dir = self.work_root / f"{run_id}_{uuid.uuid4().hex[:8]}"
        self._episode_dir.mkdir(parents=True, exist_ok=False)
        self.last_metadata = None

    @staticmethod
    def _schema():
        return {
            "type": "object",
            "properties": {
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "quaternion": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper": {"type": "integer", "enum": [0, 1]},
            },
            "required": ["position", "quaternion", "gripper"],
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
                found.append({"line": line_number, "event": event})
            item = event.get("item")
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type in cls.TOOL_EVENT_TYPES or "tool_call" in item_type:
                    found.append({"line": line_number, "event": event})
                elif item_type not in cls.SAFE_ITEM_TYPES:
                    found.append({"line": line_number, "event": event})
        return found

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
            "The target orientation must be an ABSOLUTE unit quaternion in this "
            "order:\n\n"
            "[qx, qy, qz, qw]\n\n"
            "Gripper command:\n0 = close\n1 = open\n\n"
            "Use only the task description, attached images, and current robot "
            "state. Do not use tools, shell commands, filesystem inspection, "
            "source code, demonstrations, or external information.\n\n"
            "Return only the structured action required by the output schema."
        )

    def act(self, observation: AstraObservation) -> AstraAction:
        if self._episode_dir is None:
            self.reset(getattr(observation, "instruction", ""))

        step_dir = self._episode_dir / f"step_{self._step_index:03d}"
        self._step_index += 1
        step_dir.mkdir(parents=True, exist_ok=False)

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
        self._write_json(schema_path, self._schema())
        action_path = step_dir / "action.json"
        events_path = step_dir / "codex_events.jsonl"
        stderr_path = step_dir / "codex_stderr.log"

        codex = shutil.which("codex")
        if codex is None:
            raise CodexAstraPolicyError("Codex CLI executable was not found in PATH")

        metadata = {
            "model": self.model,
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
        }

        with tempfile.TemporaryDirectory(prefix="rlbench_codex_cwd_", dir="/tmp") as cwd:
            metadata["codex_working_directory"] = cwd
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
            except subprocess.TimeoutExpired as exc:
                event_text = exc.stdout or ""
                stderr_text = exc.stderr or ""
                if isinstance(event_text, bytes):
                    event_text = event_text.decode("utf-8", errors="replace")
                if isinstance(stderr_text, bytes):
                    stderr_text = stderr_text.decode("utf-8", errors="replace")
                metadata["latency_seconds"] = time.monotonic() - started
                metadata["raw_structured_output"] = (
                    action_path.read_text(encoding="utf-8")
                    if action_path.exists() else None
                )
                events_path.write_text(event_text, encoding="utf-8")
                stderr_path.write_text(stderr_text, encoding="utf-8")
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
        events_path.write_text(event_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        metadata["raw_structured_output"] = (
            action_path.read_text(encoding="utf-8")
            if action_path.exists() else None
        )
        try:
            metadata["tool_call_events"] = self._tool_events(event_text)
            metadata["tool_call_detected"] = bool(metadata["tool_call_events"])
            metadata["token_usage"] = self._token_usage(event_text)
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
            detail = stderr_text.strip()[-2000:]
            raise CodexAstraPolicyError(
                f"Codex CLI exited with code {completed.returncode}: {detail}"
            )
        if not action_path.is_file():
            raise CodexAstraPolicyError("Codex did not create action.json")

        try:
            action_data = json.loads(metadata["raw_structured_output"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise CodexAstraPolicyError(f"Codex action JSON is invalid: {exc}") from exc
        if not isinstance(action_data, dict):
            raise CodexAstraPolicyError("Codex action must be a JSON object")
        expected_keys = {"position", "quaternion", "gripper"}
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

        metadata["quaternion_norm"] = raw_quaternion_norm
        metadata["validated_action"] = {
            "position": position,
            "quaternion": quaternion,
            "gripper": gripper,
        }
        self._save_metadata(step_dir, metadata)
        return AstraAction(
            position=position,
            quaternion=quaternion,
            gripper=gripper,
        )
