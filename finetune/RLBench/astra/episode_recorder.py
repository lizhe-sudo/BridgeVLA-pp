"""Episode artifacts, passive recording-camera capture, and MP4 assembly."""

import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import uuid

import numpy as np
from PIL import Image

from .visualization import (
    BLUE,
    GREEN,
    RED,
    render_recording_view,
    render_summary_card,
    render_title_card,
)


POLICY_CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
CAMERA_FOV_DEG = 60.0
PANEL_WIDTH = 320


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _write_json(path, value):
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2,
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _rotation_to_quaternion(matrix):
    """Convert a right-handed 3x3 rotation matrix to XYZW quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = [
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
            0.25 * s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
             0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.tolist()


def _look_at_quaternion(camera_position, target):
    position = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right_axis = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right_axis)
    if right_norm < 1e-8:
        world_up = np.array([0.0, 1.0, 0.0])
        right_axis = np.cross(world_up, forward)
        right_norm = np.linalg.norm(right_axis)
    right_axis /= right_norm
    up_axis = np.cross(forward, right_axis)
    rotation = np.column_stack((right_axis, up_axis, forward))
    return _rotation_to_quaternion(rotation)


class RecordingCamera:
    """A non-policy, non-collidable PyRep vision sensor fixed for one episode."""

    def __init__(self, scene, azimuth_deg=225.0, elevation_deg=30.0,
                 width=1280, height=720, radius=None):
        from pyrep.objects.vision_sensor import VisionSensor

        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.panel_width = PANEL_WIDTH
        self.azimuth_deg = float(azimuth_deg)
        self.elevation_deg = float(elevation_deg)

        bounds = np.array([
            [scene._workspace_minx, scene._workspace_maxx],
            [scene._workspace_miny, scene._workspace_maxy],
            [scene._workspace_minz, scene._workspace_maxz],
        ], dtype=np.float64)
        self.look_at = np.mean(bounds, axis=1)
        spans = bounds[:, 1] - bounds[:, 0]
        self.radius = float(radius) if radius is not None else max(
            1.8, 2.25 * float(np.max(spans))
        )
        azimuth = math.radians(self.azimuth_deg)
        elevation = math.radians(self.elevation_deg)
        offset = np.array([
            self.radius * math.cos(elevation) * math.cos(azimuth),
            self.radius * math.cos(elevation) * math.sin(azimuth),
            self.radius * math.sin(elevation),
        ])
        self.position = self.look_at + offset
        orientation = _look_at_quaternion(self.position, self.look_at)
        self.sensor = VisionSensor.create(
            [self.width, self.height],
            explicit_handling=True,
            perspective_mode=True,
            use_local_lights=False,
            show_volume_not_detecting=False,
            show_volume_detecting=False,
            near_clipping_plane=0.01,
            far_clipping_plane=10.0,
            view_angle=CAMERA_FOV_DEG,
            position=self.position.tolist(),
        )
        self.sensor.set_quaternion(orientation)
        self._matrix = self.sensor.get_matrix()
        self._intrinsic = self.sensor.get_intrinsic_matrix()
        self.fov_deg = float(self.sensor.get_perspective_angle())
        self.orientation_xyzw = orientation
        self.resolution = [self.width, self.height]

    def project_xyz(self, point):
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        camera_xyz = self._matrix[:3, :3].T @ (point - self._matrix[:3, 3])
        if camera_xyz[2] <= 1e-6:
            return None
        homogeneous = self._intrinsic @ camera_xyz
        return (float(homogeneous[0] / homogeneous[2]),
                float(homogeneous[1] / homogeneous[2]),
                float(camera_xyz[2]))

    def capture_rgb(self):
        self.sensor.handle_explicitly()
        image = np.asarray(self.sensor.capture_rgb())
        if image.shape[:2] != (self.height, self.width):
            if image.shape[:2] == (self.width, self.height):
                image = np.transpose(image, (1, 0, 2))
            else:
                raise RuntimeError(
                    f"recording sensor returned unexpected image shape {image.shape}"
                )
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size \
                    and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image).clip(0, 255).astype(np.uint8)
        return image

    def metadata(self):
        return {
            "type": "fixed_third_person",
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "radius_m": self.radius,
            "position": self.position.tolist(),
            "orientation_xyzw": self.orientation_xyzw,
            "look_at": self.look_at.tolist(),
            "resolution": self.resolution,
            "fov_deg": self.fov_deg,
            "fov_kind": "perspective_view_angle",
            "policy_input": False,
        }

    def close(self):
        if self.sensor is not None:
            self.sensor.remove()
            self.sensor = None


class EpisodeRecorder:
    def __init__(self, output_root, task, episode, instruction="",
                 model="gpt-6-luna", reasoning="max", record_video=True,
                 azimuth_deg=225.0, elevation_deg=30.0, width=1280,
                 height=720, fps=20, radius=None):
        self.output_root = Path(output_root).expanduser().absolute()
        self.task = str(task)
        self.episode = int(episode)
        self.instruction = str(instruction or "")
        self.model = str(model)
        self.reasoning = str(reasoning)
        self.record_video = bool(record_video)
        self.azimuth_deg = float(azimuth_deg)
        self.elevation_deg = float(elevation_deg)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.radius = radius
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("recording width, height, and fps must be positive")
        self.panel_width = PANEL_WIDTH
        self.video_width = self.width + self.panel_width
        self.run_id = self._new_run_id()
        self.run_dir = self.output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.steps_dir = self.run_dir / "steps"
        self.steps_dir.mkdir()
        self.video_dir = self.run_dir / "video"
        self.video_frames_dir = self.video_dir / "frames"
        if self.record_video:
            self.video_frames_dir.mkdir(parents=True)
        self.log_path = self.run_dir / "episode_log.jsonl"
        self._log_file = self.log_path.open("a", encoding="utf-8")
        self._camera = None
        self._video_frame_paths = []
        self._frame_index = 0
        self._video_duration = 0.0
        self._next_video_time = 0.0
        self._last_step = None
        self._target_history = []
        self._run_meta = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": self.task,
            "episode": self.episode,
            "instruction": self.instruction,
            "model": self.model,
            "reasoning_effort": self.reasoning,
            "policy_input_cameras": list(POLICY_CAMERAS),
            "recording_camera_passed_to_policy": False,
            "visualization_markers_in_policy_rgb": False,
            "record_video": self.record_video,
            "recording_camera": None,
            "recording_fps": self.fps,
            "video_resolution": [self.video_width, self.height],
            "steps": 0,
            "success": False,
        }
        _write_json(self.run_dir / "run_meta.json", self._run_meta)

    def _new_run_id(self):
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        task_name = "".join(c.lower() if c.isalnum() else "_" for c in self.task)
        base = f"{timestamp}_{task_name}_ep{self.episode}"
        candidate = base
        suffix = 1
        while (self.output_root / candidate).exists():
            candidate = f"{base}_{suffix:02d}"
            suffix += 1
        return candidate

    def initialize_camera(self, scene):
        if not self.record_video:
            return
        self._camera = RecordingCamera(
            scene, azimuth_deg=self.azimuth_deg,
            elevation_deg=self.elevation_deg, width=self.width,
            height=self.height, radius=self.radius,
        )
        self._run_meta["recording_camera"] = self._camera.metadata()
        _write_json(self.run_dir / "run_meta.json", self._run_meta)
        title = render_title_card(self.video_width, self.height, {
            "task": self.task, "episode": self.episode,
            "instruction": self.instruction, "model": self.model,
            "reasoning": self.reasoning,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
        })
        title_path = self._save_video_frame(title)
        self._video_frame_paths.extend([title_path] * (2 * self.fps))
        self._video_duration += 2.0

    def set_instruction(self, instruction):
        self.instruction = str(instruction or "")
        self._run_meta["instruction"] = self.instruction
        _write_json(self.run_dir / "run_meta.json", self._run_meta)

    @staticmethod
    def _observation_pose(observation):
        pose = getattr(observation, "gripper_pose", None)
        return None if pose is None else np.asarray(pose, dtype=np.float64).tolist()

    @staticmethod
    def _save_policy_images(observation, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name in POLICY_CAMERAS:
            image = getattr(observation, f"{name}_rgb", None)
            if image is None:
                raise RuntimeError(f"raw RLBench observation is missing {name}_rgb")
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise RuntimeError(f"raw RLBench {name}_rgb has invalid shape {array.shape}")
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating) and array.size \
                        and float(np.nanmax(array)) <= 1.0:
                    array = array * 255.0
                array = np.nan_to_num(array).clip(0, 255).astype(np.uint8)
            Image.fromarray(array, mode="RGB").save(directory / f"{name}.png")

    @staticmethod
    def _copy_or_write(src, target, fallback=""):
        if src and Path(src).is_file():
            shutil.copyfile(src, target)
        else:
            target.write_text(fallback, encoding="utf-8")

    def _save_policy_artifacts(self, step_dir, policy_metadata):
        codex_dir = step_dir
        metadata = policy_metadata if isinstance(policy_metadata, dict) else {}
        self._copy_or_write(metadata.get("prompt_path"), codex_dir / "prompt.txt",
                            metadata.get("prompt", ""))
        schema_source = None
        if metadata.get("action_json_path"):
            candidate = Path(metadata["action_json_path"]).with_name("action_schema.json")
            if candidate.exists():
                schema_source = str(candidate)
        if schema_source is None:
            from .codex_policy import CodexAstraPolicy
            _write_json(codex_dir / "action_schema.json", CodexAstraPolicy._schema())
        else:
            self._copy_or_write(schema_source, codex_dir / "action_schema.json")
        self._copy_or_write(metadata.get("action_json_path"),
                            codex_dir / "codex_action.json")
        self._copy_or_write(metadata.get("event_jsonl_path"),
                            codex_dir / "codex_events.jsonl")
        if metadata.get("event_jsonl_path"):
            stderr_source = Path(metadata["event_jsonl_path"]).with_name("codex_stderr.log")
            stderr_source = str(stderr_source)
        else:
            stderr_source = None
        self._copy_or_write(stderr_source, codex_dir / "codex_stderr.log")
        _write_json(codex_dir / "policy_metadata.json", metadata)

    def _panel(self, step_index, current_xyz, target_xyz, actual_xyz=None,
               execution=None, policy_metadata=None):
        execution = execution or {}
        policy_metadata = policy_metadata or {}
        usage = policy_metadata.get("token_usage") or {}

        def status(value):
            return "YES" if value else "NO" if value is not None else "N/A"

        def number(value, unit=""):
            return f"{float(value):.3f}{unit}" if value is not None else "N/A"

        return {
            "task": self.task,
            "instruction": self.instruction,
            "step": step_index,
            "model": self.model,
            "reasoning": self.reasoning,
            "position_error_m": number(execution.get("position_error_m"), " m"),
            "orientation_error_deg": number(
                execution.get("orientation_error_deg"), " deg"
            ),
            "gripper_before": execution.get("gripper_before"),
            "gripper_command": execution.get("gripper_command"),
            "gripper_after": execution.get("gripper_after"),
            "planner_returned": status(execution.get("planner_returned")),
            "target_reached": status(execution.get("target_reached")),
            "reward": number(execution.get("reward")),
            "success": status(execution.get("success")),
            "policy_latency_seconds": number(
                policy_metadata.get("latency_seconds"), " s"
            ),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": usage.get(
                "reasoning_output_tokens", usage.get("reasoning_tokens")
            ),
        }

    def _save_video_frame(self, rgb):
        path = self.video_frames_dir / f"frame_{self._frame_index:06d}.jpg"
        self._frame_index += 1
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB").save(
            path, format="JPEG", quality=92
        )
        return path

    def _add_video_frame(self, rgb, sim_time=None):
        path = self._save_video_frame(rgb)
        if sim_time is None:
            self._video_frame_paths.append(path)
            self._video_duration += 1.0 / self.fps
            return
        frame_interval = 1.0 / self.fps
        if self._next_video_time <= 0.0:
            self._next_video_time = frame_interval
        while sim_time + 1e-9 >= self._next_video_time:
            self._video_frame_paths.append(path)
            self._next_video_time += frame_interval
            self._video_duration += frame_interval

    def _render(self, rgb, phase, current_xyz=None, target_xyz=None,
                actual_xyz=None, panel=None, failure=None):
        return render_recording_view(
            rgb, self._camera, phase,
            current_xyz=current_xyz, target_xyz=target_xyz,
            actual_xyz=actual_xyz, target_history=self._target_history,
            panel=panel, failure=failure,
        )

    def begin_step(self, step, raw_observation, action=None,
                   final_action=None, policy_metadata=None):
        step_dir = self.steps_dir / f"step_{int(step):03d}"
        step_dir.mkdir(parents=True, exist_ok=False)
        self._save_policy_images(raw_observation, step_dir / "observation_before")
        self._save_policy_artifacts(step_dir, policy_metadata)
        pose = self._observation_pose(raw_observation)
        current_xyz = pose[:3] if pose is not None else None
        target_xyz = None if final_action is None else list(final_action[:3])
        if target_xyz is not None:
            self._target_history.append(target_xyz)
        (step_dir / "recording").mkdir()
        if self._camera is not None:
            recording_rgb = self._camera.capture_rgb()
            view = self._render(
                recording_rgb, "BEFORE EXECUTION", current_xyz,
                target_xyz, panel=self._panel(
                    step, current_xyz, target_xyz,
                    policy_metadata=policy_metadata,
                    execution={
                        "gripper_before": getattr(raw_observation, "gripper_open", None),
                        "gripper_command": None if action is None else action.gripper,
                    },
                ),
            )
            Image.fromarray(self._render(
                recording_rgb, "BEFORE EXECUTION", current_xyz, target_xyz
            )[:, :self.width], mode="RGB").save(
                step_dir / "recording" / "before.png"
            )
            before_video_path = self._save_video_frame(view)
            self._video_frame_paths.extend([before_video_path] * self.fps)
            self._video_duration += 1.0
        self._last_step = {
            "step": int(step),
            "dir": step_dir,
            "observation": raw_observation,
            "pose_before": pose,
            "current_xyz": current_xyz,
            "target_xyz": target_xyz,
            "target_quaternion": None if action is None else list(action.quaternion),
            "action": action,
            "final_action": final_action,
            "policy_metadata": policy_metadata or {},
            "gripper_before": getattr(raw_observation, "gripper_open", None),
        }
        return self._last_step

    @contextmanager
    def capture_during_execution(self, scene):
        """Wrap scene.step with a passive RGB capture after each real sim step."""
        step = self._last_step
        if self._camera is None or step is None:
            yield
            return
        original_step = scene.step
        try:
            sim_dt = float(scene.pyrep.get_simulation_timestep())
        except Exception:
            sim_dt = 1.0 / self.fps
        sim_time = 0.0

        def observed_step():
            nonlocal sim_time
            result = original_step()
            sim_time += sim_dt
            try:
                rgb = self._camera.capture_rgb()
                actual_xyz = scene.robot.arm.get_tip().get_position()
                panel = self._panel(
                    step["step"], step["current_xyz"], step["target_xyz"],
                    actual_xyz=actual_xyz,
                    policy_metadata=step["policy_metadata"],
                    execution={
                        "gripper_before": step["gripper_before"],
                        "gripper_command": (
                            None if step["action"] is None else step["action"].gripper
                        ),
                    },
                )
                composed = self._render(
                    rgb, "EXECUTION", step["current_xyz"],
                    step["target_xyz"], actual_xyz=actual_xyz, panel=panel,
                )
                self._add_video_frame(composed, sim_time=sim_time)
            except Exception as exc:
                # Never let a rendering failure abort or alter the robot action.
                if step.get("recording_error") is None:
                    step["recording_error"] = f"{type(exc).__name__}: {exc}"
            return result

        scene.step = observed_step
        try:
            yield
        finally:
            scene.step = original_step

    def finish_step(self, execution, raw_observation_after=None, scene=None):
        step = self._last_step
        if step is None:
            raise RuntimeError("finish_step() called without begin_step()")
        step_dir = step["dir"]
        actual_pose = self._observation_pose(raw_observation_after)
        actual_xyz = actual_pose[:3] if actual_pose is not None else None
        if raw_observation_after is not None:
            self._save_policy_images(raw_observation_after,
                                     step_dir / "observation_after")
        elif scene is not None:
            try:
                actual_xyz = scene.robot.arm.get_tip().get_position()
            except Exception:
                actual_xyz = None

        failure = execution.get("error")
        if self._camera is not None:
            recording_rgb = self._camera.capture_rgb()
            after_only = self._render(
                recording_rgb, "AFTER EXECUTION", step["current_xyz"],
                step["target_xyz"], actual_xyz=actual_xyz,
                failure=failure,
            )
            Image.fromarray(after_only[:, :self.width], mode="RGB").save(
                step_dir / "recording" / "after.png"
            )
            panel = self._panel(
                step["step"], step["current_xyz"], step["target_xyz"],
                actual_xyz=actual_xyz, execution=execution,
                policy_metadata=step["policy_metadata"],
            )
            after_video = self._render(
                recording_rgb, "AFTER EXECUTION", step["current_xyz"],
                step["target_xyz"], actual_xyz=actual_xyz,
                panel=panel, failure=failure,
            )
            after_path = self._save_video_frame(after_video)
            self._video_frame_paths.extend([after_path] * self.fps)
            self._video_duration += 1.0

        execution = dict(execution)
        execution.setdefault("task", self.task)
        execution.setdefault("episode", self.episode)
        execution.setdefault("step", step["step"])
        execution.setdefault("instruction", self.instruction)
        execution.setdefault("eef_pose_before", step["pose_before"])
        execution.setdefault("target_position", step["target_xyz"])
        execution.setdefault("target_quaternion", step["target_quaternion"])
        execution.setdefault("gripper_before", step["gripper_before"])
        execution.setdefault("gripper_command", (
            None if step["action"] is None else step["action"].gripper
        ))
        execution.setdefault("final_9d_action", step["final_action"])
        execution.setdefault("eef_pose_after", actual_pose)
        execution.setdefault("gripper_after", getattr(raw_observation_after,
                                                        "gripper_open", None))
        policy_metadata = step["policy_metadata"]
        execution.setdefault("policy_latency_seconds",
                             policy_metadata.get("latency_seconds"))
        execution.setdefault("token_usage", policy_metadata.get("token_usage"))
        execution.setdefault("tool_call_detected",
                             policy_metadata.get("tool_call_detected"))
        if step.get("recording_error"):
            execution["recording_error"] = step["recording_error"]
        _write_json(step_dir / "execution.json", execution)
        line = json.dumps(_jsonable(execution), ensure_ascii=False, allow_nan=False)
        self._log_file.write(line + "\n")
        self._log_file.flush()
        self._last_step = None
        return execution

    def record_step_failure(self, step, raw_observation, error, policy_metadata=None):
        self.begin_step(step, raw_observation, policy_metadata=policy_metadata)
        return self.finish_step({
            "planner_returned": False,
            "position_error_m": None,
            "orientation_error_deg": None,
            "target_reached": False,
            "reward": 0.0,
            "terminal": True,
            "success": False,
            "error": error,
        })

    def write_summary(self, summary):
        record = dict(summary)
        record.setdefault("kind", "episode_summary")
        record.setdefault("task", self.task)
        record.setdefault("episode", self.episode)
        line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
        self._log_file.write(line + "\n")
        self._log_file.flush()
        self._run_meta.update(_jsonable(record))
        self._run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._run_meta["video_duration_seconds"] = self._video_duration
        _write_json(self.run_dir / "run_meta.json", self._run_meta)

    def finalize_video(self, summary):
        if self._camera is None:
            return None
        card = render_summary_card(self.video_width, self.height, summary)
        card_path = self._save_video_frame(card)
        self._video_frame_paths.extend([card_path] * (2 * self.fps))
        self._video_duration += 2.0
        import cv2

        video_path = self.video_dir / "episode_summary.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
            (self.video_width, self.height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError("OpenCV could not open MP4 video writer")
        try:
            for frame_path in self._video_frame_paths:
                bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if bgr is None or bgr.shape[:2] != (self.height, self.video_width):
                    raise RuntimeError(f"invalid composed video frame: {frame_path}")
                writer.write(bgr)
        finally:
            writer.release()
        self._run_meta["video_path"] = str(video_path)
        self._run_meta["video_resolution"] = [self.video_width, self.height]
        self._run_meta["video_fps"] = self.fps
        self._run_meta["video_frame_count"] = len(self._video_frame_paths)
        self._run_meta["video_duration_seconds"] = len(self._video_frame_paths) / self.fps
        _write_json(self.run_dir / "run_meta.json", self._run_meta)
        return video_path

    def close(self):
        if self._camera is not None:
            try:
                self._camera.close()
            finally:
                self._camera = None
        if not self._log_file.closed:
            self._log_file.close()
