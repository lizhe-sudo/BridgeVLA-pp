"""Episode artifacts, passive recording-camera capture, and MP4 assembly."""

import json
import math
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .visualization import (
    render_policy_grid,
)
from .errors import CoreArtifactWriteError, sanitize_diagnostic


POLICY_CAMERAS = ("front", "left_shoulder", "right_shoulder", "wrist")
SCENE_CAMERA_ATTRS = {
    "front": "_cam_front",
    "left_shoulder": "_cam_over_shoulder_left",
    "right_shoulder": "_cam_over_shoulder_right",
    "wrist": "_cam_wrist",
}


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
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False,
                      indent=2, allow_nan=False)
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
            f"failed to persist JSON artifact {Path(path).name}"
        ) from exc


class PolicyCameraView:
    """Video rendering view backed by an existing Luna input sensor."""

    def __init__(self, name, sensor, display_size):
        self.name = name
        self.sensor = sensor
        self.display_size = int(display_size)
        resolution = np.asarray(sensor.get_resolution(), dtype=np.int64).reshape(-1)
        if resolution.shape != (2,) or np.any(resolution <= 0):
            raise RuntimeError(f"{name} camera has invalid resolution {resolution}")
        self.source_width, self.source_height = map(int, resolution)
        self.width = self.height = self.display_size
        self._scale = min(
            self.width / self.source_width,
            self.height / self.source_height,
        )
        self._offset_x = (self.width - self.source_width * self._scale) / 2.0
        self._offset_y = (self.height - self.source_height * self._scale) / 2.0
        self._matrix = None
        self._intrinsic = None
        self.update_projection()

    def update_projection(self):
        matrix = np.asarray(self.sensor.get_matrix(), dtype=np.float64)
        if matrix.shape == (4, 4):
            self._matrix = matrix[:3, :]
        else:
            self._matrix = matrix.reshape(3, 4)
        self._intrinsic = np.asarray(
            self.sensor.get_intrinsic_matrix(), dtype=np.float64
        ).reshape(3, 3)

    def project_xyz(self, point):
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        camera_xyz = self._matrix[:, :3].T @ (point - self._matrix[:, 3])
        if camera_xyz[2] <= 1e-6:
            return None
        homogeneous = self._intrinsic @ camera_xyz
        source_x = float(homogeneous[0] / homogeneous[2])
        source_y = float(homogeneous[1] / homogeneous[2])
        return (
            self._offset_x + source_x * self._scale,
            self._offset_y + source_y * self._scale,
            float(camera_xyz[2]),
        )

    def prepare_rgb(self, image):
        image = np.asarray(image)
        if image.shape[:2] != (self.source_height, self.source_width):
            if image.shape[:2] == (self.source_width, self.source_height):
                image = np.transpose(image, (1, 0, 2))
            else:
                raise RuntimeError(
                    f"{self.name} camera returned unexpected image shape {image.shape}"
                )
        if image.ndim != 3 or image.shape[-1] < 3:
            raise RuntimeError(f"{self.name} camera returned invalid RGB shape {image.shape}")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size \
                    and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.nan_to_num(image).clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(image, mode="RGB")
        resized_size = (
            max(1, int(round(self.source_width * self._scale))),
            max(1, int(round(self.source_height * self._scale))),
        )
        pil_image = pil_image.resize(resized_size, Image.LANCZOS)
        tile = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        tile.paste(pil_image, (int(round(self._offset_x)),
                               int(round(self._offset_y))))
        return np.asarray(tile)

    def capture_rgb(self):
        self.sensor.handle_explicitly()
        image = self.sensor.capture_rgb()
        self.update_projection()
        return self.prepare_rgb(image)

    def metadata(self, sensor_attr):
        return {
            "sensor": sensor_attr,
            "camera_name": self.name,
            "source_resolution": [self.source_width, self.source_height],
            "video_tile_resolution": [self.width, self.height],
            "position": np.asarray(self.sensor.get_position(), dtype=float).tolist(),
            "orientation_xyzw": np.asarray(
                self.sensor.get_quaternion(), dtype=float
            ).tolist(),
            "view_angle_deg": float(self.sensor.get_perspective_angle()),
            "pose_dynamic": self.name == "wrist",
        }


class PolicyCameraRig:
    """Passive capture and projection for the exact four policy input sensors."""

    def __init__(self, scene, display_size):
        self.cameras = {}
        self.sensor_attrs = {}
        for name in POLICY_CAMERAS:
            attr = SCENE_CAMERA_ATTRS[name]
            sensor = getattr(scene, attr, None)
            if sensor is None:
                raise RuntimeError(f"RLBench scene is missing policy camera {attr}")
            self.cameras[name] = PolicyCameraView(name, sensor, display_size)
            self.sensor_attrs[name] = attr
        self.display_size = int(display_size)

    def capture_views(self):
        return {name: camera.capture_rgb()
                for name, camera in self.cameras.items()}

    def views_from_observation(self, observation):
        views = {}
        for name, camera in self.cameras.items():
            image = getattr(observation, f"{name}_rgb", None)
            if image is None:
                raise RuntimeError(f"raw RLBench observation is missing {name}_rgb")
            camera.update_projection()
            views[name] = camera.prepare_rgb(image)
        return views

    def metadata(self):
        return {name: camera.metadata(self.sensor_attrs[name])
                for name, camera in self.cameras.items()}


class EpisodeRecorder:
    def __init__(self, output_root, task, episode, instruction="",
                 model="gpt-6-luna", reasoning="max", record_video=True,
                 view_size=512, fps=20, policy_type="mock",
                 collision_mode="fixed0", run_context=None):
        self.output_root = Path(output_root).expanduser().absolute()
        self.task = str(task)
        self.episode = int(episode)
        self.instruction = str(instruction or "")
        self.model = str(model)
        self.reasoning = str(reasoning)
        self.policy_type = str(policy_type)
        self.collision_mode = str(collision_mode)
        self.run_context = dict(run_context or {})
        self.record_video = bool(record_video)
        self.view_size = int(view_size)
        self.fps = int(fps)
        if self.view_size <= 0 or self.fps <= 0:
            raise ValueError("recording view size and fps must be positive")
        self.video_width = self.video_height = self.view_size * 2
        self.run_id = self._new_run_id()
        self.run_dir = self.output_root / self.run_id
        self.steps_dir = self.run_dir / "steps"
        self.video_dir = self.run_dir / "video"
        self.video_frames_dir = self.video_dir / "frames"
        self.log_path = self.run_dir / "episode_log.jsonl"
        try:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self.steps_dir.mkdir()
            if self.record_video:
                self.video_frames_dir.mkdir(parents=True)
            self._log_file = self.log_path.open("a", encoding="utf-8")
        except OSError as exc:
            raise CoreArtifactWriteError(
                "failed to prepare episode artifact directories"
            ) from exc
        self._camera_rig = None
        self._video_frame_paths = []
        self._frame_index = 0
        self._video_duration = 0.0
        self._next_video_time = 0.0
        self.simulation_step_count = 0
        self.simulation_time_seconds = 0.0
        self.recording_error = None
        self.redundant_record_errors = []
        self.recording_status = "pending" if self.record_video else "disabled"
        self.summary_path = self.run_dir / "episode_summary.json"
        self._episode_summary = None
        self._last_step = None
        self._target_history = []
        self._run_meta = {
            "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "task": self.task,
            "episode": self.episode,
            "instruction": self.instruction,
            "model": self.model,
            "experiment_name": "Astra-Direct-RGB",
            "policy_type": self.policy_type,
            "requested_model": self.model if self.policy_type == "codex_cli" else None,
            "resolved_model": None,
            "codex_cli_version": None,
            "reasoning_effort": self.reasoning,
            "collision_mode": self.collision_mode,
            "collision_action_space_matches_predictive_baseline": (
                self.collision_mode == "predict"
            ),
            **self.run_context,
            "policy_input_cameras": list(POLICY_CAMERAS),
            "recording_views_match_policy_inputs": True,
            "video_composite_passed_to_policy": False,
            "visualization_markers_in_policy_rgb": False,
            "record_video": self.record_video,
            "recording_views": None,
            "recording_view_order": list(POLICY_CAMERAS),
            "recording_layout": "2x2",
            "recording_view_size": self.view_size,
            "recording_fps": self.fps,
            "video_resolution": [self.video_width, self.video_height],
            "steps": 0,
            "success": False,
            "simulation_step_count": 0,
            "simulation_time_seconds": 0.0,
            "recording_status": self.recording_status,
            "recording_error": None,
        }
        try:
            _write_json(self.run_dir / "run_meta.json", self._run_meta)
        except Exception as exc:
            self._add_redundant_record_error("run_meta.json", exc)

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

    def initialize_camera(self, scene, raw_observation=None):
        if not self.record_video:
            return
        try:
            self._initialize_camera(scene, raw_observation)
        except Exception as exc:
            self._set_recording_error(exc)
            self._camera_rig = None

    def _set_recording_error(self, exc):
        raw_message = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        message = sanitize_diagnostic(raw_message)["text"]
        self.recording_error = self.recording_error or message
        self.recording_status = "failed"
        self._run_meta["recording_status"] = self.recording_status
        self._run_meta["recording_error"] = self.recording_error
        return message

    def _initialize_camera(self, scene, raw_observation=None):
        self._camera_rig = PolicyCameraRig(scene, self.view_size)
        self._run_meta["recording_views"] = self._camera_rig.metadata()
        try:
            _write_json(self.run_dir / "run_meta.json", self._run_meta)
        except Exception as exc:
            self._add_redundant_record_error("run_meta.json", exc)
        views = (
            self._camera_rig.views_from_observation(raw_observation)
            if raw_observation is not None
            else self._camera_rig.capture_views()
        )
        try:
            pose = np.asarray(scene.robot.arm.get_tip().get_pose(), dtype=float)
            current_xyz = pose[:3].tolist()
        except Exception:
            current_xyz = None
        ready = self._render(
            views, "READY", 0, current_xyz=current_xyz,
            panel={"task": self.task, "instruction": self.instruction},
        )
        ready_path = self._save_video_frame(ready)
        self._video_frame_paths.extend([ready_path] * (2 * self.fps))
        self._video_duration += 2.0
        self.recording_status = "recording"

    def set_instruction(self, instruction):
        self.instruction = str(instruction or "")
        self._run_meta["instruction"] = self.instruction
        try:
            _write_json(self.run_dir / "run_meta.json", self._run_meta)
        except Exception as exc:
            self._add_redundant_record_error("run_meta.json", exc)

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
        try:
            if src and Path(src).is_file():
                shutil.copyfile(src, target)
            else:
                target.write_text(fallback, encoding="utf-8")
        except Exception as exc:
            raise CoreArtifactWriteError(
                f"failed to copy core policy artifact {Path(target).name}"
            ) from exc

    def _save_policy_artifacts(self, step_dir, policy_metadata):
        codex_dir = step_dir
        metadata = policy_metadata if isinstance(policy_metadata, dict) else {}
        if self.policy_type != "codex_cli":
            _write_json(codex_dir / "policy_metadata.json", {
                "policy_type": self.policy_type,
                "requested_model": None,
                "resolved_model": None,
                "codex_cli_version": None,
                "reasoning_effort": None,
                "service_call": "not_applicable",
            })
            return
        self._copy_or_write(metadata.get("prompt_path"), codex_dir / "prompt.txt",
                            metadata.get("prompt", ""))
        schema_source = metadata.get("action_schema_path")
        if not schema_source and metadata.get("action_json_path"):
            candidate = Path(metadata["action_json_path"]).with_name("action_schema.json")
            if candidate.exists():
                schema_source = str(candidate)
        if schema_source is None:
            from .codex_policy import CodexAstraPolicy
            _write_json(
                codex_dir / "action_schema.json",
                CodexAstraPolicy._schema_for_mode(self.collision_mode),
            )
        else:
            self._copy_or_write(schema_source, codex_dir / "action_schema.json")
        accepted_action = metadata.get("accepted_action_path") or metadata.get("action_json_path")
        if accepted_action and Path(accepted_action).is_file():
            self._copy_or_write(accepted_action, codex_dir / "accepted_action.json")
        rejected = metadata.get("rejected_output_path")
        if rejected and Path(rejected).is_file():
            self._copy_or_write(rejected, codex_dir / "rejected_output.json")
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

    def _render(self, rgb_views, phase, step_index, current_xyz=None,
                target_xyz=None, actual_xyz=None, panel=None, failure=None):
        return render_policy_grid(
            rgb_views, self._camera_rig.cameras, phase, step_index,
            current_xyz=current_xyz, target_xyz=target_xyz,
            actual_xyz=actual_xyz, target_history=self._target_history,
            panel=panel, failure=failure,
        )

    def begin_step(self, step, raw_observation, action=None,
                   final_action=None, policy_metadata=None):
        step_dir = self.steps_dir / f"step_{int(step):03d}"
        pose = self._observation_pose(raw_observation)
        current_xyz = pose[:3] if pose is not None else None
        target_xyz = None if final_action is None else list(final_action[:3])
        if target_xyz is not None:
            self._target_history.append(target_xyz)
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
        try:
            step_dir.mkdir(parents=True, exist_ok=False)
            self._save_policy_images(raw_observation, step_dir / "observation_before")
            self._save_policy_artifacts(step_dir, policy_metadata)
        except CoreArtifactWriteError:
            raise
        except Exception as exc:
            raise CoreArtifactWriteError(
                f"failed to persist core inputs for step {step}"
            ) from exc

        if self._camera_rig is not None:
            try:
                (step_dir / "recording").mkdir()
                recording_rgb = self._camera_rig.views_from_observation(raw_observation)
                view = self._render(
                    recording_rgb, "BEFORE EXECUTION", step, current_xyz,
                    target_xyz, panel=self._panel(
                        step, current_xyz, target_xyz,
                        policy_metadata=policy_metadata,
                        execution={
                            "gripper_before": getattr(raw_observation, "gripper_open", None),
                            "gripper_command": None if action is None else action.gripper,
                        },
                    ),
                )
                before_only = self._render(
                    recording_rgb, "BEFORE EXECUTION", step, current_xyz, target_xyz
                )
                Image.fromarray(before_only, mode="RGB").save(
                    step_dir / "recording" / "before.png"
                )
                before_video_path = self._save_video_frame(view)
                self._video_frame_paths.extend([before_video_path] * self.fps)
                self._video_duration += 1.0
            except Exception as exc:
                self._set_recording_error(exc)
        return self._last_step

    @contextmanager
    def capture_during_execution(self, scene):
        """Wrap scene.step with a passive RGB capture after each real sim step."""
        step = self._last_step
        if step is None:
            yield
            return
        original_step = scene.step
        try:
            sim_dt = float(scene.pyrep.get_simulation_timestep())
        except Exception:
            sim_dt = 1.0 / self.fps
        sim_time = 0.0
        self._next_video_time = 1.0 / self.fps

        def observed_step():
            nonlocal sim_time
            result = original_step()
            sim_time += sim_dt
            self.simulation_step_count += 1
            self.simulation_time_seconds += sim_dt
            if self._camera_rig is not None:
                try:
                    rgb = self._camera_rig.capture_views()
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
                        rgb, "EXECUTION", step["step"], step["current_xyz"],
                        step["target_xyz"], actual_xyz=actual_xyz, panel=panel,
                    )
                    self._add_video_frame(composed, sim_time=sim_time)
                except Exception as exc:
                    # Never let a rendering failure abort or alter the robot action.
                    if step.get("recording_error") is None:
                        step["recording_error"] = self._set_recording_error(exc)
                    else:
                        self._set_recording_error(exc)
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
            try:
                self._save_policy_images(raw_observation_after,
                                         step_dir / "observation_after")
            except Exception as exc:
                self._set_recording_error(exc)
        elif scene is not None:
            try:
                actual_xyz = scene.robot.arm.get_tip().get_position()
            except Exception:
                actual_xyz = None

        failure = execution.get("error")
        if self._camera_rig is not None:
            try:
                recording_rgb = (
                    self._camera_rig.views_from_observation(raw_observation_after)
                    if raw_observation_after is not None
                    else self._camera_rig.capture_views()
                )
                after_only = self._render(
                    recording_rgb, "AFTER EXECUTION", step["step"], step["current_xyz"],
                    step["target_xyz"], actual_xyz=actual_xyz, failure=failure,
                )
                Image.fromarray(after_only, mode="RGB").save(
                    step_dir / "recording" / "after.png"
                )
                panel = self._panel(
                    step["step"], step["current_xyz"], step["target_xyz"],
                    actual_xyz=actual_xyz, execution=execution,
                    policy_metadata=step["policy_metadata"],
                )
                after_video = self._render(
                    recording_rgb, "AFTER EXECUTION", step["step"], step["current_xyz"],
                    step["target_xyz"], actual_xyz=actual_xyz,
                    panel=panel, failure=failure,
                )
                after_path = self._save_video_frame(after_video)
                self._video_frame_paths.extend([after_path] * self.fps)
                self._video_duration += 1.0
            except Exception as exc:
                self._set_recording_error(exc)

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
        if isinstance(policy_metadata, dict):
            for field in ("requested_model", "resolved_model", "codex_cli_version"):
                if field in policy_metadata:
                    self._run_meta[field] = policy_metadata[field]
        execution.setdefault("policy_latency_seconds",
                             policy_metadata.get("latency_seconds"))
        execution.setdefault("token_usage", policy_metadata.get("token_usage"))
        execution.setdefault("tool_call_detected",
                             policy_metadata.get("tool_call_detected"))
        if step.get("recording_error"):
            execution["recording_error"] = step["recording_error"]
        execution["simulation_step_count"] = self.simulation_step_count
        execution["simulation_time_seconds"] = self.simulation_time_seconds
        if self.recording_error:
            execution["recording_error"] = self.recording_error
        try:
            _write_json(step_dir / "execution.json", execution)
        except Exception as exc:
            raise CoreArtifactWriteError(
                f"failed to persist canonical execution record for step {step['step']}"
            ) from exc
        try:
            line = json.dumps(_jsonable(execution), ensure_ascii=False, allow_nan=False)
            self._log_file.write(line + "\n")
            self._log_file.flush()
        except Exception as exc:
            self._add_redundant_record_error("episode_log.jsonl", exc)
        self._last_step = None
        return execution

    def _add_redundant_record_error(self, path_name, exc):
        detail = sanitize_diagnostic(f"{type(exc).__name__}: {exc}")["text"]
        self.redundant_record_errors.append({"path": path_name, "error": detail})

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
        record.setdefault("simulation_step_count", self.simulation_step_count)
        record.setdefault("simulation_time_seconds", self.simulation_time_seconds)
        record["recording_status"] = self.recording_status
        record["recording_error"] = self.recording_error
        self._episode_summary = _jsonable(record)
        self._write_summary_files(append_log=True)

    def _write_summary_files(self, append_log=False):
        if self._episode_summary is None:
            return
        def persist_authoritative_summary():
            self._episode_summary["redundant_record_errors"] = list(
                self.redundant_record_errors
            )
            _write_json(self.summary_path, self._episode_summary)

        persist_authoritative_summary()
        if append_log:
            try:
                line = json.dumps(self._episode_summary, ensure_ascii=False, allow_nan=False)
                self._log_file.write(line + "\n")
                self._log_file.flush()
            except Exception as exc:
                self._add_redundant_record_error("episode_log.jsonl", exc)
                persist_authoritative_summary()
        self._run_meta.update(self._episode_summary)
        self._run_meta["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._run_meta.setdefault("video_duration_seconds", None)
        self._run_meta["video_duration_planned_seconds"] = self._video_duration
        self._run_meta["simulation_step_count"] = self.simulation_step_count
        self._run_meta["simulation_time_seconds"] = self.simulation_time_seconds
        self._run_meta["recording_status"] = self.recording_status
        self._run_meta["recording_error"] = self.recording_error
        try:
            _write_json(self.run_dir / "run_meta.json", self._run_meta)
        except Exception as exc:
            self._add_redundant_record_error("run_meta.json", exc)
            persist_authoritative_summary()

    def update_summary(self, summary):
        """Update the saved summary without appending a duplicate episode row."""
        if self._episode_summary is None:
            raise RuntimeError("write_summary() must be called before update_summary()")
        self._episode_summary.update(_jsonable(summary))
        self._episode_summary["recording_status"] = self.recording_status
        self._episode_summary["recording_error"] = self.recording_error
        self._write_summary_files(append_log=False)

    def finalize_video(self, summary):
        if not self.record_video:
            self.recording_status = "disabled"
            if self._episode_summary is not None:
                self.update_summary(summary)
            return None
        video_path = None
        if self._camera_rig is None:
            if self.recording_error is None:
                self.recording_status = "failed"
                self.recording_error = "camera_not_initialized"
        else:
            try:
                import cv2

                video_path = self.video_dir / "episode_summary.mp4"
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                    (self.video_width, self.video_height),
                )
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError("OpenCV could not open MP4 video writer")
                try:
                    for frame_path in self._video_frame_paths:
                        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                        if bgr is None or bgr.shape[:2] != (
                                self.video_height, self.video_width):
                            raise RuntimeError("invalid composed video frame")
                        writer.write(bgr)
                finally:
                    writer.release()
                self._run_meta["video_path"] = str(video_path)
                self._run_meta["video_resolution"] = [self.video_width, self.video_height]
                self._run_meta["video_fps"] = self.fps
                self._run_meta["video_frame_count"] = len(self._video_frame_paths)
                self._run_meta["video_duration_seconds"] = len(self._video_frame_paths) / self.fps
                self.recording_status = (
                    "partial" if self.recording_error is not None else "complete"
                )
            except Exception as exc:
                video_path = None
                self._set_recording_error(exc)
        if self._episode_summary is not None:
            # This canonical write is deliberately outside the video exception
            # handler so a core-record failure cannot be relabeled as recording.
            self.update_summary(summary)
        return video_path

    def close(self):
        if not self._log_file.closed:
            self._log_file.close()
