"""Evaluate a mock/manual absolute-pose policy in the existing RLBench stack."""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(FINETUNE_DIR)
DEFAULT_STEP_LIMITS = os.path.join(SCRIPT_DIR, "configs", "eval_step_limit.yml")


def _add_project_paths(allow_shared=False):
    """Select the same dedicated simulation dependencies as the baseline."""
    from astra.sim_paths import configure_project_paths

    return configure_project_paths(
        SCRIPT_DIR, FINETUNE_DIR, allow_shared=allow_shared
    )


def _build_parser():
    from astra.experiment import EVAL_TASKS

    parser = argparse.ArgumentParser(
        description=(
            "Run the Astra-compatible RLBench control loop using a mock, "
            "manual, or Codex-generated absolute EEF action."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=["open_drawer"],
                        help="task names (default: open_drawer; formal mode still requires the fixed five-task main suite)")
    parser.add_argument("--eval-datafolder", default=None,
                        help="RLBench demonstration dataset root")
    parser.add_argument("--run-mode", choices=("debug", "formal"), default="debug",
                        help="label run as debug or formal (default: debug)")
    parser.add_argument("--budget-protocol", choices=("uniform25", "repo_step_limits"),
                        default="uniform25", help="explicit waypoint budget protocol")
    parser.add_argument("--step-limit-config", default=DEFAULT_STEP_LIMITS,
                        help="task budget YAML used by repo_step_limits")
    parser.add_argument("--max-waypoints", "--episode-length", dest="max_waypoints",
                        type=int, default=None,
                        help="debug-only waypoint cap (cannot increase protocol budget)")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="first demonstration episode index")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="number of episodes per task (default: 1)")
    episode_source = parser.add_mutually_exclusive_group()
    episode_source.add_argument("--episode-ids", type=int, nargs="+", default=None,
                                help="explicit reusable demonstration episode IDs")
    episode_source.add_argument("--episode-list", default=None,
                                help="JSON file containing an episode_ids list")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat each task's selected episode list (default: 1)")
    parser.add_argument("--seed", type=int, default=None,
                        help="base Python/NumPy RNG seed; does not imply simulator determinism")
    parser.add_argument("--policy", choices=("mock", "manual", "codex"), default="mock",
                        help="policy implementation (default: mock)")
    parser.add_argument("--collision-mode", choices=("fixed0", "fixed1", "predict"),
                        default=None, help="explicit ignore_collisions policy (default: fixed0)")
    parser.add_argument("--position-tolerance-m", type=float, default=0.01,
                        help="position diagnostic tolerance in meters")
    parser.add_argument("--orientation-tolerance-deg", type=float, default=5.0,
                        help="pose diagnostic orientation tolerance in degrees")
    parser.add_argument(
        "--manual-action", type=str, default=None,
        help='absolute action "x,y,z,qx,qy,qz,qw,g" (g is 0 or 1)',
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction,
                        default=True, help="run simulator headless (default: true)")
    parser.add_argument("--codex-model", default="gpt-6-luna",
                        help="Codex model (default: gpt-6-luna)")
    parser.add_argument("--codex-reasoning", default="max",
                        help="Codex model reasoning effort (default: max)")
    parser.add_argument("--codex-timeout", type=float, default=180.0,
                        help="Codex App Server turn timeout in seconds (default: 180)")
    parser.add_argument(
        "--codex-work-root",
        default=None,
        help=("explicit directory for persistent Codex policy artifacts; defaults "
              "to <run-directory>/policy_work"),
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="output root (default: <repository-root>/outputs, independent of cwd)",
    )
    parser.add_argument(
        "--record-video", action=argparse.BooleanOptionalAction, default=True,
        help="record separate four-view high-resolution video (default: true)",
    )
    parser.add_argument("--session-mode", choices=("episode",), default="episode",
                        help="native Codex control thread lifetime (one thread per episode)")
    parser.add_argument("--motion-prompt-profile",
                        choices=("adaptive_small_steps",),
                        default="adaptive_small_steps",
                        help="prompt-only movement-size guidance")
    parser.add_argument("--recording-width", type=int, default=1280,
                        help="native high-resolution width of each recording camera")
    parser.add_argument("--recording-height", type=int, default=720,
                        help="native high-resolution height of each recording camera")
    parser.add_argument("--recording-fps", type=int, default=20,
                        help="output video frame rate")
    parser.add_argument("--log-file", default=None,
                        help="optional JSONL file for per-step logs")
    parser.add_argument("--allow-shared-sim-stack", action="store_true",
                        help="explicitly allow nonstandard shared RLBench/PyRep dependencies")
    return parser


def _parse_manual_action(value):
    if value is None:
        return None
    from astra.schemas import AstraAction

    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("--manual-action must be eight comma-separated numbers") from exc
    if len(values) != 8:
        raise ValueError(
            "--manual-action must be x,y,z,qx,qy,qz,qw,g (eight values)"
        )
    return AstraAction(
        position=values[:3], quaternion=values[3:7], gripper=values[7]
    )


def _jsonable(value):
    import numpy as np

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def _exception_category(exc):
    return type(exc).__name__ or "Error"


def _token_counts(policy_metadata):
    usage = (policy_metadata or {}).get("token_usage")
    if not isinstance(usage, dict):
        return None
    def known_int(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value) if value >= 0 else None
    return {
        "input_tokens": known_int(usage.get("input_tokens")),
        "output_tokens": known_int(usage.get("output_tokens")),
        "reasoning_tokens": known_int(usage.get(
            "reasoning_output_tokens", usage.get("reasoning_tokens")
        )),
    }


def _classify_error(exc, phase, metadata=None):
    """Classify from typed errors and exception classes, never message text."""
    from astra.errors import (
        CoreArtifactWriteError,
        InferenceDeadlineExceeded,
        InvalidPolicyOutput,
        ModelServiceError,
        PolicyToolViolation,
        SimulatorInfrastructureError,
    )

    name = _exception_category(exc)
    metadata = metadata or {}
    if isinstance(exc, PolicyToolViolation) or metadata.get("tool_call_detected") is True:
        return "policy_failure", "tool_use_violation"
    if isinstance(exc, InvalidPolicyOutput):
        return "policy_failure", exc.error_code
    if isinstance(exc, InferenceDeadlineExceeded):
        return "policy_failure", exc.error_code
    if isinstance(exc, (ModelServiceError, SimulatorInfrastructureError,
                        CoreArtifactWriteError)):
        return "infrastructure_error", exc.error_code
    if phase == "action_validation":
        return "policy_failure", "invalid_policy_action"
    if phase == "environment_step":
        known_policy_errors = {"ConfigurationPathError", "IKError", "InvalidActionError"}
        if any(base.__name__ in known_policy_errors for base in type(exc).__mro__):
            return "policy_failure", name
        if isinstance(exc, (OSError, EOFError, ChildProcessError)):
            return "infrastructure_error", name
        return "unknown", "unclassified_environment_error"
    if phase in ("reset", "observation") and isinstance(
            exc, (OSError, EOFError, ImportError, ModuleNotFoundError)):
        return "infrastructure_error", name
    return "unknown", "unclassified_error"


def _safe_error_message(exc):
    from astra.errors import safe_exception_record

    detail = safe_exception_record(exc)
    return f"{detail['error_type']} [{detail['error_code']}]: {detail['error_summary']}"


_GIT_QUERY_FAILURES = {}


def _git_value(*args):
    query = "git " + " ".join(str(value) for value in args)
    try:
        result = subprocess.run(
            ["git", "-C", REPO_ROOT, *args], check=True,
            capture_output=True, text=True, timeout=3,
        )
        if not isinstance(result.stdout, str):
            _GIT_QUERY_FAILURES[query] = "git command returned no text stdout"
            return None
        _GIT_QUERY_FAILURES.pop(query, None)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        from astra.errors import sanitize_diagnostic
        _GIT_QUERY_FAILURES[query] = sanitize_diagnostic(
            f"{type(exc).__name__}: {exc}"
        )["text"]
        return None


def _write_json_atomic(path, value):
    from astra.errors import CoreArtifactWriteError

    path = os.path.abspath(path)
    temp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception as exc:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise CoreArtifactWriteError(
            f"failed to persist core run artifact {os.path.basename(path)}"
        ) from exc


def _metadata_for_step(policy, expected):
    metadata = getattr(policy, "last_metadata", None)
    if not isinstance(metadata, dict):
        return None, False
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None, True
    return dict(metadata), False


def _emit(record, log_file=None):
    line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def run_eval(args):
    from astra.errors import CoreArtifactWriteError
    from astra.experiment import (
        EVAL_TASKS, SANITY_CHECK_TASKS, aggregate_results,
        load_episode_ids, read_step_limits, resolve_budgets,
        select_episode_ids, validate_run_protocol,
    )

    tasks = list(args.tasks)
    validate_run_protocol(args.run_mode, tasks, args.eval_episodes, args.repeats)
    if args.start_episode < 0:
        raise ValueError("--start-episode must be non-negative")
    if args.recording_width <= 0 or args.recording_height <= 0 or args.recording_fps <= 0:
        raise ValueError("recording width, height, and fps must be positive")
    if not math.isfinite(args.codex_timeout) or args.codex_timeout <= 0:
        raise ValueError("--codex-timeout must be positive")
    if (not math.isfinite(args.position_tolerance_m)
            or not math.isfinite(args.orientation_tolerance_deg)
            or args.position_tolerance_m <= 0
            or args.orientation_tolerance_deg <= 0):
        raise ValueError("pose diagnostic tolerances must be positive")
    if args.max_waypoints is not None and args.max_waypoints <= 0:
        raise ValueError("--max-waypoints must be positive")
    if args.max_waypoints is not None and args.run_mode != "debug":
        raise ValueError("--max-waypoints is a debug-only budget cap")
    if args.run_mode == "formal" and args.allow_shared_sim_stack:
        raise ValueError("formal runs require the dedicated RLBench/PyRep stacks")
    if args.policy == "manual" and args.manual_action is None:
        raise ValueError("--policy manual requires --manual-action")
    if args.policy != "manual" and args.manual_action is not None:
        raise ValueError("--manual-action is only valid with --policy manual")
    if args.run_mode == "formal" and args.collision_mode is None:
        raise ValueError("formal runs must explicitly specify --collision-mode")
    args.collision_mode = args.collision_mode or "fixed0"

    step_limits = read_step_limits(args.step_limit_config)
    budgets = resolve_budgets(
        tasks, args.budget_protocol, step_limits,
        debug_override=args.max_waypoints,
    )
    episode_ids = (
        load_episode_ids(args.episode_list) if args.episode_list else args.episode_ids
    )
    episode_ids = select_episode_ids(
        args.start_episode, args.eval_episodes, episode_ids
    )
    task_groups = {
        task: (
            "main" if task in EVAL_TASKS else
            "sanity_check" if task in SANITY_CHECK_TASKS else "custom"
        ) for task in tasks
    }
    if args.collision_mode not in ("fixed0", "fixed1", "predict"):
        raise ValueError("unsupported collision mode")

    evaluation_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    from astra.output_paths import resolve_output_layout

    layout = resolve_output_layout(
        REPO_ROOT,
        evaluation_id,
        output_root=args.output_root,
        codex_work_root=args.codex_work_root,
        log_file=args.log_file,
    )
    run_dir = str(layout.run_dir)
    policy_work_dir = str(layout.policy_work_dir)
    output_dir = os.path.join(run_dir, "episodes")
    run_log_path = str(layout.run_log_path)
    try:
        os.makedirs(run_dir, exist_ok=False)
        os.makedirs(policy_work_dir, exist_ok=True)
        os.makedirs(output_dir)
        os.makedirs(os.path.dirname(run_log_path), exist_ok=True)
        log_file = open(run_log_path, "a", encoding="utf-8")
    except OSError as exc:
        raise CoreArtifactWriteError(
            "failed to prepare persistent run artifact directories"
        ) from exc
    if layout.external_overrides:
        print(
            "[eval_astra] non-unified artifact paths explicitly selected: "
            + ", ".join(layout.external_overrides),
            file=sys.stderr,
            flush=True,
        )

    planned_total = len(tasks) * args.eval_episodes * args.repeats
    planned_manifest = [
        {"repeat_id": repeat_id, "task": task, "episode_id": episode_id}
        for task in tasks
        for repeat_id in range(1, args.repeats + 1)
        for episode_id in episode_ids
    ]
    start_status = _git_value("status", "--porcelain=v1")
    start_head = _git_value("rev-parse", "HEAD")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    manifest = {
        "evaluation_id": evaluation_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "initializing",
        "experiment_name": "Astra-Direct-RGB",
        "policy_type": {"codex": "codex_app_server", "mock": "mock", "manual": "manual"}[args.policy],
        "requested_model": args.codex_model if args.policy == "codex" else None,
        "resolved_model": None,
        "codex_cli_version": None,
        "reasoning_effort": args.codex_reasoning if args.policy == "codex" else None,
        "code_head": start_head,
        "code_head_evidence": {
            "status": "unknown" if start_head is None else "known",
            "error": _GIT_QUERY_FAILURES.get("git rev-parse HEAD"),
        },
        "workspace_dirty_at_start": (None if start_status is None else bool(start_status)),
        "workspace_status_at_start": (None if start_status is None else start_status.splitlines()),
        "workspace_status_evidence": {
            "status": "unknown" if start_status is None else "known",
            "error": _GIT_QUERY_FAILURES.get("git status --porcelain=v1"),
        },
        "run_mode": args.run_mode,
        "protocol": {
            "budget_protocol": args.budget_protocol,
            "step_limit_config_path": os.path.abspath(args.step_limit_config),
            "parsed_step_limit_config": step_limits,
            "resolved_waypoint_budget_by_task": budgets,
            "task_groups": task_groups,
            "main_tasks": list(EVAL_TASKS),
            "sanity_check_tasks": list(SANITY_CHECK_TASKS),
            "selected_tasks": tasks,
            "planned_episodes_per_task_per_repeat": args.eval_episodes,
            "repeats": args.repeats,
            "repeat_ids": list(range(1, args.repeats + 1)),
            "episode_ids": episode_ids,
            "planned_episode_units": planned_manifest,
            "action_horizon": 1,
            "history_enabled": args.policy == "codex",
            "history_mode": "native_episode_session" if args.policy == "codex" else "not_applicable",
            "application_replays_full_history": False,
            "session_scope": args.session_mode if args.policy == "codex" else None,
            "motion_guidance": "prompt_only" if args.policy == "codex" else None,
            "motion_prompt_profile": args.motion_prompt_profile if args.policy == "codex" else None,
            "programmatic_motion_limiter_added": False,
            "collision_mode": args.collision_mode,
            "collision_action_space_matches_predictive_baseline": args.collision_mode == "predict",
            "position_tolerance_m": args.position_tolerance_m,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
            "success_reward_threshold_exclusive": 99.0,
            "success_rate_units": "fraction in [0, 1]",
            "repeated_episode_ids_are_new_scenes": False,
        },
        "observation_contract": {
            "inputs": ["language instruction", "front RGB", "left_shoulder RGB",
                       "right_shoulder RGB", "wrist RGB", "absolute EEF pose XYZW",
                       "gripper open state"],
            "excluded": ["depth", "point cloud", "object pose", "segmentation",
                         "handles", "demonstration actions", "task internals",
                         "success detector", "extra cameras"],
            "claim": "four RGB views plus robot state direct-pose policy; not information-identical to point-cloud BridgeVLA",
            "camera_order": ["front", "left_shoulder", "right_shoulder", "wrist"],
            "configured_image_size": None,
            "policy_input_resolution": None,
            "recording_source_resolution": [args.recording_width, args.recording_height],
            "recording_output_resolution": [args.recording_width * 2, args.recording_height * 2],
            "recording_fps": args.recording_fps,
            "observed_image_shapes_by_episode": [],
            "video_tile_resolution_is_model_input_resolution": False,
        },
        "action_contract": {
            "position": "absolute world-frame meters",
            "orientation": "absolute unit quaternion XYZW",
            "gripper": "0 close, 1 open; applied after arm motion",
            "collision_mode": args.collision_mode,
            "targets_per_policy_decision_max": 1,
            "planner_substeps_count_as_policy_targets": False,
            "planner_execution_target_source": "Astra passive calculation matching EndEffectorPoseViaPlanning2 XYZ clipping",
        },
        "seed": {
            "base_seed": args.seed,
            "controlled_sources": ["Python random", "NumPy legacy global RNG"] if args.seed is not None else [],
            "pyrep_coppeliasim_seed": "not explicitly set by Astra",
            "determinism_claim": False,
        },
        "dataset": {
            "root": os.path.abspath(args.eval_datafolder) if args.eval_datafolder else None,
            "version": None,
        },
        "simulator": {
            "dependency_selection": "pending",
            "rlbench": None,
            "pyrep": None,
            "headless": args.headless,
            "simulation_timestep_seconds": None,
            "standard_sim_stack": not args.allow_shared_sim_stack,
            "shared_sim_stack_allowed": args.allow_shared_sim_stack,
        },
        "codex_tool_boundary": {
            "sandbox": "read-only",
            "tool_call_detection": "reject output after detecting tool events",
            "backend": "codex_app_server",
            "thread_id_and_session_id_source": "thread/start response",
            "tool_disable_verified": False,
            "strong_tool_isolation_claim": False,
            "automatic_service_retry_added": False,
            "underlying_model_request_count": None,
        },
        "output": {
            "run_directory": run_dir,
            "artifact_layout": layout.as_dict(),
            "manifest": manifest_path,
            "run_log": run_log_path,
            "policy_work_directory": policy_work_dir,
            "episode_artifacts_directory": output_dir,
            "recording_enabled": args.record_video,
        },
        "arguments": vars(args),
        "episode_results": [],
    }
    try:
        _write_json_atomic(manifest_path, manifest)
    except Exception:
        log_file.close()
        raise

    run_log_errors = []

    def emit(record):
        record = {"evaluation_id": evaluation_id, **record}
        line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        try:
            log_file.write(line + "\n")
            log_file.flush()
        except Exception as exc:
            from astra.errors import sanitize_diagnostic
            run_log_errors.append(sanitize_diagnostic(
                f"{type(exc).__name__}: {exc}"
            )["text"])

    def save_manifest():
        _write_json_atomic(manifest_path, manifest)

    def policy_context(repeat_id, task_name, episode_id):
        return {
            "evaluation_id": evaluation_id,
            "repeat_id": repeat_id,
            "task": task_name,
            "episode_id": episode_id,
            "waypoint_budget": budgets[task_name],
        }

    results = []
    active_recorder = None
    active_policy = None
    try:
        try:
            sim_paths = _add_project_paths(args.allow_shared_sim_stack)
        except Exception as exc:
            manifest["status"] = "incomplete"
            manifest["initialization_error"] = {
                "error_class": "infrastructure_error",
                "type": type(exc).__name__,
                "message": _safe_error_message(exc),
            }
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            save_manifest()
            try:
                emit({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "initialization_error",
                    **manifest["initialization_error"],
                })
            except OSError as log_error:
                manifest["run_log_error"] = (
                    f"{type(log_error).__name__}: {log_error}"
                )
                save_manifest()
            _write_json_atomic(os.path.join(run_dir, "run_summary.json"), {
                "evaluation_id": evaluation_id, "completion_status": "incomplete",
                "infrastructure_errors": 1, "error": manifest["initialization_error"],
            })
            raise

        import numpy as np
        from astra.action_adapter import AstraActionAdapter
        from astra.codex_policy import CodexAstraPolicy
        from astra.episode_recorder import EpisodeRecorder
        from astra.mock_policy import ManualPolicy, MockPolicy
        from astra.observation_adapter import AstraObservationAdapter
        from astra.sim_paths import module_identity
        from astra.visualization import orientation_error_degrees

        # Simulator-dependent imports remain deferred so --help and schema tests
        # work without CoppeliaSim, RLBench, or compiled PyRep extensions.
        import pyrep
        import rlbench
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.backend import task as rlbench_task
        from rlbench.backend.utils import task_file_to_task_class
        from astra.env import AstraRLBenchEnv
        from utils.peract_utils_rlbench import CAMERAS, DATA_FOLDER, IMAGE_SIZE
        from utils.rlbench_planning import EndEffectorPoseViaPlanning2, MoveArmThenGripper2
        from bridgevla.libs.peract.helpers import utils
        from yarr.agents.agent import ActResult

        simulator_roots = {
            "rlbench": sim_paths.get("rlbench_sim_stack"),
            "pyrep": sim_paths.get("pyrep_sim_stack"),
        }
        module_info = {
            "rlbench": module_identity(rlbench),
            "pyrep": module_identity(pyrep),
        }
        for name, module in (("rlbench", rlbench), ("pyrep", pyrep)):
            root = simulator_roots[name]
            module_file = module_info[name]["file"]
            if root and module_file and not os.path.commonpath(
                    [os.path.realpath(root), os.path.realpath(module_file)]
            ) == os.path.realpath(root):
                raise RuntimeError(
                    f"{name} resolved outside the selected simulator stack: {module_file}"
                )
        manifest["simulator"].update({
            "dependency_selection": "dedicated" if not args.allow_shared_sim_stack else "nonstandard_shared_allowed",
            "path_setup": sim_paths,
            "coppeliasim_root": sim_paths.get("coppeliasim_root"),
            "coppeliasim_root_exists": sim_paths.get("coppeliasim_root_exists"),
            "display": os.environ.get("DISPLAY"),
            "rlbench": module_info["rlbench"],
            "pyrep": module_info["pyrep"],
            "configured_camera_names": list(CAMERAS),
            "configured_policy_rgb_resolution_hw": [IMAGE_SIZE, IMAGE_SIZE],
            "action_mode": "MoveArmThenGripper2(EndEffectorPoseViaPlanning2, Discrete)",
            "reward_success_semantics": "RLBench reward > 99 (verified against eval.py sparse success check)",
        })
        manifest["observation_contract"]["configured_image_size"] = [IMAGE_SIZE, IMAGE_SIZE]
        manifest["observation_contract"]["policy_input_resolution"] = [IMAGE_SIZE, IMAGE_SIZE]
        data_root = args.eval_datafolder or DATA_FOLDER
        manifest["dataset"]["root"] = os.path.abspath(data_root)
        manifest["dataset"]["version"] = _git_value("-C", data_root, "rev-parse", "HEAD")

        task_files = {
            name[:-3] for name in os.listdir(rlbench_task.TASKS_PATH)
            if name.endswith(".py") and name != "__init__.py"
        }
        task_classes = []
        for task_name in tasks:
            if task_name not in task_files:
                raise ValueError(
                    f"Task {task_name!r} is not recognised by the selected RLBench install"
                )
            task_classes.append(task_file_to_task_class(task_name))

        manual_action = _parse_manual_action(args.manual_action)
        if args.policy == "codex":
            policy = CodexAstraPolicy(
                model=args.codex_model,
                reasoning_effort=args.codex_reasoning,
                timeout=args.codex_timeout,
                work_root=policy_work_dir,
                collision_mode=args.collision_mode,
            )
            manifest["codex_cli_version"] = policy.codex_cli_version
            manifest["codex_cli_version_probe_invocation_count"] = (
                policy.codex_cli_version_probe_invocation_count
            )
        elif args.policy == "manual":
            policy = ManualPolicy(manual_action)
        else:
            policy = MockPolicy()
        active_policy = policy

        manifest["requested_model"] = args.codex_model if args.policy == "codex" else None
        manifest["reasoning_effort"] = args.codex_reasoning if args.policy == "codex" else None
        manifest["status"] = "running"
        save_manifest()

        obs_config = utils.create_obs_config(
            CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name=""
        )
        action_mode = MoveArmThenGripper2(EndEffectorPoseViaPlanning2(), Discrete())
        eval_env = AstraRLBenchEnv(
            task_classes=task_classes,
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=data_root,
            episode_length=budgets[tasks[0]],
            headless=args.headless,
            swap_task_every=args.eval_episodes * args.repeats,
            include_lang_goal_in_obs=True,
            time_in_state=False,
            record_every_n=-1,
        )
        eval_env.eval = True
        observation_adapter = AstraObservationAdapter()
        action_adapter = AstraActionAdapter(collision_mode=args.collision_mode)
        env_launched = False
        try:
            eval_env.launch()
            env_launched = True
            try:
                manifest["simulator"]["simulation_timestep_seconds"] = float(
                    eval_env._task._scene.pyrep.get_simulation_timestep()
                )
            except Exception:
                manifest["simulator"]["simulation_timestep_seconds"] = None
            save_manifest()
            for task_index, task_name in enumerate(tasks):
                eval_env._episode_length = budgets[task_name]
                for repeat_id in range(1, args.repeats + 1):
                    for episode_order, episode_id in enumerate(episode_ids):
                        episode_wall_start = time.monotonic()
                        instruction = ""
                        raw_obs = None
                        episode_success = False
                        episode_reward = 0.0
                        termination = "unknown"
                        error_class = None
                        error_reason = None
                        error_message = None
                        policy_decisions = 0
                        cli_calls = 0
                        env_action_attempts = 0
                        policy_time = 0.0
                        cli_time = 0.0
                        action_execution_time = 0.0
                        token_observations = []
                        metadata_mismatch_count = 0
                        latest_resolved_model = None
                        latest_cli_version = None
                        app_server_turns = 0
                        app_server_requests = 0
                        app_server_sessions_created = 0
                        latest_thread_id = None
                        latest_session_id = None
                        recorder = EpisodeRecorder(
                            output_root=output_dir,
                            task=task_name,
                            episode=episode_id,
                            model=args.codex_model if args.policy == "codex" else "not_applicable",
                            reasoning=args.codex_reasoning if args.policy == "codex" else "not_applicable",
                            record_video=args.record_video,
                            recording_width=args.recording_width,
                            recording_height=args.recording_height,
                            fps=args.recording_fps,
                            policy_type=manifest["policy_type"],
                            collision_mode=args.collision_mode,
                            run_context={
                                "evaluation_id": evaluation_id,
                                "repeat_id": repeat_id,
                                "task_group": task_groups[task_name],
                                "episode_id": episode_id,
                                "run_mode": args.run_mode,
                                "budget_protocol": args.budget_protocol,
                                "waypoint_budget": budgets[task_name],
                                "action_horizon": 1,
                                "history_enabled": args.policy == "codex",
                                "history_mode": "native_episode_session" if args.policy == "codex" else "not_applicable",
                                "application_replays_full_history": False,
                                "session_scope": args.session_mode if args.policy == "codex" else None,
                                "motion_guidance": "prompt_only" if args.policy == "codex" else None,
                                "motion_prompt_profile": args.motion_prompt_profile if args.policy == "codex" else None,
                                "programmatic_motion_limiter_added": False,
                                "policy_input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
                                "recording_source_resolution": [args.recording_width, args.recording_height],
                                "recording_output_resolution": [args.recording_width * 2, args.recording_height * 2],
                                "recording_fps": args.recording_fps,
                                "codex_cli_version_probe_invocation_count": (
                                    policy.codex_cli_version_probe_invocation_count
                                    if args.policy == "codex" else 0
                                ),
                            },
                        )
                        active_recorder = recorder
                        actual_input_shapes = None
                        eval_env._last_exception = None
                        try:
                            if args.seed is not None:
                                effective_seed = (
                                    int(args.seed) + repeat_id * 1_000_000
                                    + task_index * 10_000 + episode_order
                                )
                                random.seed(effective_seed)
                                np.random.seed(effective_seed % (2 ** 32))
                            else:
                                effective_seed = None
                            eval_env.reset_to_demo(episode_id)
                            try:
                                active_task = eval_env._task._task.get_name()
                            except Exception:
                                active_task = None
                            if active_task and active_task != task_name:
                                raise RuntimeError(
                                    f"environment task mismatch: expected {task_name}, got {active_task}"
                                )
                            instruction = eval_env._lang_goal
                            recorder.set_instruction(instruction)
                            raw_obs = eval_env.last_raw_observation
                            if raw_obs is None:
                                raise RuntimeError(
                                    "reset_to_demo() did not cache an RLBench Observation"
                                )
                            context = policy_context(repeat_id, task_name, episode_id)
                            if hasattr(policy, "set_evaluation_context"):
                                policy.set_evaluation_context(**context)
                            policy.reset(instruction)
                            actual_input_shapes = {
                                camera: list(np.asarray(getattr(raw_obs, camera + "_rgb")).shape[:2])
                                for camera in CAMERAS
                            }
                            manifest["observation_contract"]["observed_image_shapes_by_episode"].append({
                                **context, "image_shapes_hw": actual_input_shapes,
                            })
                            recorder.initialize_camera(eval_env._task._scene, raw_obs)
                        except Exception as exc:
                            error_class, error_reason = _classify_error(exc, "reset")
                            error_message = _safe_error_message(exc)
                            termination = "infrastructure_error_during_reset"

                        if raw_obs is not None and error_class is None:
                            episode_terminal = False
                            for step_id in range(budgets[task_name]):
                                policy_output = None
                                final_action = None
                                current_pose = None
                                policy_metadata = None
                                metadata_mismatch = False
                                phase = "observation"
                                call_elapsed = None
                                try:
                                    astra_observation = observation_adapter.adapt(raw_obs, instruction)
                                    current_pose = list(astra_observation.eef_pose)
                                    if tuple(astra_observation.images) != (
                                            "front", "left_shoulder", "right_shoulder", "wrist"):
                                        raise RuntimeError(
                                            "policy input cameras differ from the four configured RGB views"
                                        )
                                    phase = "policy"
                                    policy_decisions += 1
                                    if hasattr(policy, "last_metadata"):
                                        policy.last_metadata = None
                                    started = time.monotonic()
                                    policy_output = policy.act(astra_observation)
                                    call_elapsed = time.monotonic() - started
                                    policy_time += call_elapsed
                                    expected_metadata = {
                                        **context, "step_id": step_id,
                                    }
                                    policy_metadata, metadata_mismatch = _metadata_for_step(
                                        policy, expected_metadata
                                    )
                                    if metadata_mismatch:
                                        metadata_mismatch_count += 1
                                    if policy_metadata is not None:
                                        cli_calls += int(policy_metadata.get("cli_invocation_count", 0) or 0)
                                        app_server_turns += int(policy_metadata.get("turn_id") is not None)
                                        app_server_requests += int(policy_metadata.get("app_server_request_count", 0) or 0)
                                        app_server_sessions_created += int(policy_metadata.get("app_server_session_create_count", 0) or 0)
                                        latest_thread_id = policy_metadata.get("thread_id") or latest_thread_id
                                        latest_session_id = policy_metadata.get("control_session_id") or latest_session_id
                                        cli_time += float(policy_metadata.get("latency_seconds") or 0.0)
                                        token_observations.append(_token_counts(policy_metadata))
                                        latest_resolved_model = policy_metadata.get("resolved_model") or latest_resolved_model
                                        latest_cli_version = policy_metadata.get("codex_cli_version") or latest_cli_version
                                    phase = "action_validation"
                                    final_action = action_adapter.adapt(policy_output)
                                    scene = eval_env._task._scene
                                    try:
                                        bounds_min = [scene._workspace_minx, scene._workspace_miny,
                                                      scene._workspace_minz]
                                        bounds_max = [scene._workspace_maxx, scene._workspace_maxy,
                                                      scene._workspace_maxz]
                                        action_adapter.set_workspace_bounds(bounds_min, bounds_max)
                                    except Exception:
                                        action_adapter.last_diagnostics["effective_target_source"] = None
                                except Exception as exc:
                                    if phase == "policy":
                                        call_elapsed = time.monotonic() - started
                                        policy_time += call_elapsed
                                        expected_metadata = {**context, "step_id": step_id}
                                        policy_metadata, metadata_mismatch = _metadata_for_step(
                                            policy, expected_metadata
                                        )
                                        if metadata_mismatch:
                                            metadata_mismatch_count += 1
                                        if policy_metadata is not None:
                                            cli_calls += int(policy_metadata.get("cli_invocation_count", 0) or 0)
                                            app_server_turns += int(policy_metadata.get("turn_id") is not None)
                                            app_server_requests += int(policy_metadata.get("app_server_request_count", 0) or 0)
                                            app_server_sessions_created += int(policy_metadata.get("app_server_session_create_count", 0) or 0)
                                            latest_thread_id = policy_metadata.get("thread_id") or latest_thread_id
                                            latest_session_id = policy_metadata.get("control_session_id") or latest_session_id
                                            cli_time += float(policy_metadata.get("latency_seconds") or 0.0)
                                            token_observations.append(_token_counts(policy_metadata))
                                            latest_resolved_model = policy_metadata.get("resolved_model") or latest_resolved_model
                                            latest_cli_version = policy_metadata.get("codex_cli_version") or latest_cli_version
                                    error_class, error_reason = _classify_error(
                                        exc, phase, policy_metadata
                                    )
                                    error_message = _safe_error_message(exc)
                                    recorder.record_step_failure(
                                        step_id, raw_obs, error_message, policy_metadata
                                    )
                                    termination = f"{error_class}:{error_reason}"
                                    episode_terminal = True
                                    break

                                diagnostics = dict(action_adapter.last_diagnostics or {})
                                recorder.begin_step(
                                    step_id, raw_obs, action=policy_output,
                                    final_action=final_action,
                                    policy_metadata=policy_metadata,
                                )
                                eval_env._last_exception = None
                                transition = None
                                unexpected_error = None
                                scene = eval_env._task._scene
                                attempt_started = time.monotonic()
                                env_action_attempts += 1
                                try:
                                    with recorder.capture_during_execution(scene):
                                        transition = eval_env.step(ActResult(
                                            np.asarray(final_action, dtype=np.float64)
                                        ))
                                except Exception as exc:
                                    unexpected_error = exc
                                step_action_execution_time = time.monotonic() - attempt_started
                                action_execution_time += step_action_execution_time
                                planner_error = getattr(eval_env, "_last_exception", None)
                                if planner_error is None:
                                    planner_error = unexpected_error
                                if planner_error is not None:
                                    error_class, error_reason = _classify_error(
                                        planner_error, "environment_step", policy_metadata
                                    )
                                    error_message = _safe_error_message(planner_error)
                                    raw_after = None
                                else:
                                    raw_after = eval_env.last_raw_observation
                                    if raw_after is None:
                                        error_class = "infrastructure_error"
                                        error_reason = "missing_raw_observation"
                                        error_message = (
                                            "RLBench step did not cache a new returned Observation"
                                        )
                                episode_reward = float(transition.reward) if transition is not None else 0.0
                                episode_success = episode_reward > 99.0
                                if transition is None and error_message is None:
                                    error_class = "infrastructure_error"
                                    error_reason = "missing_transition"
                                    error_message = "RLBench step returned no transition"
                                elif (transition is not None and transition.terminal
                                      and not episode_success and error_message is None):
                                    error_class = "policy_failure"
                                    error_reason = "environment_terminal_without_success"
                                    error_message = "RLBench episode terminated without success"
                                after_pose_value = (
                                    getattr(raw_after, "gripper_pose", None)
                                    if raw_after is not None else None
                                )
                                after_pose = (
                                    np.asarray(after_pose_value, dtype=np.float64).reshape(-1)
                                    if after_pose_value is not None else None
                                )
                                validated = diagnostics.get("validated_action", {})
                                effective_position = diagnostics.get("effective_planner_target")
                                requested_position = diagnostics.get("raw_policy_action", {}).get("position")
                                effective_position_error = None
                                requested_position_error = None
                                orientation_error = None
                                if after_pose is not None and after_pose.shape == (7,):
                                    if effective_position is not None:
                                        effective_position_error = float(np.linalg.norm(
                                            after_pose[:3] - np.asarray(effective_position, dtype=float)
                                        ))
                                    if requested_position is not None:
                                        requested_position_error = float(np.linalg.norm(
                                            after_pose[:3] - np.asarray(requested_position, dtype=float)
                                        ))
                                    orientation_error = orientation_error_degrees(
                                        validated.get("quaternion_xyzw"), after_pose[3:7]
                                    )
                                position_reached = (
                                    effective_position_error is not None
                                    and effective_position_error < args.position_tolerance_m
                                ) if effective_position_error is not None else None
                                pose_reached = (
                                    bool(position_reached)
                                    and orientation_error is not None
                                    and orientation_error < args.orientation_tolerance_deg
                                ) if position_reached is not None else None
                                planner_returned = bool(planner_error is None and raw_after is not None)
                                is_terminal = bool(
                                    transition is None
                                    or (transition.terminal if transition is not None else False)
                                    or error_message is not None
                                )
                                execution = {
                                    "task": task_name,
                                    "task_group": task_groups[task_name],
                                    "repeat_id": repeat_id,
                                    "episode_id": episode_id,
                                    "step_id": step_id,
                                    "observation_id": (policy_metadata or {}).get("observation_id"),
                                    "action_id": (policy_metadata or {}).get("action_id"),
                                    "control_session_id": (policy_metadata or {}).get("control_session_id"),
                                    "thread_id": (policy_metadata or {}).get("thread_id"),
                                    "turn_id": (policy_metadata or {}).get("turn_id"),
                                    "instruction": instruction,
                                    "eef_pose_before": current_pose,
                                    "actual_eef_pose": after_pose_value,
                                    "raw_policy_action": diagnostics.get("raw_policy_action"),
                                    "validated_action": diagnostics.get("validated_action"),
                                    "effective_planner_target": effective_position,
                                    "effective_target_source": diagnostics.get("effective_target_source"),
                                    "workspace_clip": diagnostics.get("workspace_clip"),
                                    "quaternion_was_normalized": diagnostics.get("quaternion_was_normalized"),
                                    "target_position": validated.get("position"),
                                    "target_quaternion_xyzw": validated.get("quaternion_xyzw"),
                                    "collision_mode": args.collision_mode,
                                    "ignore_collisions": final_action[-1],
                                    "gripper_before": bool(raw_obs.gripper_open),
                                    "gripper_command": int(policy_output.gripper),
                                    "final_9d_action": list(final_action),
                                    "gripper_after": bool(raw_after.gripper_open) if raw_after is not None else None,
                                    "reward": episode_reward,
                                    "terminal": is_terminal,
                                    "success": episode_success,
                                    "policy_decision_count": 1,
                                    "cli_invocation_count": (
                                        int(policy_metadata.get("cli_invocation_count", 0) or 0)
                                        if policy_metadata else None
                                    ),
                                    "underlying_model_request_count": None,
                                    "policy_latency_seconds": call_elapsed,
                                    "cli_latency_seconds": (policy_metadata or {}).get("latency_seconds"),
                                    "action_execution_seconds": step_action_execution_time,
                                    "token_usage": (policy_metadata or {}).get("token_usage"),
                                    "tool_call_detected": (policy_metadata or {}).get("tool_call_detected"),
                                    "position_error_to_requested_m": requested_position_error,
                                    "position_error_to_effective_target_m": effective_position_error,
                                    "orientation_error_deg": orientation_error,
                                    "position_reached": position_reached,
                                    "pose_reached": pose_reached,
                                    "target_reached": position_reached,
                                    "position_tolerance_m": args.position_tolerance_m,
                                    "orientation_tolerance_deg": args.orientation_tolerance_deg,
                                    "planner_returned": planner_returned,
                                    "environment_step_returned": transition is not None,
                                    "error_category": {
                                        "class": error_class,
                                        "reason": error_reason,
                                    },
                                    "error": error_message,
                                    "error_class": error_class,
                                    "error_reason": error_reason,
                                }
                                execution = recorder.finish_step(
                                    execution, raw_observation_after=raw_after, scene=scene
                                )
                                if (args.policy == "codex"
                                        and hasattr(policy, "record_execution_feedback")
                                        and (policy_metadata or {}).get("action_id")):
                                    from astra.execution_feedback import build_execution_feedback
                                    observation_id_after = (
                                        policy.observation_id_for_step(step_id + 1)
                                        if raw_after is not None else None
                                    )
                                    feedback = build_execution_feedback(
                                        execution,
                                        action_id=policy_metadata["action_id"],
                                        observation_id_before=policy_metadata.get("observation_id"),
                                        observation_id_after=observation_id_after,
                                    )
                                    policy.record_execution_feedback(feedback)
                                emit({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "kind": "step", **execution,
                                    "policy_output": policy_output,
                                    "policy_metadata": policy_metadata,
                                    "metadata_mismatch_discarded": metadata_mismatch,
                                })

                                if error_message is not None:
                                    termination = f"{error_class}:{error_reason}"
                                    episode_terminal = True
                                elif episode_success:
                                    termination = "task_success"
                                    episode_terminal = True
                                if raw_after is not None:
                                    raw_obs = raw_after
                                if episode_terminal:
                                    break

                            if not episode_terminal and error_class is None:
                                error_class = "policy_failure"
                                error_reason = "action_budget_exhausted"
                                termination = "waypoint_budget_exhausted"

                        if args.policy == "codex" and hasattr(policy, "end_episode"):
                            policy.end_episode(termination)

                        if error_class in ("infrastructure_error", "unknown"):
                            manifest["status"] = "incomplete"
                        if args.policy == "codex":
                            manifest["resolved_model"] = latest_resolved_model
                            manifest["codex_cli_version"] = latest_cli_version or policy.codex_cli_version

                        valid_token_calls = [item for item in token_observations if item is not None]
                        token_fields = ("input_tokens", "output_tokens", "reasoning_tokens")
                        token_totals = {
                            key: (
                                sum(item[key] for item in valid_token_calls)
                                if len(valid_token_calls) == cli_calls
                                and cli_calls > 0
                                and all(item.get(key) is not None for item in valid_token_calls)
                                else None
                            ) for key in token_fields
                        }
                        attempted = True
                        evaluable = error_class not in ("infrastructure_error", "unknown")
                        episode_result = {
                            "task": task_name,
                            "task_group": task_groups[task_name],
                            "repeat_id": repeat_id,
                            "episode_id": episode_id,
                            "random_seed": effective_seed,
                            "planned": True,
                            "attempted": attempted,
                            "evaluable": evaluable,
                            "success": bool(episode_success) if evaluable else None,
                            "observed_task_success": bool(episode_success),
                            "reward": episode_reward,
                            "steps": policy_decisions,
                            "waypoint_budget": budgets[task_name],
                            "policy_decision_count": policy_decisions,
                            "cli_invocation_count": cli_calls if args.policy == "codex" else 0,
                            "app_server_session_create_count": app_server_sessions_created if args.policy == "codex" else 0,
                            "app_server_turn_count": app_server_turns if args.policy == "codex" else 0,
                            "app_server_request_count": app_server_requests if args.policy == "codex" else 0,
                            "control_session_id": latest_session_id,
                            "thread_id": latest_thread_id,
                            "underlying_model_request_count": None,
                            "environment_action_attempt_count": env_action_attempts,
                            "simulation_step_count": recorder.simulation_step_count,
                            "simulation_time_seconds": recorder.simulation_time_seconds,
                            "policy_time_seconds": policy_time,
                            "cli_time_seconds": None,
                            "app_server_turn_time_seconds": cli_time if args.policy == "codex" else None,
                            "action_execution_time_seconds": action_execution_time,
                            "episode_wall_clock_seconds": time.monotonic() - episode_wall_start,
                            "token_usage": token_totals if cli_calls else None,
                            "token_usage_complete": bool(
                                cli_calls > 0 and len(valid_token_calls) == cli_calls
                                and all(all(item.get(key) is not None for key in token_fields)
                                        for item in valid_token_calls)
                            ),
                            "metadata_mismatch_count": metadata_mismatch_count,
                            "termination_reason": termination,
                            "error_class": error_class,
                            "error_reason": error_reason,
                            "error": error_message,
                            "recording_error": recorder.recording_error,
                            "redundant_record_errors": list(recorder.redundant_record_errors),
                            "core_records_complete": True,
                            "output_directory": str(recorder.run_dir),
                        }
                        summary = {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "episode_summary",
                            **episode_result,
                            "requested_model": args.codex_model if args.policy == "codex" else None,
                            "resolved_model": latest_resolved_model,
                            "codex_cli_version": latest_cli_version or (
                                policy.codex_cli_version if args.policy == "codex" else None
                            ),
                            "reasoning_effort": args.codex_reasoning if args.policy == "codex" else None,
                            "collision_mode": args.collision_mode,
                            "history_enabled": args.policy == "codex",
                            "history_mode": "native_episode_session" if args.policy == "codex" else "not_applicable",
                            "application_replays_full_history": False,
                            "session_scope": args.session_mode if args.policy == "codex" else None,
                            "motion_guidance": "prompt_only" if args.policy == "codex" else None,
                            "motion_prompt_profile": args.motion_prompt_profile if args.policy == "codex" else None,
                            "programmatic_motion_limiter_added": False,
                            "policy_input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
                            "recording_source_resolution": [args.recording_width, args.recording_height],
                            "recording_output_resolution": [args.recording_width * 2, args.recording_height * 2],
                            "recording_fps": args.recording_fps,
                            "image_shapes_hw": actual_input_shapes,
                        }
                        # The episode summary is authoritative; save the result and
                        # manifest before attempting optional video encoding.
                        recorder.write_summary(summary)
                        episode_result["recording_error"] = recorder.recording_error
                        episode_result["output_directory"] = str(recorder.run_dir)
                        results.append(episode_result)
                        manifest["episode_results"] = results
                        manifest["last_updated_at"] = datetime.now(timezone.utc).isoformat()
                        save_manifest()
                        recorder.finalize_video(summary)
                        summary["recording_error"] = recorder.recording_error
                        summary["recording_status"] = recorder.recording_status
                        recorder.update_summary(summary)
                        episode_result["recording_error"] = recorder.recording_error
                        episode_result["redundant_record_errors"] = list(
                            recorder.redundant_record_errors
                        )
                        manifest["episode_results"] = results
                        save_manifest()
                        emit(summary)
                        recorder.close()
                        active_recorder = None
                        if error_class in ("infrastructure_error", "unknown"):
                            break
                    if manifest["status"] == "incomplete":
                        break
                if manifest["status"] == "incomplete":
                    break
        finally:
            if env_launched:
                eval_env.shutdown()

        aggregates = aggregate_results(
            results,
            planned_by_task={task: args.eval_episodes for task in tasks},
            repeat_ids=list(range(1, args.repeats + 1)),
            selected_tasks=tasks,
            expected_episode_ids={task: episode_ids for task in tasks},
        )
        planned = planned_total
        attempted = len(results)
        infrastructure_count = sum(
            result.get("error_class") == "infrastructure_error" for result in results
        )
        unknown_count = sum(result.get("error_class") == "unknown" for result in results)
        status = (
            "complete" if attempted == planned and infrastructure_count == 0
            and unknown_count == 0 and aggregates["invalid_identity_rows"] == 0
            and not aggregates["unexpected_experiment_units"]
            and not aggregates["duplicate_experiment_units"]
            and all(
                row["completion_status"] == "complete"
                for row in aggregates["task_results"]
            )
            else "incomplete"
        )
        summary = {
            "evaluation_id": evaluation_id,
            "experiment_name": "Astra-Direct-RGB",
            "run_mode": args.run_mode,
            "policy_type": manifest["policy_type"],
            "requested_model": manifest["requested_model"],
            "resolved_model": manifest["resolved_model"],
            "codex_cli_version": manifest["codex_cli_version"],
            "reasoning_effort": manifest["reasoning_effort"],
            "completion_status": status,
            "planned_episodes": planned,
            "attempted_episodes": attempted,
            "evaluable_episodes": sum(result.get("evaluable") is True for result in results),
            "successes": sum(result.get("success") is True for result in results),
            "policy_failures": sum(result.get("error_class") == "policy_failure" for result in results),
            "infrastructure_errors": infrastructure_count,
            "unknown_errors": unknown_count,
            "recording_errors": sum(bool(result.get("recording_error")) for result in results),
            "run_log_errors": list(run_log_errors),
            "success_rate_units": "fraction in [0, 1]",
            "episode_results": results,
            **aggregates,
        }
        manifest["status"] = status
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["run_summary"] = {
            key: value for key, value in summary.items()
            if key not in ("episode_results",)
        }
        save_manifest()
        _write_json_atomic(os.path.join(run_dir, "run_summary.json"), summary)
        emit({"timestamp": datetime.now(timezone.utc).isoformat(),
              "kind": "run_summary", **summary})
        return results
    except Exception as exc:
        from astra.errors import safe_exception_record

        manifest["status"] = "incomplete"
        error_class, error_reason = _classify_error(exc, "run")
        manifest["runtime_error"] = {
            **safe_exception_record(exc),
            "error_class": error_class,
            "error_reason": error_reason,
        }
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        try:
            save_manifest()
        except Exception:
            manifest["runtime_manifest_write_failed"] = True
        try:
            exception_aggregates = aggregate_results(
                results,
                planned_by_task={task: args.eval_episodes for task in tasks},
                repeat_ids=list(range(1, args.repeats + 1)),
                selected_tasks=tasks,
                expected_episode_ids={task: episode_ids for task in tasks},
            )
            _write_json_atomic(os.path.join(run_dir, "run_summary.json"), {
                "evaluation_id": evaluation_id,
                "completion_status": "incomplete",
                "planned_episodes": planned_total,
                "attempted_episodes": len(results),
                "infrastructure_errors": sum(
                    result.get("error_class") == "infrastructure_error"
                    for result in results
                ),
                "unknown_errors": sum(
                    result.get("error_class") == "unknown" for result in results
                ),
                "episode_results": results,
                "run_log_errors": list(run_log_errors),
                "error": manifest["runtime_error"],
                **exception_aggregates,
            })
        except Exception as summary_error:
            raise RuntimeError(
                "evaluation failed and run_summary.json could not be persisted; "
                "the run cannot be claimed complete"
            ) from summary_error
        raise
    finally:
        if active_policy is not None and hasattr(active_policy, "close"):
            try:
                active_policy.close()
            except Exception:
                pass
        if active_recorder is not None:
            try:
                active_recorder.close()
            except Exception:
                pass
        if not log_file.closed:
            log_file.close()


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_eval(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(_safe_error_message(exc))


if __name__ == "__main__":
    main()
