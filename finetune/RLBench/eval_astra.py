"""Evaluate a mock/manual absolute-pose policy in the existing RLBench stack."""

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINETUNE_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(FINETUNE_DIR)


def _add_project_paths():
    """Make the source checkout usable without requiring eval.sh first."""
    sim_stack = os.environ.get(
        "RLBENCH_SIM_STACK",
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "RLBench_peract587"),
    )
    if sim_stack and os.path.isdir(os.path.join(sim_stack, "rlbench")):
        if sim_stack not in sys.path:
            sys.path.insert(0, sim_stack)

    # Keep the repository copies of YARR/PerAct/PyRep available in the same
    # arrangement as finetune/RLBench/eval.sh. Existing PYTHONPATH entries
    # retain their order and take precedence.
    local_paths = [
        FINETUNE_DIR,
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "point-renderer"),
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "peract_colab"),
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "YARR"),
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "peract"),
        os.path.join(FINETUNE_DIR, "GemBench"),
        os.path.join(FINETUNE_DIR, "bridgevla", "libs", "PyRep"),
    ]
    for path in reversed(local_paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    # This evaluator's ``utils`` package owns custom_rlbench_env. The shared
    # dependency paths above also contain packages named ``utils``, so keep
    # this script's directory first for unambiguous Astra-side imports.
    if SCRIPT_DIR in sys.path:
        sys.path.remove(SCRIPT_DIR)
    sys.path.insert(0, SCRIPT_DIR)


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run the Astra-compatible RLBench control loop using a mock, "
            "manual, or Codex-generated absolute EEF action."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="RLBench task names or 'all' (default: all)")
    parser.add_argument("--eval-datafolder", default=None,
                        help="RLBench demonstration dataset root")
    parser.add_argument("--start-episode", type=int, default=0,
                        help="first demonstration episode index")
    parser.add_argument("--eval-episodes", type=int, default=1,
                        help="number of episodes per task (default: 1)")
    parser.add_argument("--episode-length", type=int, default=25,
                        help="maximum policy waypoints per episode")
    parser.add_argument("--policy", choices=("mock", "manual", "codex"), default="mock",
                        help="policy implementation (default: mock)")
    parser.add_argument(
        "--manual-action", type=str, default=None,
        help='absolute action "x,y,z,qx,qy,qz,qw,g" (g is 0 or 1)',
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction,
                        default=True, help="run simulator headless (default: true)")
    parser.add_argument("--codex-model", default="gpt-6-luna",
                        help="Codex CLI model (default: gpt-6-luna)")
    parser.add_argument("--codex-reasoning", default="max",
                        help="Codex model reasoning effort (default: max)")
    parser.add_argument("--codex-timeout", type=float, default=180.0,
                        help="Codex CLI timeout in seconds (default: 180)")
    parser.add_argument(
        "--codex-work-root",
        default=os.path.join(REPO_ROOT, "tmp", "rlbench_codex_policy"),
        help="directory for per-episode Codex inference artifacts",
    )
    parser.add_argument("--log-file", default=None,
                        help="optional JSONL file for per-step logs")
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
    name = type(exc).__name__
    if name in ("IKError", "ConfigurationPathError", "InvalidActionError"):
        return name
    return name or "Error"


def _emit(record, log_file=None):
    line = json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False)
    print(line, flush=True)
    if log_file is not None:
        log_file.write(line + "\n")
        log_file.flush()


def run_eval(args):
    _add_project_paths()

    import numpy as np
    from astra.action_adapter import AstraActionAdapter
    from astra.codex_policy import CodexAstraPolicy
    from astra.mock_policy import ManualPolicy, MockPolicy
    from astra.observation_adapter import AstraObservationAdapter

    # Keep the simulator-dependent imports out of module initialization so
    # --help and the schemas remain usable without an installed RLBench stack.
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.backend import task as rlbench_task
    from rlbench.backend.utils import task_file_to_task_class
    from astra.env import AstraRLBenchEnv
    from utils.peract_utils_rlbench import CAMERAS, DATA_FOLDER, IMAGE_SIZE
    from utils.rlbench_planning import (
        EndEffectorPoseViaPlanning2,
        MoveArmThenGripper2,
    )
    from bridgevla.libs.peract.helpers import utils
    from yarr.agents.agent import ActResult
    from bridgevla.utils.rvt_utils import RLBENCH_TASKS

    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive")
    if args.start_episode < 0:
        raise ValueError("--start-episode must be non-negative")

    manual_action = _parse_manual_action(args.manual_action)
    if args.policy == "manual" and manual_action is None:
        raise ValueError("--policy manual requires --manual-action")
    if args.policy == "mock" and manual_action is not None:
        policy = MockPolicy(manual_action=manual_action)
    elif args.policy == "manual":
        policy = ManualPolicy(manual_action)
    elif args.policy == "codex":
        if manual_action is not None:
            raise ValueError("--manual-action cannot be combined with --policy codex")
        policy = CodexAstraPolicy(
            model=args.codex_model,
            reasoning_effort=args.codex_reasoning,
            timeout=args.codex_timeout,
            work_root=args.codex_work_root,
        )
    else:
        policy = MockPolicy()

    tasks = list(RLBENCH_TASKS) if args.tasks == ["all"] else args.tasks
    task_files = {
        name[:-3]
        for name in os.listdir(rlbench_task.TASKS_PATH)
        if name.endswith(".py") and name != "__init__.py"
    }
    task_classes = []
    for task_name in tasks:
        if task_name not in task_files:
            raise ValueError(f"Task {task_name!r} is not recognised by this RLBench install")
        task_classes.append(task_file_to_task_class(task_name))

    camera_resolution = [IMAGE_SIZE, IMAGE_SIZE]
    obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")
    # Astra consumes the same four RGB views and the real Observation.gripper_pose.
    # create_obs_config already enables gripper_pose and gripper_open.
    action_mode = MoveArmThenGripper2(
        EndEffectorPoseViaPlanning2(), Discrete()
    )
    eval_env = AstraRLBenchEnv(
        task_classes=task_classes,
        observation_config=obs_config,
        action_mode=action_mode,
        dataset_root=args.eval_datafolder or DATA_FOLDER,
        episode_length=args.episode_length,
        headless=args.headless,
        swap_task_every=args.eval_episodes,
        include_lang_goal_in_obs=True,
        # Astra consumes raw Observation fields; omitting the unused time
        # feature also makes a one-waypoint episode valid (length - 1 == 0).
        time_in_state=False,
        record_every_n=-1,
    )
    eval_env.eval = True

    observation_adapter = AstraObservationAdapter()
    action_adapter = AstraActionAdapter()
    log_file = None
    if args.log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.log_file)), exist_ok=True)
        log_file = open(args.log_file, "a", encoding="utf-8")

    results = []
    eval_env.launch()
    try:
        for task_name in tasks:
            eval_env._episode_length = args.episode_length
            task_successes = 0
            for episode in range(args.start_episode,
                                 args.start_episode + args.eval_episodes):
                instruction = ""
                steps = 0
                episode_reward = 0.0
                episode_success = False
                episode_error = None
                eval_env._last_exception = None
                try:
                    eval_env.reset_to_demo(episode)
                    instruction = eval_env._lang_goal
                    raw_obs = eval_env.last_raw_observation
                    if raw_obs is None:
                        raise RuntimeError(
                            "reset_to_demo() did not cache an RLBench Observation"
                        )
                    policy.reset(instruction)
                except Exception as exc:
                    episode_error = _exception_category(exc)
                    _emit({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "episode_error", "task": task_name,
                        "episode": episode, "step": 0,
                        "instruction": instruction, "current_eef_pose": None,
                        "policy_output": None, "final_action": None,
                        "reward": 0.0, "terminal": True, "success": False,
                        "error": f"{episode_error}: {exc}",
                    }, log_file)
                    results.append({
                        "task": task_name, "episode": episode, "success": False,
                        "reward": 0.0, "steps": 0, "error": episode_error,
                    })
                    continue

                for step in range(args.episode_length):
                    policy_output = None
                    final_action = None
                    current_pose = None
                    phase = "observation"
                    try:
                        astra_observation = observation_adapter.adapt(
                            raw_obs, instruction
                        )
                        current_pose = list(astra_observation.eef_pose)
                        phase = "policy"
                        policy_output = policy.act(astra_observation)
                        phase = "action_validation"
                        final_action = action_adapter.adapt(policy_output)
                    except Exception as exc:
                        episode_error = (
                            "InvalidActionError"
                            if phase == "action_validation"
                            else _exception_category(exc)
                        )
                        _emit({
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "kind": "step", "task": task_name,
                            "episode": episode, "step": step,
                            "instruction": instruction,
                            "current_eef_pose": current_pose,
                            "eef_pose_after": None,
                            "policy_output": policy_output,
                            "final_action": final_action,
                            "policy_metadata": getattr(policy, "last_metadata", None),
                            "planner_ik_status": "not_run",
                            "reward": 0.0, "terminal": True, "success": False,
                            "error": f"{episode_error}: {exc}",
                        }, log_file)
                        break

                    transition = eval_env.step(ActResult(
                        np.asarray(final_action, dtype=np.float64)
                    ))
                    steps += 1
                    episode_reward = float(transition.reward)
                    episode_success = episode_reward > 99.0
                    planner_error = getattr(eval_env, "_last_exception", None)
                    transition_terminal = bool(transition.terminal or
                                               step == args.episode_length - 1)
                    error_text = None
                    if planner_error is not None:
                        episode_error = _exception_category(planner_error)
                        error_text = f"{episode_error}: {planner_error}"
                        raw_obs = None
                        transition_terminal = True
                    else:
                        # The wrapper cached the exact Observation returned by
                        # TaskEnvironment.step() before extract_obs() processed
                        # it. Reuse it directly; do not recapture camera images.
                        raw_obs = eval_env.last_raw_observation
                        if raw_obs is None:
                            episode_error = "MissingRawObservationError"
                            error_text = (
                                "MissingRawObservationError: RLBench step did not "
                                "cache its returned Observation"
                            )
                            transition_terminal = True

                    eef_pose_after = (
                        getattr(raw_obs, "gripper_pose", None)
                        if raw_obs is not None else None
                    )
                    planner_ik_status = (
                        "failed" if planner_error is not None
                        else "succeeded" if raw_obs is not None
                        else "unknown"
                    )

                    _emit({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "kind": "step", "task": task_name,
                        "episode": episode, "step": step,
                        "instruction": instruction,
                        "current_eef_pose": current_pose,
                        "eef_pose_after": eef_pose_after,
                        "policy_output": policy_output,
                        "final_action": final_action,
                        "policy_metadata": getattr(policy, "last_metadata", None),
                        "planner_ik_status": planner_ik_status,
                        "reward": episode_reward,
                        "terminal": transition_terminal,
                        "success": episode_success,
                        "error": error_text,
                    }, log_file)

                    if episode_success or transition_terminal:
                        break

                task_successes += int(episode_success)
                results.append({
                    "task": task_name, "episode": episode,
                    "success": episode_success, "reward": episode_reward,
                    "steps": steps, "error": episode_error,
                })
                _emit({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "kind": "episode_summary", "task": task_name,
                    "episode": episode, "instruction": instruction,
                    "reward": episode_reward, "success": episode_success,
                    "steps": steps, "error": episode_error,
                }, log_file)

            _emit({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": "task_summary", "task": task_name,
                "episodes": args.eval_episodes, "successes": task_successes,
                "success_rate": task_successes / args.eval_episodes,
            }, log_file)
    finally:
        eval_env.shutdown()
        if log_file is not None:
            log_file.close()

    return results


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_eval(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
