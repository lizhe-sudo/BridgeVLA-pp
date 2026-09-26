"""Pure experiment protocol and result aggregation helpers for Astra RLBench."""

import json
import math
import statistics
from pathlib import Path


EVAL_TASKS = (
    "place_cups",
    "place_shape_in_shape_sorter",
    "put_groceries_in_cupboard",
    "stack_blocks",
    "stack_cups",
)
SANITY_CHECK_TASKS = ("meat_off_grill",)


def read_step_limits(path):
    """Read the repository's flat ``task: integer|null`` YAML config."""
    path = Path(path)
    try:
        import yaml
    except ImportError:
        # Keep ``--help`` and pure evaluator tests usable in light installs.
        result = {}
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"invalid step-limit entry at {path}:{line_number}")
            task, value = (part.strip() for part in line.split(":", 1))
            if value.lower() in ("null", "none", "~"):
                parsed = None
            else:
                try:
                    parsed = int(value)
                except ValueError as exc:
                    raise ValueError(
                        f"invalid integer step limit at {path}:{line_number}"
                    ) from exc
            result[task] = parsed
        return result

    with path.open("r", encoding="utf-8") as stream:
        result = yaml.safe_load(stream) or {}
    if not isinstance(result, dict):
        raise ValueError(f"step-limit config must be a mapping: {path}")
    for task, value in result.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"step limit for {task!r} must be an integer or null")
        if value is not None and value <= 0:
            raise ValueError(f"step limit for {task!r} must be positive")
    return result


def resolve_budgets(tasks, protocol, step_limits, debug_override=None):
    """Resolve budgets with explicit protocol semantics and optional debug cap."""
    if protocol not in ("uniform25", "repo_step_limits"):
        raise ValueError(f"unknown budget protocol: {protocol}")
    budgets = {}
    for task in tasks:
        if protocol == "uniform25":
            budget = 25
        else:
            value = step_limits.get(task)
            if value is None:
                raise ValueError(
                    f"repo_step_limits has no non-null entry for task {task!r}"
                )
            budget = int(value)
        if debug_override is not None:
            budget = min(budget, int(debug_override))
        budgets[task] = budget
    return budgets


def select_episode_ids(start_episode, count, episode_ids=None):
    if episode_ids is None:
        values = list(range(int(start_episode), int(start_episode) + int(count)))
    else:
        values = [int(value) for value in episode_ids]
        if len(values) != int(count):
            raise ValueError("episode list length must equal --eval-episodes")
    if not values or any(value < 0 for value in values):
        raise ValueError("episode IDs must be a non-empty list of non-negative integers")
    if len(set(values)) != len(values):
        raise ValueError("episode IDs must not contain duplicates")
    return values


def load_episode_ids(path):
    """Load a reusable JSON episode list (list or {"episode_ids": [...]})"""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("episode_ids")
    if not isinstance(value, list):
        raise ValueError("episode list JSON must be a list or contain episode_ids")
    return [int(item) for item in value]


def validate_run_protocol(run_mode, tasks, eval_episodes, repeats):
    if run_mode not in ("debug", "formal"):
        raise ValueError("run mode must be debug or formal")
    if not tasks:
        raise ValueError("at least one task is required")
    if len(set(tasks)) != len(tasks):
        raise ValueError("task list must not contain duplicates")
    if int(eval_episodes) <= 0 or int(repeats) <= 0:
        raise ValueError("episode and repeat counts must be positive")
    if run_mode == "formal":
        if tuple(tasks) != EVAL_TASKS:
            raise ValueError("formal runs must use the complete ordered EVAL_TASKS set")
        if int(eval_episodes) != 25 or int(repeats) != 5:
            raise ValueError("formal runs require 25 episodes and 5 repeats")


def aggregate_results(episode_results, planned_by_task, repeat_ids, selected_tasks):
    """Summarize explicit denominators; never label a subset as the full suite."""
    by_task_repeat = {}
    for result in episode_results:
        key = (result["task"], result["repeat_id"])
        by_task_repeat.setdefault(key, []).append(result)

    task_summaries = []
    for task in selected_tasks:
        for repeat_id in repeat_ids:
            rows = by_task_repeat.get((task, repeat_id), [])
            planned = int(planned_by_task[task])
            successes = sum(row.get("success") is True for row in rows)
            policy_failures = sum(row.get("error_class") == "policy_failure" for row in rows)
            infrastructure_errors = sum(
                row.get("error_class") == "infrastructure_error" for row in rows
            )
            evaluable = sum(row.get("evaluable") is True for row in rows)
            complete = len(rows) == planned and infrastructure_errors == 0
            task_summaries.append({
                "task": task,
                "task_group": (
                    "main" if task in EVAL_TASKS else
                    "sanity_check" if task in SANITY_CHECK_TASKS else "custom"
                ),
                "repeat_id": repeat_id,
                "planned_episodes": planned,
                "attempted_episodes": len(rows),
                "evaluable_episodes": evaluable,
                "successes": successes,
                "policy_failures": policy_failures,
                "infrastructure_errors": infrastructure_errors,
                "success_rate_0_1": (
                    successes / evaluable if evaluable else None
                ),
                "completion_status": "complete" if complete else "incomplete",
            })

    macro_by_repeat = []
    main_selected = tuple(task for task in selected_tasks if task in EVAL_TASKS)
    full_suite_selected = main_selected == EVAL_TASKS
    if full_suite_selected:
        for repeat_id in repeat_ids:
            rows = {
                row["task"]: row for row in task_summaries
                if row["repeat_id"] == repeat_id
            }
            if all(
                task in rows
                and rows[task]["completion_status"] == "complete"
                and rows[task]["evaluable_episodes"] == rows[task]["planned_episodes"]
                for task in EVAL_TASKS
            ):
                macro_by_repeat.append({
                    "repeat_id": repeat_id,
                    "success_rate_0_1": statistics.mean(
                        rows[task]["success_rate_0_1"] for task in EVAL_TASKS
                    ),
                })
    macro_values = [row["success_rate_0_1"] for row in macro_by_repeat]
    return {
        "task_results": task_summaries,
        "main_suite_complete": bool(full_suite_selected and len(macro_by_repeat) == len(repeat_ids)),
        "main_suite_macro_by_repeat": macro_by_repeat,
        "main_suite_repeat_mean_0_1": statistics.mean(macro_values) if macro_values else None,
        "main_suite_repeat_sample_stddev_0_1": (
            statistics.stdev(macro_values) if len(macro_values) > 1 else None
        ),
        "success_rate_units": "fraction in [0, 1]",
        "sanity_check_results": [
            row for row in task_summaries if row["task_group"] == "sanity_check"
        ],
    }


def finite_number_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
