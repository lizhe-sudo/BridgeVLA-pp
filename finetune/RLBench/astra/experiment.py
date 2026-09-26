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


def aggregate_results(episode_results, planned_by_task, repeat_ids, selected_tasks,
                      expected_episode_ids=None):
    """Aggregate only unique, valid, evaluable units with complete evidence."""
    tasks = tuple(selected_tasks)
    repeats = tuple(repeat_ids)
    expected_episode_ids = expected_episode_ids or {
        task: list(range(int(planned_by_task[task]))) for task in tasks
    }
    expected_by_pair = {
        (task, repeat_id): tuple(expected_episode_ids[task])
        for task in tasks for repeat_id in repeats
    }
    expected_units = {
        (task, repeat_id, episode_id)
        for (task, repeat_id), episode_ids in expected_by_pair.items()
        for episode_id in episode_ids
    }
    counts = {}
    keyed_rows = {}
    invalid_identity_rows = 0
    unexpected_units = []
    for index, row in enumerate(episode_results):
        if not isinstance(row, dict):
            invalid_identity_rows += 1
            continue
        try:
            unit = (row["task"], row["repeat_id"], row["episode_id"])
        except (KeyError, TypeError):
            invalid_identity_rows += 1
            continue
        if (not isinstance(unit[0], str)
                or isinstance(unit[1], bool) or not isinstance(unit[1], int)
                or isinstance(unit[2], bool) or not isinstance(unit[2], int)):
            invalid_identity_rows += 1
            continue
        counts[unit] = counts.get(unit, 0) + 1
        keyed_rows.setdefault(unit, []).append((index, row))
        if unit not in expected_units:
            unexpected_units.append({
                "row_index": index, "task": row.get("task"),
                "repeat_id": row.get("repeat_id"),
                "episode_id": row.get("episode_id"),
            })

    duplicate_units = [unit for unit, count in counts.items() if count > 1]
    duplicate_set = set(duplicate_units)
    task_summaries = []
    for task in tasks:
        for repeat_id in repeats:
            expected_ids = expected_by_pair[(task, repeat_id)]
            valid_rows = []
            invalid_rows = []
            relevant_rows = [
                row for unit, entries in keyed_rows.items()
                if unit[0:2] == (task, repeat_id)
                for _, row in entries
            ]
            attempted_units = set()
            duplicate_count = 0
            unexpected_count = 0
            conflict_count = 0
            for unit, entries in keyed_rows.items():
                if unit[0:2] != (task, repeat_id):
                    continue
                if unit not in expected_units:
                    unexpected_count += len(entries)
                    continue
                if unit in duplicate_set:
                    duplicate_count += len(entries)
                    continue
                row = entries[0][1]
                if row.get("attempted") is True:
                    attempted_units.add(unit)
                evaluable = row.get("evaluable")
                success = row.get("success")
                error_class = row.get("error_class")
                core_complete = row.get("core_records_complete") is True
                conflict = (
                    not isinstance(evaluable, bool)
                    or not isinstance(success, (bool, type(None)))
                    or (success is True and evaluable is not True)
                    or (evaluable is True and success is None)
                    or (error_class in ("infrastructure_error", "unknown")
                        and evaluable is True)
                    or (evaluable is True and not core_complete)
                )
                if conflict:
                    conflict_count += 1
                    invalid_rows.append(row)
                    continue
                if (evaluable is True and success in (True, False)
                        and core_complete
                        and error_class not in ("infrastructure_error", "unknown")):
                    valid_rows.append(row)
                else:
                    invalid_rows.append(row)

            evaluable_count = len(valid_rows)
            successes = sum(row["success"] is True for row in valid_rows)
            infrastructure_errors = sum(
                row.get("error_class") == "infrastructure_error"
                for row in relevant_rows
            )
            policy_failures = sum(
                row.get("error_class") == "policy_failure" for row in valid_rows
            )
            unknown_errors = sum(
                row.get("error_class") == "unknown" for row in relevant_rows
            )
            missing_ids = [
                episode_id for episode_id in expected_ids
                if counts.get((task, repeat_id, episode_id), 0) == 0
            ]
            unit_rows = [
                row for episode_id in expected_ids
                for _, row in keyed_rows.get((task, repeat_id, episode_id), [])
            ]
            complete = (
                not missing_ids
                and all(counts.get((task, repeat_id, episode_id), 0) == 1
                        for episode_id in expected_ids)
                and evaluable_count == len(expected_ids)
                and conflict_count == 0
                and duplicate_count == 0
                and unexpected_count == 0
                and invalid_identity_rows == 0
                and infrastructure_errors == 0
                and unknown_errors == 0
                and all(row.get("attempted") is True for row in unit_rows)
            )
            task_summaries.append({
                "task": task,
                "task_group": (
                    "main" if task in EVAL_TASKS else
                    "sanity_check" if task in SANITY_CHECK_TASKS else "custom"
                ),
                "repeat_id": repeat_id,
                "planned_episodes": len(expected_ids),
                "attempted_episodes": len(attempted_units),
                "evaluable_episodes": evaluable_count,
                "successes": successes,
                "policy_failures": policy_failures,
                "infrastructure_errors": infrastructure_errors,
                "unknown_errors": unknown_errors,
                "missing_episode_ids": missing_ids,
                "duplicate_experiment_units": duplicate_count,
                "unexpected_experiment_units": unexpected_count,
                "invalid_identity_rows": invalid_identity_rows,
                "data_conflicts": conflict_count,
                "success_rate_0_1": successes / evaluable_count if evaluable_count else None,
                "completion_status": "complete" if complete else "incomplete",
            })

    macro_by_repeat = []
    main_selected = tuple(task for task in tasks if task in EVAL_TASKS)
    full_suite_selected = main_selected == EVAL_TASKS
    if full_suite_selected:
        for repeat_id in repeats:
            rows = {
                row["task"]: row for row in task_summaries
                if row["repeat_id"] == repeat_id
            }
            if all(
                task in rows and rows[task]["completion_status"] == "complete"
                for task in EVAL_TASKS
            ):
                macro_by_repeat.append({
                    "repeat_id": repeat_id,
                    "success_rate_0_1": statistics.mean(
                        rows[task]["success_rate_0_1"] for task in EVAL_TASKS
                    ),
                })
    all_repeats_complete = (
        full_suite_selected
        and len(macro_by_repeat) == len(repeats)
        and not unexpected_units
        and not duplicate_units
        and invalid_identity_rows == 0
    )
    complete_values = [row["success_rate_0_1"] for row in macro_by_repeat]
    return {
        "task_results": task_summaries,
        "main_suite_complete": bool(all_repeats_complete),
        "main_suite_macro_by_repeat": macro_by_repeat,
        "main_suite_repeat_mean_0_1": (
            statistics.mean(complete_values) if all_repeats_complete else None
        ),
        "main_suite_repeat_sample_stddev_0_1": (
            statistics.stdev(complete_values)
            if all_repeats_complete and len(complete_values) > 1 else None
        ),
        "main_suite_partial_repeat_mean_0_1": (
            statistics.mean(complete_values)
            if full_suite_selected and complete_values and not all_repeats_complete else None
        ),
        "main_suite_partial_repeat_ids": (
            [row["repeat_id"] for row in macro_by_repeat]
            if full_suite_selected and not all_repeats_complete else []
        ),
        "success_rate_units": "fraction in [0, 1]",
        "sanity_check_results": [
            row for row in task_summaries if row["task_group"] == "sanity_check"
        ],
        "unexpected_experiment_units": unexpected_units,
        "duplicate_experiment_units": [list(unit) for unit in duplicate_units],
        "invalid_identity_rows": invalid_identity_rows,
    }


def finite_number_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None
