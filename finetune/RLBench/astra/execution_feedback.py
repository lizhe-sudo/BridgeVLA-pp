"""Allowlisted measured action feedback for the next episode control turn."""

import math


def _vector(value, length):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _difference(left, right):
    if left is None or right is None:
        return None
    return [float(a - b) for a, b in zip(left, right)]


def build_execution_feedback(execution, action_id, observation_id_before,
                             observation_id_after):
    """Select only observed robot state and execution diagnostics.

    Reward, success-detector internals, object truth, and raw environment data
    are intentionally excluded from the policy-facing feedback contract.
    """
    raw_action = execution.get("raw_policy_action") or {}
    validated_action = execution.get("validated_action") or {}
    pose_before = _vector(execution.get("eef_pose_before"), 7)
    pose_after = _vector(execution.get("actual_eef_pose"), 7)
    requested_position = _vector(raw_action.get("position"), 3)
    validated_position = _vector(validated_action.get("position"), 3)
    effective_target = _vector(execution.get("effective_planner_target"), 3)
    before_position = pose_before[:3] if pose_before else None
    after_position = pose_after[:3] if pose_after else None
    return {
        "step_id": execution.get("step_id"),
        "action_id": str(action_id),
        "observation_id_before": observation_id_before,
        "observation_id_after": observation_id_after,
        "eef_pose_before": pose_before,
        "requested_target_pose": {
            "position": requested_position,
            "quaternion_xyzw": _vector(raw_action.get("quaternion"), 4),
            "gripper_command": raw_action.get("gripper"),
        },
        "validated_action_submitted": {
            "position": validated_position,
            "quaternion_xyzw": _vector(
                validated_action.get("quaternion_xyzw"), 4
            ),
            "gripper_command": validated_action.get("gripper"),
            "ignore_collisions": execution.get("ignore_collisions"),
            "action_vector": execution.get("final_9d_action"),
        },
        "effective_planner_target": {
            "position": effective_target,
            "source": execution.get("effective_target_source"),
        },
        "eef_pose_after": pose_after,
        "gripper_open_before": execution.get("gripper_before"),
        "gripper_open_after": execution.get("gripper_after"),
        "env_step_returned": execution.get("environment_step_returned"),
        "planner_returned": execution.get("planner_returned"),
        "new_observation_available": observation_id_after is not None,
        "requested_displacement_m": _difference(
            requested_position, before_position
        ),
        "actual_displacement_m": _difference(after_position, before_position),
        "target_error_m": _difference(after_position, effective_target),
        "position_error_to_effective_target_m": execution.get(
            "position_error_to_effective_target_m"
        ),
        "orientation_error_deg": execution.get("orientation_error_deg"),
        "position_reached": execution.get("position_reached"),
        "pose_reached": execution.get("pose_reached"),
        "execution_error_category": {
            "class": execution.get("error_class"),
            "reason": execution.get("error_reason"),
        },
    }
