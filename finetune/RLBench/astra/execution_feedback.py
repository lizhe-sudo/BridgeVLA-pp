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


def _required_vector(mapping, key, length, qualified_name):
    value = mapping.get(key)
    result = _vector(value, length)
    if result is None:
        raise ValueError(
            f"execution feedback requires {qualified_name} as {length} finite numbers"
        )
    return result


def _required_gripper(mapping, key, qualified_name):
    value = mapping.get(key)
    if isinstance(value, bool) or value not in (0, 1, 0.0, 1.0):
        raise ValueError(f"execution feedback requires {qualified_name} to be 0 or 1")
    return int(value)


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
    raw_action = execution.get("raw_policy_action")
    validated_action = execution.get("validated_action")
    if not isinstance(raw_action, dict):
        raise ValueError("execution feedback requires raw_policy_action diagnostics")
    if not isinstance(validated_action, dict):
        raise ValueError("execution feedback requires validated_action diagnostics")
    pose_before = _vector(execution.get("eef_pose_before"), 7)
    pose_after = _vector(execution.get("actual_eef_pose"), 7)
    requested_position = _required_vector(
        raw_action, "position", 3, "raw_policy_action.position"
    )
    requested_quaternion = _required_vector(
        raw_action, "quaternion_xyzw", 4,
        "raw_policy_action.quaternion_xyzw",
    )
    # `quaternion_xyzw` is the action adapter's canonical diagnostic key. The
    # older `quaternion` spelling is never used as a fallback; if present it
    # must agree exactly so a mixed-version record cannot silently change the
    # requested orientation.
    if "quaternion" in raw_action:
        legacy_quaternion = _vector(raw_action.get("quaternion"), 4)
        if legacy_quaternion != requested_quaternion:
            raise ValueError(
                "raw_policy_action.quaternion conflicts with canonical "
                "raw_policy_action.quaternion_xyzw"
            )
    requested_gripper = _required_gripper(
        raw_action, "gripper", "raw_policy_action.gripper"
    )
    validated_position = _required_vector(
        validated_action, "position", 3, "validated_action.position"
    )
    submitted_quaternion = _required_vector(
        validated_action, "quaternion_xyzw", 4,
        "validated_action.quaternion_xyzw",
    )
    submitted_gripper = _required_gripper(
        validated_action, "gripper", "validated_action.gripper"
    )
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
            "quaternion_xyzw": requested_quaternion,
            "gripper_command": requested_gripper,
        },
        "validated_action_submitted": {
            "position": validated_position,
            "quaternion_xyzw": submitted_quaternion,
            "gripper_command": submitted_gripper,
            "ignore_collisions": execution.get("ignore_collisions"),
            "action_vector": execution.get("final_9d_action"),
        },
        "effective_planner_target": {
            "position": effective_target,
            "source": execution.get("effective_target_source"),
        },
        "eef_pose_after": pose_after,
        "actual_orientation_before_xyzw": pose_before[3:7] if pose_before else None,
        "actual_orientation_after_xyzw": pose_after[3:7] if pose_after else None,
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
