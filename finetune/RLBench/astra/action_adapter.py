"""Validate policy actions and map them to RLBench's 9D action layout."""

import math
import numbers

from .schemas import AstraAction


class AstraActionAdapter:
    """Convert an absolute pose action into RLBench's 9D action layout."""

    QUATERNION_NORM_EPSILON = 1e-8

    def __init__(self, collision_mode="fixed0"):
        if collision_mode not in ("fixed0", "fixed1", "predict"):
            raise ValueError("collision_mode must be fixed0, fixed1, or predict")
        self.collision_mode = collision_mode
        self.last_diagnostics = None

    @staticmethod
    def _finite_vector(values, expected_size: int, name: str):
        try:
            raw_values = list(values)
            if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
                   for value in raw_values):
                raise TypeError
            result = [float(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a sequence of numbers") from exc
        if len(result) != expected_size:
            raise ValueError(f"{name} must contain {expected_size} values, got {len(result)}")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} contains NaN or Inf")
        return result

    def adapt(self, action: AstraAction):
        if not isinstance(action, AstraAction):
            raise TypeError(
                "policy.act() must return AstraAction, "
                f"got {type(action).__name__}"
            )

        position = self._finite_vector(action.position, 3, "position")
        quaternion = self._finite_vector(action.quaternion, 4, "quaternion")
        norm = math.hypot(*quaternion)
        if norm < self.QUATERNION_NORM_EPSILON:
            raise ValueError(
                "quaternion norm is too small to normalize "
                f"(norm={norm:g}, minimum={self.QUATERNION_NORM_EPSILON:g})"
            )
        quaternion = [component / norm for component in quaternion]

        try:
            if isinstance(action.gripper, bool) or not isinstance(action.gripper, numbers.Real):
                raise TypeError
            gripper = float(action.gripper)
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper must be 0 (close) or 1 (open)") from exc
        if not math.isfinite(gripper) or gripper not in (0.0, 1.0):
            raise ValueError(f"gripper must be 0 (close) or 1 (open), got {action.gripper!r}")

        predicted = getattr(action, "ignore_collisions", None)
        if self.collision_mode == "predict":
            if (isinstance(predicted, bool) or not isinstance(predicted, int)
                    or predicted not in (0, 1)):
                raise ValueError(
                    "predict collision mode requires ignore_collisions 0 or 1"
                )
            ignore_collisions = int(predicted)
        else:
            if predicted is not None:
                raise ValueError(
                    "fixed collision modes do not accept a policy collision field"
                )
            ignore_collisions = 0 if self.collision_mode == "fixed0" else 1

        raw_position = self._finite_vector(action.position, 3, "position")
        raw_quaternion = self._finite_vector(action.quaternion, 4, "quaternion")
        self.last_diagnostics = {
            "raw_policy_action": {
                "position": raw_position,
                "quaternion_xyzw": raw_quaternion,
                "gripper": int(gripper),
                "ignore_collisions": predicted,
            },
            "validated_action": {
                "position": list(position),
                "quaternion_xyzw": list(quaternion),
                "gripper": int(gripper),
                "ignore_collisions": ignore_collisions,
            },
            "quaternion_norm_before_normalization": norm,
            "quaternion_was_normalized": abs(norm - 1.0) > 1e-9,
            "collision_mode": self.collision_mode,
            "effective_planner_target": None,
            "effective_target_source": None,
        }
        return position + quaternion + [gripper, float(ignore_collisions)]

    def set_workspace_bounds(self, workspace_min, workspace_max):
        """Passively mirror EndEffectorPoseViaPlanning2's XYZ clipping."""
        if self.last_diagnostics is None:
            return
        import numpy as np

        before = self.last_diagnostics["validated_action"]["position"]
        lower = np.asarray(workspace_min, dtype=float) + 1e-7
        upper = np.asarray(workspace_max, dtype=float) - 1e-7
        effective = np.clip(np.asarray(before, dtype=float), lower, upper)
        self.last_diagnostics["effective_planner_target"] = effective.tolist()
        self.last_diagnostics["workspace_clip"] = {
            "applied": bool(not np.array_equal(effective, before)),
            "position_before_m": list(before),
            "position_after_m": effective.tolist(),
        }
        self.last_diagnostics["effective_target_source"] = (
            "Astra passive calculation matching "
            "EndEffectorPoseViaPlanning2.action XYZ np.clip bounds +/- 1e-7"
        )

    __call__ = adapt
