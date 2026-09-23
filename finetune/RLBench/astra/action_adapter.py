"""Validate policy actions and map them to RLBench's 9D action layout."""

import math

from .schemas import AstraAction


class AstraActionAdapter:
    """Convert [XYZ, quaternion XYZW, gripper] into the RLBench action."""

    QUATERNION_NORM_EPSILON = 1e-8

    @staticmethod
    def _finite_vector(values, expected_size: int, name: str):
        try:
            result = [float(value) for value in values]
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
            gripper = float(action.gripper)
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper must be 0 (close) or 1 (open)") from exc
        if not math.isfinite(gripper) or gripper not in (0.0, 1.0):
            raise ValueError(f"gripper must be 0 (close) or 1 (open), got {action.gripper!r}")

        # Collision checking stays enabled in this baseline. Workspace bounds,
        # IK, and path planning belong to RLBench's existing action mode.
        return position + quaternion + [gripper, 0.0]

    __call__ = adapt
