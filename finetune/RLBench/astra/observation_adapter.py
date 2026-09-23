"""Convert a raw RLBench observation into the policy-facing schema."""

import numpy as np

from .schemas import AstraObservation


class AstraObservationAdapter:
    CAMERA_NAMES = ("front", "left_shoulder", "right_shoulder", "wrist")

    def adapt(self, observation, instruction: str) -> AstraObservation:
        """Adapt one raw ``rlbench.backend.observation.Observation``.

        This intentionally reads pose and RGB directly from the RLBench
        observation object, rather than from the evaluator's reduced
        ``low_dim_state`` dictionary. No object poses, masks, or task state are
        included.
        """
        pose = getattr(observation, "gripper_pose", None)
        if pose is None:
            raise ValueError("RLBench Observation.gripper_pose is unavailable")
        pose = np.asarray(pose, dtype=np.float64).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(
                "RLBench Observation.gripper_pose must contain 7 values "
                f"(XYZ + quaternion XYZW), got shape {pose.shape}"
            )
        if not np.isfinite(pose).all():
            raise ValueError("RLBench Observation.gripper_pose contains NaN or Inf")

        images = {}
        for camera in self.CAMERA_NAMES:
            rgb = getattr(observation, f"{camera}_rgb", None)
            if rgb is None:
                raise ValueError(f"RLBench Observation.{camera}_rgb is unavailable")
            rgb = np.asarray(rgb)
            if rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(
                    f"RLBench Observation.{camera}_rgb must be HxWx3 RGB, "
                    f"got shape {rgb.shape}"
                )
            images[camera] = rgb.copy()

        gripper_open = getattr(observation, "gripper_open", None)
        if gripper_open is None:
            raise ValueError("RLBench Observation.gripper_open is unavailable")

        return AstraObservation(
            instruction=str(instruction),
            images=images,
            eef_pose=pose.tolist(),
            gripper_open=bool(gripper_open),
        )

    __call__ = adapt
