"""Small, API-neutral schemas passed between the evaluator and a policy."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

@dataclass(frozen=True)
class AstraObservation:
    """Unprivileged observations exposed to an Astra-compatible policy.

    Images are RGB, channel-last NumPy arrays in the RLBench image value range.
    ``eef_pose`` is the current absolute world-frame pose from RLBench's
    ``Observation.gripper_pose``: XYZ followed by quaternion XYZW.
    """

    instruction: str
    images: Dict[str, Any]
    eef_pose: Sequence[float]
    gripper_open: bool


@dataclass(frozen=True)
class AstraAction:
    """Absolute world-frame target; ``ignore_collisions`` is mode-dependent."""

    position: Sequence[float]
    quaternion: Sequence[float]
    gripper: int
    ignore_collisions: Optional[int] = None
