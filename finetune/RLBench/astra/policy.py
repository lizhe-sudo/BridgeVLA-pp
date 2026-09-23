"""Base interface for policies consumed by the RLBench Astra evaluator."""

from abc import ABC, abstractmethod
from typing import Optional

from .schemas import AstraAction, AstraObservation


class AstraPolicy(ABC):
    """A policy is reset once per episode and called once per waypoint."""

    def reset(self, instruction: Optional[str] = None) -> None:
        """Clear episode-local state. Real API clients can also set context here."""

    @abstractmethod
    def act(self, observation: AstraObservation) -> AstraAction:
        """Return an absolute world-frame EEF target and gripper command."""
        raise NotImplementedError
