"""Policies for validating the evaluator without calling an Astra service."""

from typing import Optional

from .policy import AstraPolicy
from .schemas import AstraAction, AstraObservation


class MockPolicy(AstraPolicy):
    """Hold the current EEF pose and gripper, or repeat a supplied action."""

    def __init__(self, manual_action: Optional[AstraAction] = None):
        self.manual_action = manual_action

    def act(self, observation: AstraObservation) -> AstraAction:
        if self.manual_action is not None:
            return self.manual_action
        return AstraAction(
            position=list(observation.eef_pose[:3]),
            quaternion=list(observation.eef_pose[3:7]),
            gripper=1 if observation.gripper_open else 0,
        )


class ManualPolicy(MockPolicy):
    """Repeat the action supplied with ``--manual-action`` every waypoint."""

    def __init__(self, action: AstraAction):
        super().__init__(manual_action=action)
