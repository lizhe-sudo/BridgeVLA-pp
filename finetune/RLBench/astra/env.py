"""RLBench wrapper that exposes Astra's raw observation without recapturing it."""

from utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2


class AstraRLBenchEnv(CustomMultiTaskRLBenchEnv2):
    """Keep the latest raw RLBench observation separate from YARR extraction."""

    def __init__(self, *args, **kwargs):
        super(AstraRLBenchEnv, self).__init__(*args, **kwargs)
        self._last_raw_observation = None

    @property
    def last_raw_observation(self):
        """The latest raw Observation returned during reset or step."""
        return self._last_raw_observation

    def extract_obs(self, obs, t=None, prev_action=None):
        # Save the exact RLBench object before the parent temporarily removes
        # gripper_pose and constructs the processed observation dictionary.
        # The parent restores gripper_pose, and leaves RGB/gripper_open intact.
        self._last_raw_observation = obs
        return super(AstraRLBenchEnv, self).extract_obs(
            obs, t=t, prev_action=prev_action
        )

    def reset(self):
        self._last_raw_observation = None
        return super(AstraRLBenchEnv, self).reset()

    def reset_to_demo(self, i, variation_number=-1):
        self._last_raw_observation = None
        return super(AstraRLBenchEnv, self).reset_to_demo(
            i, variation_number=variation_number
        )

    def step(self, act_result):
        # A planning/IK/invalid-action failure produces no new Observation.
        # Clear the previous one before delegating so failures cannot expose it.
        self._last_raw_observation = None
        return super(AstraRLBenchEnv, self).step(act_result)
