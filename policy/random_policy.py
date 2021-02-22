from abc import ABC

import gym
import numpy as np

from policy.base_policy import BasePolicy


class RandomPolicy(BasePolicy, ABC):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.action_high = env.action_space.high
        self.action_low = env.action_space.low

    def infer(self, obs: np.ndarray):
        return np.random.uniform(low=self.action_low, high=self.action_high)
