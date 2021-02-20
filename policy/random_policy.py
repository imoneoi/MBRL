from abc import ABC

import gym
import numpy as np

from policy.base_policy import BasePolicy


class RandomPolicy(BasePolicy, ABC):
    def __init__(self, env: gym.Env):
        super().__init__()

        self.action_space = env.action_space

    def infer(self, obs: np.ndarray):
        return self.action_space.sample()
