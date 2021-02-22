from abc import ABC
from typing import Callable

import gym
import numpy as np
import torch

from model.base_model import BaseModel
from policy.base_policy import BasePolicy


class MPCRandomSamplePolicy(BasePolicy, ABC):
    def __init__(self,
                 env: gym.Env,
                 model: BaseModel,
                 objective: Callable,
                 horizon: int = 20,
                 num_sample: int = 16384):
        super().__init__()
        # env
        self.action_dims = env.action_space.shape[0]
        self.action_high = torch.tensor(env.action_space.high, device=model.device)
        self.action_low = torch.tensor(env.action_space.low, device=model.device)

        # model
        self.model = model

        # objective
        self.objective = objective

        # parameters
        self.horizon = horizon
        self.num_sample = num_sample

    def infer(self, obs: np.ndarray):
        # generate action sequences
        x = torch.rand((self.num_sample, self.horizon, self.action_dims), device=self.model.device)
        action_seqs = self.action_low + x * (self.action_high - self.action_low)

        # evaluate sequences
        s_t = torch.tensor(obs, dtype=torch.float, device=self.model.device)
        s_t = s_t.repeat(self.num_sample, 1)

        rew = torch.zeros(self.num_sample, device=self.model.device)
        for t in range(self.horizon):
            a_t = action_seqs[:, t]
            with torch.no_grad():
                s_t_next = self.model.infer(s_t, a_t)

            rew = rew + self.objective(s_t, a_t, s_t_next)

            s_t = s_t_next

        # get max reward
        a_idx = torch.argmin(rew)
        print("Min objective: {}".format(rew[a_idx].item()))

        return action_seqs[a_idx, 0].cpu().numpy()
