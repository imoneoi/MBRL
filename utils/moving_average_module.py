from abc import ABC

import torch
from torch import nn


class MovingAverageModule(nn.Module, ABC):
    def __init__(self,
                 dims: int,
                 eps: torch.float = torch.finfo(torch.float).eps,

                 device: torch.device = torch.device("cuda:0")):
        super().__init__()
        self.device = device

        self.dims = dims

        self.n = self.static_parameter(1)
        self.mean = self.static_parameter(dims)
        self.mean_sq = self.static_parameter(dims)

        self.eps = eps

    '''
        Static parameters
    '''
    def static_parameter(self, shape):
        return torch.nn.Parameter(
            torch.zeros(shape, device=self.device),
            requires_grad=False)

    def update(self, batch: torch.Tensor):
        # weight
        n_cur = batch.shape[0]
        w_cur = n_cur / (n_cur + self.n)

        # get mean and std
        mean_cur = torch.mean(batch, dim=0).to(self.device)
        mean_sq_cur = torch.mean(batch ** 2, dim=0).to(self.device)

        # update
        self.mean.copy_((1 - w_cur) * self.mean + w_cur * mean_cur)
        self.mean_sq.copy_((1 - w_cur) * self.mean_sq + w_cur * mean_sq_cur)
        self.n.copy_(self.n + n_cur)

    def get_mean_std(self):
        mean = self.mean
        std = torch.sqrt(torch.maximum(
            torch.zeros(1, device=self.device),
            self.mean_sq - self.mean ** 2))  # clip to zero to avoid NaN

        return mean, std

    def normalize(self, x: torch.Tensor):
        mean, std = self.get_mean_std()
        # move device
        mean = mean.to(x.device)
        std = std.to(x.device)

        return (x - mean) / (std + self.eps)

    def denormalize(self, x: torch.Tensor):
        mean, std = self.get_mean_std()
        # move device
        mean = mean.to(x.device)
        std = std.to(x.device)

        return x * (std + self.eps) + mean
