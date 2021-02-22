from abc import ABC

import torch
from torch import nn


# Welford's online algorithm (Parallel algorithm)
# ref: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
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
        self.M2 = self.static_parameter(dims)

        self.eps = eps

    '''
        Static parameters
    '''
    def static_parameter(self, shape):
        return torch.nn.Parameter(
            torch.zeros(shape, device=self.device),
            requires_grad=False)

    def update(self, batch: torch.Tensor):
        # update mean
        b_sz = batch.shape[0]
        b_mean = torch.mean(batch, dim=0).to(self.device)
        b_M2 = torch.sum((batch - b_mean) ** 2, dim=0).to(self.device)

        # update std
        delta = b_mean - self.mean
        self.M2.copy_(self.M2 + b_M2 + (delta ** 2) * self.n * b_sz / (self.n + b_sz))

        # update mean
        self.mean.copy_((self.n * self.mean + b_sz * b_mean) / (self.n + b_sz))

        # update N
        self.n.copy_(self.n + b_sz)

    def get_mean_std(self):
        mean = self.mean
        std = torch.sqrt(self.M2 / (self.n - 1))

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
