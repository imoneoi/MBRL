from abc import ABC
from typing import Union, Dict

import torch
import numpy as np
from sklearn.linear_model import Lasso, Ridge

from model.base_model import BaseModel
from utils.moving_average_module import MovingAverageModule


class LinearKernelModel(BaseModel, ABC):
    def __init__(self,
                 state_dims: int,
                 action_dims: int):
        super().__init__()
        self.device = torch.device("cpu")  # Must use CPU

        self.x_dims = self.kernel_size(state_dims + action_dims)
        self.y_dims = state_dims

        # Model
        self.model = None

        # Parameter
        self.K = None

        # Normalizers
        self.norm_x = MovingAverageModule(self.x_dims, device=self.device)
        self.norm_y = MovingAverageModule(self.y_dims, device=self.device)

    @staticmethod
    def trigonometric_kernel(x: np.ndarray):
        return np.concatenate([x, np.sin(x), np.cos(x)], axis=-1)

    @staticmethod
    def quadratic_kernel(x: np.ndarray):
        # concat bias
        x_with_bias = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=-1)

        # quadratic matrix
        x_quad = np.expand_dims(x_with_bias, -1) * np.expand_dims(x_with_bias, -2)

        # get upper triangular
        triu_indices = np.triu_indices(x_quad.shape[-1])
        y = x_quad[:, triu_indices[0], triu_indices[1]]

        return y

    @staticmethod
    def kernel_size(x: int):
        y = x

        # trigonometric kernel
        y = 3 * y

        # quadratic kernel
        # y = y + 1
        # y = y * (y - 1) // 2 + y
        return y

    def infer(self,
              obs: Union[torch.Tensor, np.ndarray],
              act: Union[torch.Tensor, np.ndarray]):
        pass

    def fit(self,
            dataset: Dict[str, Union[torch.Tensor, np.ndarray]],
            **kwargs):
        assert self.model is None, "Model can only be trained once"

        # Convert dataset
        dataset_x = np.concatenate([dataset["obs"], dataset["act"]], axis=-1)
        dataset_y = dataset["next_obs"] - dataset["obs"]

        # Kernels
        dataset_x = self.trigonometric_kernel(dataset_x)
        # dataset_x = self.quadratic_kernel(dataset_x)

        # Normalize data
        self.norm_x.update(dataset_x)
        self.norm_y.update(dataset_y)
        dataset_x = self.norm_x.normalize(dataset_x)
        dataset_y = self.norm_y.normalize(dataset_y)

        # Fit model
        self.model = Ridge()
        self.model.fit(dataset_x, dataset_y)

        print("ok")

    def test(self,
             dataset: Dict[str, Union[torch.Tensor, np.ndarray]]):
        pass
