from abc import ABC, abstractmethod
from typing import Union, Dict

import numpy as np
import torch


class BaseModel(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def infer(self,
              obs: Union[torch.Tensor, np.ndarray],
              act: Union[torch.Tensor, np.ndarray]):
        pass

    @abstractmethod
    def fit(self, dataset: Dict[str, Union[torch.Tensor, np.ndarray]], **kwargs):
        pass

    @abstractmethod
    def test(self, dataset: Dict[str, Union[torch.Tensor, np.ndarray]]):
        pass
