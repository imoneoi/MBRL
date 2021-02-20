from abc import ABC, abstractmethod

import numpy as np


class BasePolicy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def infer(self, obs: np.ndarray):
        pass
