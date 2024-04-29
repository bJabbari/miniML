from abc import ABCMeta, abstractmethod
import numpy as np


class Initializer(metaclass=ABCMeta):
    def __init__(self, seed: int = None):
        self.seed = seed

    @abstractmethod
    def __call__(self, shape: tuple, dtype: np.dtype = np.float32):
        if self.seed is not None:
            np.random.seed(self.seed)
