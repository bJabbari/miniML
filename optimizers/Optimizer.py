from abc import ABC, abstractmethod
from typing import List

import numpy as np


class Optimizer(ABC):
    """Base class for all optimizers."""
    @abstractmethod
    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray], in_place=True):
        raise NotImplementedError('Must be implemented in subclass')
