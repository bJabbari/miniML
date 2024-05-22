from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import numpy as np


class Optimizer(ABC):
    """Base class for all optimizers."""
    @abstractmethod
    def update(self, grads_and_params: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
               in_place: bool = True) -> Optional[List[Optional[np.ndarray]]]:
        raise NotImplementedError('Must be implemented in subclass')
