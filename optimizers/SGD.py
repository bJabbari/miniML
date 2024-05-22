from typing import List, Tuple, Optional

import numpy as np

from miniML.optimizers.Optimizer import Optimizer


class SGD(Optimizer):
    """
    SGD optimizer class implementing the Stochastic Gradient Descent optimization algorithm.
    Attributes:
        learning_rate (float): The step size for parameter updates.
        momentum (float): The momentum factor for parameter updates.
    """
    def __init__(self, learning_rate=0.01, momentum=0.0):
        """
        Initializes the SGD optimizer with the specified hyperparameters
        Args:
            learning_rate (float): The step size for parameter updates.
            momentum (float): The momentum factor for parameter updates.
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, grads_and_params: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]], in_place: bool = True)\
            -> Optional[List[Optional[np.ndarray]]]:
        """
        Updates the parameters using the SGD optimization algorithm with momentum.

        Args:
            grads_and_params (List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]): Each tuple contains a gradient
            and its corresponding parameter.
            in_place (bool): If True, update the parameters in place. Otherwise, return a new list of updated
            parameters.

        Returns:
            Union[None, List[np.ndarray]]: None if in_place is True, otherwise a list of updated parameters.
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) if param is not None else None for _, param in grads_and_params]

        if not in_place:
            new_params = []

        for i, (grad, param) in enumerate(grads_and_params):
            if param is None:  # some layers may not have any trainable parameters
                if not in_place:
                    new_params.append(None)
                continue
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            if in_place:
                param += self.velocity[i]
            else:
                new_params.append(param + self.velocity[i])

        if not in_place:
            return new_params
