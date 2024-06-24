from typing import List, Tuple, Optional

import numpy as np
from miniML.optimizers.Optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, learning_rate=1e-3, rho=0.9, momentum=0.0, epsilon=1e-7):
        """
        Initializes the RMSprop ('Root mean squared propagation') optimizer with the specified hyperparameters.
        Args:
            learning_rate (float): The step size for parameter updates.
            rho (float): The exponential decay rate for the moving average of squared gradients. Defaults to 0.9.
            momentum (float): The `momentum` parameter introduces a momentum term into the RMSprop optimizer. When `momentum` is greater than zero, the optimizer accumulates an exponentially decaying moving average of past gradients and uses this accumulated value to update the parameters. This helps to accelerate gradients vectors in the right directions, leading to faster converging.
            epsilon (float): A small constant to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.momentum = momentum
        self.epsilon = epsilon
        self.t = 0
        self.grad_squared = None
        self.grad_momentum = None

    def update(self, grads_and_params: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]], in_place=True) \
            -> Optional[List[Optional[np.ndarray]]]:
        """
        Updates the parameters using the RMSprop algorithm
        Args:
            grads_and_params (list of tuples): Each tuple contains a gradient and its corresponding parameter.
            in_place (bool): If True, update the parameters in place. Otherwise, return a new list of updated parameters
        Returns:
            Union[None, List[np.ndarray]]: None if in_place is True, otherwise a list of updated parameters.
        """
        if self.grad_squared is None:
            self.grad_squared = [np.zeros_like(param) if param is not None else None for _, param in grads_and_params]
        if self.momentum > 0 and self.grad_momentum is None:
            self.grad_momentum = [np.zeros_like(param) if param is not None else None for _, param in grads_and_params]

        # self.t += 1
        lr_t = self.learning_rate #this is a place hoder for decaying learning rate
        if not in_place:
            new_params = []

        for i, (grad, param) in enumerate(grads_and_params):
            if param is None:  # some layers may not have any trainable parameters
                if not in_place:
                    new_params.append(None)
                continue
            self.grad_squared[i] = self.rho * self.grad_squared[i] + (1 - self.rho) * (grad ** 2)
            v_hat = self.grad_squared[i]  # / (1 - self.rho ** self.t)
            update_value = lr_t * grad / (np.sqrt(v_hat) + self.epsilon)
            if self.momentum > 0:
                self.grad_momentum[i] = self.momentum * self.grad_momentum[i] + update_value
                update_value = self.grad_momentum[i]

            if in_place:
                param -= update_value
            else:
                new_params.append(param - update_value)

        if not in_place:
            return new_params
