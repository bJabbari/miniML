from typing import List, Tuple, Optional

import numpy as np

from miniML.optimizers.Optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer class implementing the Adam optimization algorithm.

    Attributes:
        learning_rate (float): The step size for parameter updates.
        beta1 (float): The exponential decay rate for the first moment estimates.
        beta2 (float): The exponential decay rate for the second moment estimates.
        epsilon (float): A small constant to prevent division by zero.
        t (int): Time step counter.
        m (List[np.ndarray] or None): List of first moment vectors.
        v (List[np.ndarray] or None): List of second moment vectors.
    """

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7):
        """
        Initializes the Adam optimizer with the specified hyperparameters.
        Args:
            learning_rate (float): The step size for parameter updates.
            beta_1 (float): The exponential decay rate for the first moment estimates.
            beta_2 (float): The exponential decay rate for the second moment estimates.
            epsilon (float): A small constant to prevent division by zero.
        """
        self.learning_rate = learning_rate
        self.beta1 = beta_1
        self.beta2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.m = None
        self.v = None

    def update(self, grads_and_params: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]], in_place=True) \
            -> Optional[List[Optional[np.ndarray]]]:
        """
        Updates the parameters using the Adam optimization algorithm
        Args:
            grads_and_params (list of tuples): Each tuple contains a gradient and its corresponding parameter.
            in_place (bool): If True, update the parameters in place. Otherwise, return a new list of updated parameters
        Returns:
            Union[None, List[np.ndarray]]: None if in_place is True, otherwise a list of updated parameters.
        """
        if self.m is None:
            self.m = [np.zeros_like(param) if param is not None else None for _, param in grads_and_params]
            self.v = [np.zeros_like(param) if param is not None else None for _, param in grads_and_params]

        self.t += 1
        lr_t = self.learning_rate  # * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        if not in_place:
            new_params = []

        for i, (grad, param) in enumerate(grads_and_params):
            if param is None:  # some layers may not have any trainable parameters
                if not in_place:
                    new_params.append(None)
                continue
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            if in_place:
                param -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                new_params.append(param - lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon))

        if not in_place:
            return new_params
