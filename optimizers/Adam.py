from typing import List

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
    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
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

    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray], in_place=True):
        """
        Updates the parameters using the Adam optimization algorithm
        Args:
            parameters (List[np.ndarray]): The list of parameters to be updated.
            gradients (List[np.ndarray]): The list of gradients for each parameter.
            in_place (bool): If True, update the parameters in place. Otherwise, return a new list of updated parameters
        Returns:
            Union[None, List[np.ndarray]]: None if in_place is True, otherwise a list of updated parameters.
        """
        if self.m is None:
            self.m = [np.zeros_like(param) for param in parameters]
            self.v = [np.zeros_like(param) for param in parameters]

        self.t += 1
        lr_t = self.learning_rate #* np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        if not in_place:
            new_params = []

        for i, (param, grad) in enumerate(zip(parameters, gradients)):
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
