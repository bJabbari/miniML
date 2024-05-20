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

    def update(self, parameters, gradients, in_place=True):
        """
        Updates the parameters using the SGD optimization algorithm with momentum.

        Args:
            parameters (List[np.ndarray]): The list of parameters to be updated.
            gradients (List[np.ndarray]): The list of gradients for each parameter.
            in_place (bool): If True, update the parameters in place. Otherwise, return a new list of updated parameters.

        Returns:
            Union[None, List[np.ndarray]]: None if in_place is True, otherwise a list of updated parameters.
        """
        if self.velocity is None:
            self.velocity = [np.zeros_like(param) for param in parameters]

        if not in_place:
            new_params = []

        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            self.velocity[i] = self.momentum * self.velocity[i] - self.learning_rate * grad
            if in_place:
                param += self.velocity[i]
            else:
                new_params.append(param + self.velocity[i])

        if not in_place:
            return new_params
