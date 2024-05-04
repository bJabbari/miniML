from abc import ABCMeta, abstractmethod

import numpy as np


class Activation(metaclass=ABCMeta):

    def __init__(self, check_validity=False):
        self.check_validity = check_validity

    @abstractmethod
    def derivative(self, input_values, return_as_matrix=False, activation_output=False):
        """
Compute the derivative of the activation function.

        Args:
            input_values (numeric or array-like): The input value(s) at which to compute the derivative.
            activation_output (bool, optional):  If True, indicates that input_values represent the output
                of the activation function. The derivative will be computed from this output.
                Defaults to False, meaning input_values are directly used as inputs to the activation function.
            return_as_matrix (bool, optional): If True, return the derivative as a 2-D matrix.
                Defaults to False.

        Returns:
            numeric or array-like: The derivative(s) of the activation function.

        Raises:
            NotImplementedError: If the derivative is not implemented by the subclass.

        """
        pass

    @abstractmethod
    def __call__(self, x):
        """
        Apply the activation function to the input
        :param x: numpy.ndarray
            Input to the activation function
        :return: numpy.ndarray
            Output of the activation function.
        """
        pass

    @abstractmethod
    def back_propagation(self, gradient, x, activation_output=True):
        """
        Compute the backpropagated gradient of the activation function
        :param gradient: numpy.ndarray
            Gradient from the previous layer.
            Its shape should be `(number of batches, number of outputs of the layer)`
        :param x: numpy.ndarray
            The point at which the derivative of the activation function is calculated.
            It can be either the output of the activation function or the input to it,
            depending on the value of `activation_output`.
            If possible, it is desired to calculate the derivative using the output
            values of the activation function for performance reasons. Otherwise,
            you must use the input values to the activation function and set
            `activation_output` flag as False.
        :param activation_output: bool, optional
            Flag indicating whether the input 'x' is the output of
            the activation function or the input to the activation
            function. Default is True
        :return: numpy.ndarray
            Backpropagated gradient of the activation function.
        """
        if not isinstance(gradient, np.ndarray):
            raise TypeError('gradient must be a numpy array')
        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy array')


        pass
