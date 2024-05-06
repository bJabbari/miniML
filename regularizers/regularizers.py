from abc import ABC, abstractmethod
import numpy as np

from miniML.helper.validation import validate_non_negative_float


class Regularizer(ABC):
    """Abstract base class for regularizers.

    Regularizers are used to impose penalties on the parameters of machine learning models to
    prevent overfitting and promote simplicity.

    Attributes:
        None
    """
    @abstractmethod
    def __call__(self, matrix):
        """Calculate the regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            float: The value of the regularization loss.
        """
        pass

    @abstractmethod
    def backward(self, matrix):
        """Calculate the gradient of the regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            numpy.ndarray: The gradient of the regularization term with respect to the weights.
        """
        pass


class L1(Regularizer):
    """L1 regularization, also known as Lasso regularization.

    L1 regularization penalizes the absolute values of the model parameters.

    Attributes:
        l1 (float): The regularization parameter.
    """
    def __init__(self, l1: float = 0.01):
        """Initialize L1 regularization.

        Args:
            l1 (float, optional): The regularization parameter. Defaults to 0.01.
        """
        self.l1 = validate_non_negative_float(l1, 'l1')

    def __call__(self, matrix):
        """Calculate the L1 regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            float: The value of the L1 regularization loss.
        """
        return self.l1 * np.sum(np.abs(matrix), axis=None)

    def backward(self, matrix):
        """Calculate the gradient of the L1 regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            numpy.ndarray: The gradient of the L1 regularization loss with respect to the weights.
        """
        return self.l1 * np.sign(matrix)


class L2(Regularizer):
    """L2 regularization, also known as Ridge regularization.

    L2 regularization penalizes the squared values of the model parameters.

    Attributes:
        l2 (float): The regularization parameter.
    """
    def __init__(self, l2: float = 0.01):
        """Initialize L2 regularization.

        Args:
            l2 (float, optional): The regularization parameter. Defaults to 0.01.
        """
        self.l2 = validate_non_negative_float(l2, 'l2')

    def __call__(self, matrix):
        """Calculate the L2 regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            float: The value of the L2 regularization loss.
        """
        return self.l2 * np.sum(np.square(matrix), axis=None)

    def backward(self, matrix):
        """Calculate the gradient of the L2 regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            numpy.ndarray: The gradient of the L2 regularization loss with respect to the weights.
        """
        return self.l2 * 2.0 * matrix


class L1L2(Regularizer):
    """Combination of L1 and L2 regularization.

    L1L2 regularization combines the penalties of L1 and L2 regularization techniques.

    Attributes:
        l1 (float): The L1 regularization parameter.
        l2 (float): The L2 regularization parameter.
    """
    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        """Initialize L1L2 regularization.

        Args:
            l1 (float, optional): The L1 regularization parameter. Defaults to 0.01.
            l2 (float, optional): The L2 regularization parameter. Defaults to 0.01.
        """
        self.l1 = l1
        self.l2 = l2

    def __call__(self, matrix):
        """Calculate the L1L2 regularization term.

        Args:
            matrix (numpy.ndarray): The weight matrix of the model.

        Returns:
            float: The value of the L1L2 regularization loss.
        """
        return self.l1 * np.sum(np.abs(matrix), axis=None) + self.l2 * np.sum(np.square(matrix), axis=None)

    def backward(self, matrix):
        """Calculate the gradient of the L1L2 regularization term.

        Args:
            matrix (numpy.ndarray): The parameter matrix of the model.

        Returns:
            numpy.ndarray: The gradient of the L1L2 regularization loss with respect to the weights.
        """
        return self.l1 * np.sign(matrix) + self.l2 * 2.0 * matrix
