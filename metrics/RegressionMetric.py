from abc import ABC

import numpy as np

from metrics import Metric


# Base Metric class for regression
class RegressionMetric(Metric, ABC):
    def __init__(self, name: str) -> None:
        """
        Initializes the RegressionMetric class with error sums and counts set to zero.
        Args:
            name (str): Name of the regression metric
        """
        self.name = name
        self.sum_error = 0.0
        self.count = 0

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        """
        raise NotImplementedError

    def result(self) -> float:
        """Computes and returns the metric."""
        raise NotImplementedError

    def reset_state(self) -> None:
        """Resets the error sums and counts to zero."""
        self.sum_error = 0.0
        self.count = 0


# Function-based mean absolute error
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the mean absolute error.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The mean absolute error.
    """
    return np.mean(np.mean(np.abs(y_pred - y_true), axis=-1), axis=None)


# Function-based mean squared error
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the mean squared error.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The mean squared error.
    """
    return np.mean(np.mean(np.square(y_pred - y_true), axis=-1), axis=None)


# Function-based root mean squared error
def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the root mean squared error.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The root mean squared error.
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)
