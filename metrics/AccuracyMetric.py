from abc import ABC

import numpy as np

from metrics import Metric


# Base Accuracy class
class AccuracyMetric(Metric, ABC):
    def __init__(self, zero_division=0.0) -> None:
        """
        Initializes the Accuracy class with correct and total counts set to zero.
        Args:
            zero_division (float, optional): Sets the value to be returned when there is a zero division situation (e.g., when
            there are no positive cases). Defaults to 0.0.
        """
        self.correct = 0
        self.total = 0
        if not zero_division in [0.0, 1.0, np.nan]:
            raise ValueError("Invalid value for argument `zero_division`. "
                             f"Expected 0.0, 1.0 or `None`. Received: {zero_division}"
                             )
        self.zero_division = zero_division

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        """
        raise NotImplementedError

    def result(self) -> float:
        """
        Computes and returns the accuracy.

        Returns:
            float: The accuracy as the ratio of correct predictions to total samples.
        """
        return self.correct / self.total if self.total > 0 else self.zero_division

    def reset_state(self) -> None:
        """Resets the correct and total counts to zero."""
        self.correct = 0
        self.total = 0


# Function-based binary accuracy
def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes binary accuracy.

    Args:
        y_true (np.ndarray): Ground truth binary values.
        y_pred (np.ndarray): Predicted binary values.
        threshold (float): Threshold to convert probabilities to binary predictions.

    Returns:
        float: Binary accuracy.
    """
    y_pred = (y_pred >= threshold).astype(int)
    return np.mean(y_true == y_pred)


# Function-based categorical accuracy
def categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes categorical accuracy.

    Args:
        y_true (np.ndarray): Ground truth one-hot encoded values.
        y_pred (np.ndarray): Predicted probabilities for each class.

    Returns:
        float: Categorical accuracy.
    """
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    return np.mean(y_true == y_pred)


# Function-based sparse categorical accuracy
def sparse_categorical_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes sparse categorical accuracy.

    Args:
        y_true (np.ndarray): Ground truth integer encoded values.
        y_pred (np.ndarray): Predicted probabilities for each class.

    Returns:
        float: Sparse categorical accuracy.
    """
    y_pred = np.argmax(y_pred, axis=-1)
    return np.mean(y_true == y_pred)
