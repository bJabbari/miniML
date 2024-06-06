import numpy as np

from metrics import RegressionMetric
from metrics import MeanSquaredError


class RootMeanSquaredError(RegressionMetric):
    def __init__(self) -> None:
        super().__init__('rmse')
        self._mse = MeanSquaredError()

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for root mean squared error calculation.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        """
        self._mse.update_state(y_true, y_pred)

    def result(self) -> float:
        """
        Computes and returns the root mean squared error.

        Returns:
            float: The root mean squared error.
        """
        return np.sqrt(self._mse.result())
