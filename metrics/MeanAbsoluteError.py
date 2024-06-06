import numpy as np

from metrics import RegressionMetric


class MeanAbsoluteError(RegressionMetric):
    def __init__(self):
        super().__init__('mae')
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for mean absolute error calculation.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        """
        error = np.abs(y_true - y_pred)
        if y_pred.ndim <= 1:
            self.sum_error += np.sum(error)
            self.count += y_pred.size
        else:
            self.sum_error += np.sum(np.mean(error, axis=-1), axis=None)
            self.count += y_pred.shape[0]

    def result(self) -> float:
        """
        Computes and returns the mean absolute error.

        Returns:
            float: The mean absolute error.
        """
        return self.sum_error / self.count if self.count > 0 else 0.0
