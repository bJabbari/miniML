import numpy as np

from metrics import AccuracyMetric


class SparseCategoricalAccuracy(AccuracyMetric):
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for sparse categorical classification.

        Args:
            y_true (np.ndarray): Ground truth integer encoded values.
            y_pred (np.ndarray): Predicted probabilities for each class.
        """
        y_pred = np.argmax(y_pred, axis=-1)
        if y_pred.ndim <= 1:
            y_true = np.squeeze(y_true)
        self.correct += np.sum(y_true == y_pred)
        self.total += len(y_true)
