import numpy as np

from metrics import AccuracyMetric


class CategoricalAccuracy(AccuracyMetric):
    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for categorical classification.

        Args:
            y_true (np.ndarray): Ground truth one-hot encoded values.
            y_pred (np.ndarray): Predicted probabilities for each class.
        """
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.argmax(y_pred, axis=-1)
        self.correct += np.sum(y_true == y_pred)
        self.total += len(y_true)
