import numpy as np

from metrics import AccuracyMetric


class BinaryAccuracy(AccuracyMetric):
    def __init__(self, threshold: float = 0.5) -> None:
        """
            Initializes the Binary Accuracy class with correct and total counts set to zero.

            Args:
                threshold (float): Threshold to convert probabilities to binary predictions. Default is 0.5.
        """
        if threshold is not None and (threshold <= 0 or threshold >= 1):
            raise ValueError(
                "Invalid value for argument `threshold`. "
                "Expected a value in interval (0, 1). "
                f"Received: threshold={threshold}"
            )
        super().__init__()
        self.threshold = threshold

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for binary classification.

        Args:
            y_true (np.ndarray): Ground truth binary values.
            y_pred (np.ndarray): Predicted binary values.
        """
        y_pred = (y_pred >= self.threshold).astype(int)
        self.correct += np.sum(y_true == y_pred)
        self.total += len(y_true)
