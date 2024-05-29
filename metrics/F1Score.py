from typing import Union

import numpy as np

from metrics import ClassificationMetric, Precision, Recall


class F1Score(ClassificationMetric):
    def __init__(self, threshold: float = 0.5, average: str = 'macro') -> None:
        """
        Initializes the F1Score class with the averaging method.

        Args:
            threshold (float): Threshold to convert probabilities to binary predictions, only for Binary Classification.
            Defaults to 0.5
            average (str): Averaging method, 'micro' or 'macro'.
        """
        super().__init__()
        self.average = average
        self.precision_metric = Precision(threshold=threshold, average=average)
        self.recall_metric = Recall(threshold=threshold, average=average)

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for F1 score calculation.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.
        """
        self.precision_metric.update_state(y_true, y_pred)
        self.recall_metric.update_state(y_true, y_pred)

    def result(self) -> Union[float, dict]:
        """
        Computes and returns the F1 score.

        Returns:
            float: The F1 score.
        """
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        if isinstance(precision, dict):
            f1score = {}
            for key, p in precision.items():
                r = recall[key]
                f1score[key] = 2*(p * r)/(p + r) if (p + r) > 0 else 0
        else:
            f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1score
