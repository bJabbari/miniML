from typing import Union, Optional

import numpy as np

from metrics import ClassificationMetric, Precision, Recall


class F1Score(ClassificationMetric):
    def __init__(self, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> None:
        """
        Computes the F1 score for a classification model. The F1 score is the harmonic mean of precision and recall.
        Initializes the F1Score class with the averaging method.

        Args:
            threshold (float, optional): Threshold to convert probabilities to binary predictions. Defaults to 0.5.
            average (str, optional): Averaging method to use. Options include 'auto', 'binary', 'micro', 'macro', or None. Defaults to 'auto'.

                - 'auto': For binary classification, returns the result for class `1`. For multiclass or multilabel problems, it defaults to 'micro' averaging.
                - 'binary': Only for binary classification; calculates metrics for class `1`.
                - 'micro': Computes metrics globally by counting the total true positives, false negatives, and false positives.
                - 'macro': Computes metrics for each label and finds their unweighted mean.
                - None: No averaging is applied; metrics for each class are returned separately.
                Defaults to 'auto'.
            zero_division (float, optional): Sets the value to be returned when there is a zero division situation (e.g., when
            there are no positive cases). Defaults to 0.0.
        """
        super().__init__('F1score', threshold, average, zero_division)
        self.average = average
        _average = average if average != 'macro' else None
        self.precision_metric = Precision(threshold=threshold, average=_average)
        self.recall_metric = Recall(threshold=threshold, average=_average)

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for F1 score calculation.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Notes:
            - Updates the internal state of precision and recall metrics.
        """
        self.precision_metric.update_state(y_true, y_pred)
        self.recall_metric.update_state(y_true, y_pred)

    def result(self) -> Union[float, dict]:
        """
        Computes and returns the F1 score.

        Returns:
            Union[float, dict]: The F1 score. If `average` is `None`, a dictionary with F1 for each class is returned.
            Otherwise, a single F1 score is returned based on the specified averaging method.

        Notes:
            - For binary classification, returns the F1 score for the positive class.
            - For multiclass or multilabel classification, returns F1 score based on the specified averaging method ('micro' or 'macro').
            - If no positive predictions or true positives are present, returns the value specified by `zero_division`.
        """
        precision = self.precision_metric.result()
        recall = self.recall_metric.result()
        if isinstance(precision, dict):
            f1score = {}
            for key, p in precision.items():
                r = recall[key]
                f1score[key] = 2*(p * r)/(p + r) if (p + r) > 0 else self.zero_division
        else:
            f1score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 \
                else self.zero_division
        if self.average == 'macro':
           f1score = np.mean(list(f1score.values()))
        return f1score
