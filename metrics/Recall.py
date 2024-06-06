from typing import Union, Optional

import numpy as np

from metrics import ClassificationMetric


class Recall(ClassificationMetric):
    def __init__(self, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> None:
        """
        Computes recall for a classification model. Recall is defined as the number of true positives
        divided by the number of true positives plus the number of false negatives.

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
        super().__init__('recall', threshold, average, zero_division)

    def update_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Updates the state with the true and predicted values for recall calculation.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Notes:
            - Determines the problem type (binary, multiclass, or multilabel) if not already set.
            - Converts predictions to class labels based on the problem type.
            - Updates true positive and false negative counts for each class.
        """
        if self._problem_type is None:
            self._problem_type = self.determine_problem_type(y_true)
            if self.average == 'auto':
                if self._problem_type == 'binary':
                    self.average = 'binary'
                else:
                    self.average = 'micro'
        y_true, y_pred = self.preprocess_predictions(y_true, y_pred, self._problem_type, self.threshold)
        if self._problem_type in ['binary', 'multiclass']:
            if self._problem_type == 'binary':
                classes = range(2)
            else:
                classes = range(np.max(y_true) + 1)
            for cls in classes:
                tp = np.sum((y_true == cls) & (y_pred == cls))
                fn = np.sum((y_true == cls) & (y_pred != cls))

                if cls in self.true_positives_dict:
                    self.true_positives_dict[cls] += tp
                    self.false_negatives_dict[cls] += fn
                else:
                    self.true_positives_dict[cls] = tp
                    self.false_negatives_dict[cls] = fn
        else:  # self._problem_type == 'multilabel'
            classes = range(y_true.shape[-1])
            for cls in classes:
                tp = np.sum((y_true[:, cls] == 1) & (y_pred[:, cls] == 1))
                fn = np.sum((y_true[:, cls] == 1) & (y_pred[:, cls] == 0))

                if cls in self.true_positives_dict:
                    self.true_positives_dict[cls] += tp
                    self.false_negatives_dict[cls] += fn
                else:
                    self.true_positives_dict[cls] = tp
                    self.false_negatives_dict[cls] = fn

    def result(self) -> Union[float, dict]:
        """
        Computes and returns the recall.

        Returns:
            Union[float, dict]: The recall score. If `average` is `None`, a dictionary with recall for each class is returned.
            Otherwise, a single recall score is returned based on the specified averaging method.

        Notes:
            - For binary classification, returns the recall for the positive class.
            - For multiclass or multilabel classification, returns recall based on the specified averaging method ('micro' or 'macro').
            - If no true positives and false negatives are present, returns the value specified by `zero_division`.
        """
        if self.average is None:
            recall = {
                cls:
                    self.true_positives_dict[cls] / (self.true_positives_dict[cls] + self.false_negatives_dict[cls])
                    if (self.true_positives_dict[cls] + self.false_negatives_dict[cls]) > 0 else self.zero_division
                for cls in self.true_positives_dict
            }
        # Binary case
        elif self.average == 'binary':
            tp = self.true_positives_dict[1]
            fn = self.false_negatives_dict[1]
            recall = tp / (tp + fn) if (tp + fn) > 0 else self.zero_division
        elif self.average == 'micro':
            tp = sum(self.true_positives_dict.values())
            fn = sum(self.false_negatives_dict.values())
            recall = tp / (tp + fn) if (tp + fn) > 0 else self.zero_division
        else:  # self.average == 'macro':
            recalls = [
                self.true_positives_dict[cls] / (self.true_positives_dict[cls] + self.false_negatives_dict[cls])
                if (self.true_positives_dict[cls] + self.false_negatives_dict[cls]) > 0 else self.zero_division
                for cls in self.true_positives_dict
            ]
            recall = np.mean(recalls)

        return recall
