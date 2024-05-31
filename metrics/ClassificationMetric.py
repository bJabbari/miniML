from abc import ABC
from typing import Optional, Union

import numpy as np

from metrics import Metric


# Base Metric class
class ClassificationMetric(Metric, ABC):
    def __init__(self, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> None:
        """
         Initializes the Metric class with default settings.

            Args:
              threshold (float, optional): The threshold value to convert probabilities into binary predictions.
                  This parameter is only relevant when `y_pred` contains probabilities instead of integer labels.
                  If using a loss function with `from_logits=True` (i.e., no sigmoid or softmax applied to predictions),
                  set this threshold to 0. Defaults to 0.5.

              average (str, optional): The averaging method to be used. Options include:
                  - 'auto': For binary classification, returns the result for class `1`. For multiclass or multilabel
                    problems, it defaults to 'micro' averaging.
                  - 'binary': Only for binary classification; calculates metrics for class `1`.
                  - 'micro': Computes metrics globally by counting the total true positives, false negatives, and false positives.
                  - 'macro': Computes metrics for each label and finds their unweighted mean.
                  - None: No averaging is applied; metrics for each class are returned separately.
                  Defaults to 'auto'.

              zero_division (float, optional): Sets the value to be returned when there is a zero division situation (e.g., when
                  there are no positive cases in precision or recall calculation). Defaults to 0.0.

        """
        self.true_positives_dict = {}
        self.false_positives_dict = {}
        self.false_negatives_dict = {}
        if threshold is not None and (threshold < 0 or threshold >= 1):
            raise ValueError(
                "Invalid value for argument `threshold`. "
                "Expected a value in interval [0, 1) or `None`. "
                f"Received: threshold = {threshold}"
            )
        self.threshold = threshold
        if average is not None and isinstance(average, str):
            average = average.lower()
        if not average in ['auto', 'binary', 'micro', 'macro', None]:
            raise ValueError("Invalid value for argument `average`. "
                             "Expected 'binary', 'micro', 'macro' or `None`. "
                             f"Received: {average}")
        self.average = average
        if not zero_division in [0.0, 1.0, np.nan]:
            raise ValueError("Invalid value for argument `zero_division`. "
                             f"Expected 0.0, 1.0 or `None`. Received: {zero_division}"
                             )
        self.zero_division = zero_division
        self._problem_type = None  # 'binary', 'multiclass', or 'multilabel'

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
        """Resets the counts to zero."""
        self.true_positives_dict = {}
        self.false_positives_dict = {}
        self.false_negatives_dict = {}
        self._problem_type = None


# Function-based precision
def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> Union[float, dict]:
    """
    Computes and returns the precision for binary, multiclass, and multilabel classification problems.
    You can provide `y_pred` as either class labels or probabilities.
    If `average` is `None`, a dictionary with the precision for each class is returned.
    Otherwise, a single precision score is returned based on the specified averaging method.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float, optional): The threshold value for converting probabilities into
            binary predictions. This parameter is applicable only for binary and multilabel problems,
            and only when `y_pred` contains probabilities instead of integer labels. If using a
            loss function with `from_logits=True` (i.e., no sigmoid or softmax applied to predictions),
            set this threshold to 0. Defaults to 0.5.

        average (str, optional):
            Averaging method to use. Options include 'auto', 'binary', 'micro', 'macro', or None.
            Defaults to 'auto'

            - 'auto': For binary classification, returns the result for class `1`. For multiclass or multilabel problems, it defaults to 'micro' averaging.
            - 'binary': Only for binary classification; calculates metrics for class `1`.
            - 'micro': Computes metrics globally by counting the total true positives, false negatives, and false positives.
            - 'macro': Computes metrics for each label and finds their unweighted mean.
            - None: No averaging is applied; metrics for each class are returned separately.

        zero_division (float, optional): Sets the value to be returned when there is a zero division
            situation (e.g., when there are no positive cases). Defaults to 0.0.

    Returns:
            Union[float, dict]: The precision score. If `average` is `None`, a dictionary with precision for each class
            is returned. Otherwise, a single precision score is returned based on the specified averaging method.
    """
    _problem_type = Metric.determine_problem_type(y_true)
    if average == 'auto':
        if _problem_type == 'binary':
            average = 'binary'
        else:
            average = 'micro'
    y_true, y_pred = Metric.preprocess_predictions(y_true, y_pred, _problem_type, threshold)
    true_positives_dict = {}
    false_positives_dict = {}
    if _problem_type in ['binary', 'multiclass']:
        if _problem_type == 'binary':
            classes = range(2)
        else:
            classes = range(np.max(y_true) + 1)
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            true_positives_dict[cls] = tp
            false_positives_dict[cls] = fp
    else:  # _problem_type == 'multilabel'
        classes = range(y_true.shape[-1])
        for cls in classes:
            tp = np.sum((y_true[:, cls] == 1) & (y_pred[:, cls] == 1))
            fp = np.sum((y_true[:, cls] == 0) & (y_pred[:, cls] == 1))
            true_positives_dict[cls] = tp
            false_positives_dict[cls] = fp

    if average is None:
        precision_res = {
            cls:
                true_positives_dict[cls] / (true_positives_dict[cls] + false_positives_dict[cls])
                if (true_positives_dict[cls] + false_positives_dict[cls]) > 0 else zero_division
            for cls in true_positives_dict
        }
        # Binary case
    elif average == 'binary':
        tp = true_positives_dict[1]
        fp = false_positives_dict[1]
        precision_res = tp / (tp + fp) if (tp + fp) > 0 else zero_division

    elif average == 'micro':
        tp = sum(true_positives_dict.values())
        fp = sum(false_positives_dict.values())
        precision_res = tp / (tp + fp) if (tp + fp) > 0 else zero_division

    else:  # average == 'macro':
        precisions = [
            true_positives_dict[cls] / (true_positives_dict[cls] + false_positives_dict[cls])
            if (true_positives_dict[cls] + false_positives_dict[cls]) > 0 else zero_division
            for cls in true_positives_dict
        ]
        precision_res = np.mean(precisions)

    return precision_res


# Function-based recall
def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> Union[float, dict]:
    """
    Computes and returns the recall for binary, multiclass, and multilabel classification problems.
    You can provide `y_pred` as either class labels or probabilities.
    If `average` is `None`, a dictionary with the recall for each class is returned.
    Otherwise, a single recall score is returned based on the specified averaging method.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float, optional): The threshold value for converting probabilities into
            binary predictions. This parameter is applicable only for binary and multilabel problems,
            and only when `y_pred` contains probabilities instead of integer labels. If using a
            loss function with `from_logits=True` (i.e., no sigmoid or softmax applied to predictions),
            set this threshold to 0. Defaults to 0.5.

        average (str, optional):
            Averaging method to use. Options include 'auto', 'binary', 'micro', 'macro', or None.
            Defaults to 'auto'

            - 'auto': For binary classification, returns the result for class `1`. For multiclass or multilabel problems, it defaults to 'micro' averaging.
            - 'binary': Only for binary classification; calculates metrics for class `1`.
            - 'micro': Computes metrics globally by counting the total true positives, false negatives, and false positives.
            - 'macro': Computes metrics for each label and finds their unweighted mean.
            - None: No averaging is applied; metrics for each class are returned separately.

        zero_division (float, optional): Sets the value to be returned when there is a zero division
            situation (e.g., when there are no positive cases). Defaults to 0.0.

    Returns:
            Union[float, dict]: The recall score. If `average` is `None`, a dictionary with recall for each class
            is returned. Otherwise, a single recall score is returned based on the specified averaging method.
    """
    _problem_type = Metric.determine_problem_type(y_true)
    if average == 'auto':
        if _problem_type == 'binary':
            average = 'binary'
        else:
            average = 'micro'
    y_true, y_pred = Metric.preprocess_predictions(y_true, y_pred, _problem_type, threshold)
    true_positives_dict = {}
    false_negatives_dict = {}
    if _problem_type in ['binary', 'multiclass']:
        if _problem_type == 'binary':
            classes = range(2)
        else:
            classes = range(np.max(y_true) + 1)
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            true_positives_dict[cls] = tp
            false_negatives_dict[cls] = fn
    else:  # _problem_type == 'multilabel'
        classes = range(y_true.shape[-1])
        for cls in classes:
            tp = np.sum((y_true[:, cls] == 1) & (y_pred[:, cls] == 1))
            fn = np.sum((y_true[:, cls] == 1) & (y_pred[:, cls] == 0))
            true_positives_dict[cls] = tp
            false_negatives_dict[cls] = fn

    if average is None:
        recall_res = {
            cls:
                true_positives_dict[cls] / (true_positives_dict[cls] + false_negatives_dict[cls])
                if (true_positives_dict[cls] + false_negatives_dict[cls]) > 0 else zero_division
            for cls in true_positives_dict
        }
    # Binary case
    elif average == 'binary':
        tp = true_positives_dict[1]
        fn = false_negatives_dict[1]
        recall_res = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    elif average == 'micro':
        tp = sum(true_positives_dict.values())
        fn = sum(false_negatives_dict.values())
        recall_res = tp / (tp + fn) if (tp + fn) > 0 else zero_division
    else:  # average == 'macro':
        recalls = [
            true_positives_dict[cls] / (true_positives_dict[cls] + false_negatives_dict[cls])
            if (true_positives_dict[cls] + false_negatives_dict[cls]) > 0 else zero_division
            for cls in true_positives_dict
        ]
        recall_res = np.mean(recalls)

    return recall_res


# Function-based F1 score
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: Optional[str] = 'auto', zero_division=0.0) -> Union[float, dict]:
    """
    Computes and returns the F1 score for binary, multiclass, and multilabel classification problems.
    You can provide `y_pred` as either class labels or probabilities.
    If `average` is `None`, a dictionary with the F1 score for each class is returned.
    Otherwise, a single F1 score is returned based on the specified averaging method.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float, optional): The threshold value for converting probabilities into
            binary predictions. This parameter is applicable only for binary and multilabel problems,
            and only when `y_pred` contains probabilities instead of integer labels. If using a
            loss function with `from_logits=True` (i.e., no sigmoid or softmax applied to predictions),
            set this threshold to 0. Defaults to 0.5.

        average (str, optional):
            Averaging method to use. Options include 'auto', 'binary', 'micro', 'macro', or None.
            Defaults to 'auto'

            - 'auto': For binary classification, returns the result for class `1`. For multiclass or multilabel problems, it defaults to 'micro' averaging.
            - 'binary': Only for binary classification; calculates metrics for class `1`.
            - 'micro': Computes metrics globally by counting the total true positives, false negatives, and false positives.
            - 'macro': Computes metrics for each label and finds their unweighted mean.
            - None: No averaging is applied; metrics for each class are returned separately.

        zero_division (float, optional): Sets the value to be returned when there is a zero division
            situation (e.g., when there are no positive cases). Defaults to 0.0.

    Returns:
            Union[float, dict]: The F1 score. If `average` is `None`, a dictionary with F1 score for each class
            is returned. Otherwise, a single F1 score is returned based on the specified averaging method.
    """
    _average = average if average != 'macro' else None
    precision_res = precision(y_true, y_pred, threshold=threshold, average=_average)
    recall_res = recall(y_true, y_pred, threshold=threshold, average=_average)
    if isinstance(precision_res, dict):
        f1score = {}
        for key, p in precision_res.items():
            r = recall_res[key]
            f1score[key] = 2 * (p * r) / (p + r) if (p + r) > 0 else zero_division
    else:
        f1score = 2 * (precision_res * recall_res) / (precision_res + recall_res) if (precision_res + recall_res) > 0 \
            else zero_division
    if average == 'macro':
        f1score = np.mean(list(f1score.values()))
    return f1score
