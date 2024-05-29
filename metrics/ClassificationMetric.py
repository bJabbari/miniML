from abc import ABC

import numpy as np

from metrics import Metric


# Base Metric class
class ClassificationMetric(Metric, ABC):
    def __init__(self, threshold: float = 0.5, average: str = 'micro') -> None:
        """
        Initializes the Metric class with counts set to zero.
        Args:
            threshold (float): Threshold to convert probabilities to binary predictions. It applies only when the
            `y_pred` is not integer. If used with a loss function that sets from_logits=True
            (i.e. no sigmoid or softmax is applied to predictions), thresholds should be set to 0.
            Defaults to 0.5
            average (str): Averaging method, 'micro' or 'macro'. if average is `None`, no average is applied.
        """
        self.true_positives_dict = {}
        self.false_positives_dict = {}
        self.false_negatives_dict = {}
        if threshold is not None and (threshold < 0 or threshold >= 1):
            raise ValueError(
                "Invalid value for argument `threshold`. "
                "Expected a value in interval [0, 1). "
                f"Received: threshold={threshold}"
            )
        self.threshold = threshold
        if not average in ['micro', 'macro', None]:
            raise ValueError("Invalid value for argument `average`. "
                             "Expected 'micro', 'macro' or None. "
                             f"Received: average={average}")
        self.average = average

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


# Function-based precision
def precision(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: str = 'macro') -> float:
    """
    Computes precision.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float): Threshold to convert probabilities to binary predictions.
        average (str): Averaging method, 'micro' or 'macro'.

    Returns:
        float: Precision.
    """
    y_pred = (y_pred >= threshold).astype(int) if len(y_pred.shape) == 1 else np.argmax(y_pred, axis=1)
    y_true = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=1)

    classes = np.unique(y_true)
    true_positives = {cls: np.sum((y_true == cls) & (y_pred == cls)) for cls in classes}
    false_positives = {cls: np.sum((y_true != cls) & (y_pred == cls)) for cls in classes}

    if average == 'micro':
        tp = sum(true_positives.values())
        fp = sum(false_positives.values())
        return tp / (tp + fp) if (tp + fp) > 0 else 0
    else:  # macro
        precisions = [
            true_positives[cls] / (true_positives[cls] + false_positives[cls])
            if (true_positives[cls] + false_positives[cls]) > 0 else 0
            for cls in classes
        ]
        return np.mean(precisions)


# Function-based recall
def recall(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: str = 'macro') -> float:
    """
    Computes recall.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float): Threshold to convert probabilities to binary predictions.
        average (str): Averaging method, 'micro' or 'macro'.

    Returns:
        float: Recall.
    """
    y_pred = (y_pred >= threshold).astype(int) if len(y_pred.shape) == 1 else np.argmax(y_pred, axis=-1)
    y_true = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=-1)

    classes = np.unique(y_true)
    true_positives = {cls: np.sum((y_true == cls) & (y_pred == cls)) for cls in classes}
    false_negatives = {cls: np.sum((y_true == cls) & (y_pred != cls)) for cls in classes}

    if average == 'micro':
        tp = sum(true_positives.values())
        fn = sum(false_negatives.values())
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    else:  # macro
        recalls = [
            true_positives[cls] / (true_positives[cls] + false_negatives[cls])
            if (true_positives[cls] + false_negatives[cls]) > 0 else 0
            for cls in classes
        ]
        return np.mean(recalls)


# Function-based F1 score
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5, average: str = 'macro') -> float:
    """
    Computes the F1 score.

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
        threshold (float): Threshold to convert probabilities to binary predictions.
        average (str): Averaging method, 'micro' or 'macro'.

    Returns:
        float: F1 score.
    """
    y_pred = (y_pred >= threshold).astype(int) if len(y_pred.shape) == 1 else np.argmax(y_pred, axis=1)
    y_true = y_true if len(y_true.shape) == 1 else np.argmax(y_true, axis=1)

    classes = np.unique(y_true)
    true_positives = {cls: np.sum((y_true == cls) & (y_pred == cls)) for cls in classes}
    false_positives = {cls: np.sum((y_true != cls) & (y_pred == cls)) for cls in classes}
    false_negatives = {cls: np.sum((y_true == cls) & (y_pred != cls)) for cls in classes}

    if average == 'micro':
        tp = sum(true_positives.values())
        fp = sum(false_positives.values())
        fn = sum(false_negatives.values())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    else:  # macro
        precisions = [
            true_positives[cls] / (true_positives[cls] + false_positives[cls])
            if (true_positives[cls] + false_positives[cls]) > 0 else 0
            for cls in classes
        ]
        recalls = [
            true_positives[cls] / (true_positives[cls] + false_negatives[cls])
            if (true_positives[cls] + false_negatives[cls]) > 0 else 0
            for cls in classes
        ]
        precision = np.mean(precisions)
        recall = np.mean(recalls)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
