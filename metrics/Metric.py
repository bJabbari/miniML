from abc import ABC

import numpy as np

from miniML.helper.helper import is_scalar_vector


class Metric(ABC):
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
        """Resets the states."""
        raise NotImplementedError


    @staticmethod
    def determine_problem_type(y_true):
        """
            Determines the problem type based on the shape and values of `y_true`.

            Args:
                y_true (array-like): Ground truth labels. It can be a scalar, a 1-dimensional array, or a 2-dimensional array.

            Returns:
                str: The type of problem, which can be 'binary', 'multiclass', or 'multilabel'.

            Raises:
                ValueError: If `y_true` is a 2-dimensional array with elements that are not 0 or 1, or if it has an unsupported shape.

            Notes:
                - If `y_true` is a scalar, a 1-dimensional array, or a 2-dimensional row or column vector, the function will determine
                  whether the problem is binary or multiclass based on the unique values in `y_true`.
                - For a non-vector 2-dimensional `y_true`, the function will determine whether the problem is multiclass or multilabel
                  based on the structure of the array.
                - A binary problem is identified if `y_true` contains at most two unique values.
                - A multiclass problem is identified if `y_true` contains more than two unique values or is a 2-dimensional array
                  with each row having exactly one `1` (one-hot encoded).
                - A multilabel problem is identified if `y_true` is a 2-dimensional array where each row can have multiple `1`s.

            Examples:
                >>> Metric.determine_problem_type(np.array([0, 1, 0, 1]))
                'binary'
                >>> Metric.determine_problem_type(np.array([0, 1, 2]))
                'multiclass'
                >>> Metric.determine_problem_type(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
                'multiclass'
                >>> Metric.determine_problem_type(np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]]))
                'multilabel'
            """
        if is_scalar_vector(y_true):
            unique_values = np.unique(y_true)
            if len(unique_values) <= 2:
                return 'binary'
            else:
                return 'multiclass'
        elif np.ndim(y_true) == 2:
            if np.array_equal(y_true, y_true.astype(bool)):
                row_sums = np.sum(y_true, axis=1)
                if np.all(row_sums == 1):
                    return 'multiclass'
                else:
                    return 'multilabel'
            else:
                raise ValueError("Unsupported format for y_true. "
                                 "For 2-dimensional y_true, all elements should be 0 or 1.")
        else:
            raise ValueError("Unsupported shape for y_true"
                             f"Received shape: {y_true.shape}")

    @staticmethod
    def preprocess_predictions(y_true, y_pred, problem_type, threshold=0.5):
        """
        Preprocesses predictions based on the specified problem type.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted probabilities or logits.
            problem_type (str): The type of problem. Supported types are:
                - 'binary': Binary classification.
                - 'multiclass': Multiclass classification.
                - 'multilabel': Multilabel classification.
            threshold (float, optional): Threshold for converting probabilities to binary class labels for
                binary and multilabel problems. Defaults to 0.5. if threshold value is 0 it won't be applied'

        Returns:
            tuple: A tuple containing:
                - y_true (array-like): Processed ground truth labels.
                - y_pred_labels (array-like): Processed predicted class labels.

        Raises:
            ValueError: If `problem_type` is not one of 'binary', 'multiclass', or 'multilabel'.

        Notes:
            - For binary classification, `y_pred` is thresholded to produce binary labels.
            - For multiclass classification, the class with the highest probability is selected.
            - For multilabel classification, `y_pred` is thresholded to produce binary labels for each class.
        """
        if problem_type == 'binary':
            y_true = np.squeeze(y_true)
            if threshold > 0:
                y_pred_labels = np.squeeze((y_pred >= threshold).astype(int))
            else:
                y_pred_labels = np.squeeze(y_pred)
        elif problem_type == 'multiclass':
            y_true = np.squeeze(y_true)
            y_true = np.argmax(y_true, axis=-1) if np.ndim(y_true) == 2 else y_true
            y_pred = np.squeeze(y_pred)
            y_pred_labels = np.argmax(y_pred, axis=-1) if np.ndim(y_pred) == 2 else y_pred
        elif problem_type == 'multilabel':
            if threshold > 0:
                y_pred_labels = (y_pred >= threshold).astype(int)
            else:
                y_pred_labels = y_pred
        else:
            raise ValueError("Unsupported problem type")
        return y_true, y_pred_labels

