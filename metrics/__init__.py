from typing import Union, Optional

from miniML.metrics.Metric import Metric
from .AccuracyMetric import AccuracyMetric, binary_accuracy, categorical_accuracy, sparse_categorical_accuracy
from .RegressionMetric import RegressionMetric, mean_absolute_error, mean_squared_error, root_mean_squared_error
from .ClassificationMetric import ClassificationMetric, precision, recall, f1_score
from .BinaryAccuracy import BinaryAccuracy
from .CategoricalAccuracy import CategoricalAccuracy
from .SparseCategoricalAccuracy import SparseCategoricalAccuracy
from .Precision import Precision
from .Recall import Recall
from .F1Score import F1Score
from .MeanAbsoluteError import MeanAbsoluteError
from .MeanSquaredError import MeanSquaredError
from .RootMeanSquaredError import RootMeanSquaredError

all_metrics = [BinaryAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy,
               Precision, Recall, F1Score,
               MeanAbsoluteError, MeanSquaredError, RootMeanSquaredError]
all_metrics_map = {a.__name__.lower(): a for a in all_metrics}
all_metrics_map.update(
    {
        'mse': MeanSquaredError,
        'MSE': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'MAE': MeanAbsoluteError,
        'RMSE': RootMeanSquaredError,
        'rmse': RootMeanSquaredError
    }
)


def get(metric: Union[str, Metric, None]) -> Optional[Metric]:
    if metric is None:
        return None
    elif isinstance(metric, str):
        metric_class = all_metrics_map.get(metric.lower(), None)
        if metric_class is not None:
            metric_instance = metric_class()
        else:
            raise ValueError(f"No metric found for '{metric}'.")
    else:
        metric_instance = metric

    if isinstance(metric_instance, Metric):
        return metric_instance
    else:
        raise ValueError(f"Invalid metric type. Make sure to pass a valid `Metric` subclass instance. "
                         f"Received: '{type(metric_instance).__name__}'.")
