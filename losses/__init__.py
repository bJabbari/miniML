from typing import Union, Optional

from .Loss import Loss
from .MeanSquaredError import MeanSqauredError
from .MeanAbsoluteError import MeanAbsoluteError
from .BinaryCrossentropy import BinaryCrossentropy
from .CategoricalCrossentropy import CategoricalCrossentropy

all_losses = [MeanSqauredError, MeanAbsoluteError, BinaryCrossentropy, CategoricalCrossentropy]
all_losses_map = {a.__name__.lower(): a for a in all_losses}
all_losses_map.update(
    {
        'mse': MeanSquaredError,
        'MSE': MeanSquaredError,
        'mae': MeanAbsoluteError,
        'MAE': MeanAbsoluteError,
        'bce': BinaryCrossentropy,
        'BCE': BinaryCrossentropy,
        'cce': CategoricalCrossentropy,
        'CCE': CategoricalCrossentropy
    }
)


def get(loss: Union[str, Loss, None]) -> Optional[Loss]:
    if loss is None:
        return None
    elif isinstance(loss, str):
        loss_class = all_losses_map.get(loss.lower(), None)
        if loss_class is not None:
            loss_instance = loss_class()
        else:
            raise ValueError(f"No loss function found for '{loss}'.")
    else:
        loss_instance = loss

    if isinstance(loss_instance, Loss):
        return loss_instance
    else:
        raise ValueError(f"Invalid loss type. Make sure to pass a valid \'Loss\' subclass instance. "
                         f"Received: '{type(loss_instance).__name__}'.")
