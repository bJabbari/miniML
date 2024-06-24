from typing import Union, Optional

from .Optimizer import Optimizer
from .SGD import SGD
from .Adam import Adam
from .RMSprop import RMSprop

all_optimizers = [SGD, Adam]
all_optimizers_map = {a.__name__.lower(): a for a in all_optimizers}


def get(optimizer: Union[str, Optimizer, None]) -> Optional[Optimizer]:
    if optimizer is None:
        return None
    elif isinstance(optimizer, str):
        optimizer_class = all_optimizers_map.get(optimizer.lower(), None)
        if optimizer_class is not None:
            optimizer_instance = optimizer_class()
        else:
            raise ValueError(f"No optimizer found for '{optimizer}'.")
    else:
        optimizer_instance = optimizer

    if isinstance(optimizer_instance, Optimizer):
        return optimizer_instance
    else:
        raise ValueError(f"Invalid optimizer type. Make sure to pass a valid \'Optimizer\' subclass instance. "
                         f"Received: '{type(optimizer_instance).__name__}'.")
