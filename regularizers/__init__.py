from typing import Union, Optional
from .regularizers import Regularizer, L1, L2, L1L2

all_regularizers = [L1, L2, L1L2]
all_regularizers_map = {r.__name__.lower(): r for r in all_regularizers}


def get(regularizer: Union[str, Regularizer, None]) -> Optional[Regularizer]:
    if regularizer is None:
        return None
    elif isinstance(regularizer, str):
        regularizer_class = all_regularizers_map.get(regularizer.lower(), None)
        if regularizer_class is not None:
            regularizer_instance = regularizer_class()
        else:
            raise ValueError(f"No regularizer found for '{regularizer}'.")
    else:
        regularizer_instance = regularizer

    if isinstance(regularizer_instance, Regularizer):
        return regularizer_instance
    else:
        raise ValueError(f"Invalid regularizer type. Make sure to pass a valid \'Regularizer\' subclass instance. "
                         f"Received: '{type(regularizer_instance).__name__}'.")
