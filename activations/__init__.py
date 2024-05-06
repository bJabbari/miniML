from typing import Union

from .activations import Linear, Sigmoid, Tanh
from .activations import ReLU, ReLU6, LeakyReLU
from .activations import Softmax
from .Activation_function import Activation

all_activations = [Linear, Sigmoid, Tanh, ReLU, ReLU6, LeakyReLU, Softmax]
all_activations_map = {a.__name__.lower(): a for a in all_activations}
all_activations_map['leaky_relu']= LeakyReLU


def get(activation: Union[str, Activation, None]) -> Activation:
    if activation is None:
        return Linear()
    elif isinstance(activation, str):
        activation_class = all_activations_map.get(activation.lower(), None)
        if activation_class is not None:
            activation_instance = activation_class()
        else:
            raise ValueError(f"No activation found for '{activation}'.")
    else:
        activation_instance = activation

    if isinstance(activation_instance, Activation):
        return activation_instance
    else:
        raise ValueError(f"Invalid activation type. Make sure to pass a valid \'Activation\' subclass instance. "
                         f"Received: '{type(activation_instance).__name__}'.")
