from .activations import Linear, Sigmoid, Tanh
from .activations import ReLU, ReLU6, LeakyReLU
from .activations import Softmax

all_activations = [Linear, Sigmoid, Tanh, ReLU, ReLU6, LeakyReLU, Softmax]
all_activations_map = {a.__name__.lower(): a for a in all_activations}


def get(activation):
    if activation is None:
        obj = Linear()
    elif isinstance(activation, str):
        obj = all_activations_map.get(activation.lower(), None)
        if obj is not None:
            obj = obj()
    else:
        obj = activation

    if callable(obj):
        return obj
    else:
        raise ValueError("activation function is not recognized")


