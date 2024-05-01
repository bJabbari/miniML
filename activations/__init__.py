from miniML.activations.activations import linear
from miniML.activations.activations import sigmoid
from miniML.activations.activations import tanh
from miniML.activations.activations import relu
from miniML.activations.activations import relu6
from miniML.activations.activations import softmax

all_activations = [linear, sigmoid, tanh, relu, relu6, softmax]

all_activations_map = {a.__name__: a for a in all_activations}


def get(activation):
    if isinstance(activation, str):
        obj = all_activations_map.get(activation, None)
    else:
        obj = activation

    if callable(obj):
        return obj
    else:
        raise ValueError("input is not an activation function")
