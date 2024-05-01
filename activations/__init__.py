
from miniML.activations.activations import linear
from miniML.activations.activations import sigmoid
from miniML.activations.activations import tanh
from miniML.activations.activations import relu
from miniML.activations.activations import relu6
from miniML.activations.activations import leaky_relu
from miniML.activations.activations import softmax

all_activations = [linear, sigmoid, tanh, relu, relu6, leaky_relu, softmax]

all_activations_map = {a.__name__: a for a in all_activations}


def get(activation):
    if activation is None:
        obj = linear
    elif isinstance(activation, str):
        obj = all_activations_map.get(activation, None)
    else:
        obj = activation

    if callable(obj):
        return obj
    else:
        raise ValueError("activation function is not recognized")
