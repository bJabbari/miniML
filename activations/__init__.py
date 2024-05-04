
# from miniML.activations.activations import linear
# from miniML.activations.activations import sigmoid
# from miniML.activations.activations import tanh
# from miniML.activations.activations import relu
# from miniML.activations.activations import relu6
# from miniML.activations.activations import leaky_relu
# from miniML.activations.activations import softmax

# all_activations = [linear, sigmoid, tanh, relu, relu6, leaky_relu, softmax]
all_activations=[]
all_activations_map = {a.__name__: a for a in all_activations}


def get(activation):
    if activation is None:
        obj = () #linear
    elif isinstance(activation, str):
        obj = all_activations_map.get(activation, None)
    else:
        obj = activation

    if callable(obj):
        return obj
    else:
        raise ValueError("activation function is not recognized")

# def get_derative(activation):
#     if activation.__name__ == linear.__name__:
#         return 1
#     elif activation.__name__ == sigmoid.__name__:
#         return lambda x: x*(1.0-x)
#     elif activation.__name__ == tanh.__name__:
#         return lambda x: (1.0-x**2)
#     elif activation.__name__ == relu.__name__:
#         return lambda x: 1.0 if x>=0 else 0.0
#     elif activation.__name__ == relu6.__name__:
#         return lambda x: 1.0 if 0<= x<=6 else 0.0
#     elif activation.__name__ == leaky_relu.__name__:
#         return lambda x: 1.0 if  else 0.0
