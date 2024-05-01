import numpy as np


def linear(x):
    return x


def sigmoid(x):
    return 1.0 / (1 + np.exp(-1 * x))


def tanh(x):
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.maximum(x)
    _x = x - x_max
    x_exp = np.exp(_x)
    return x_exp / np.sum(x_exp)


def relu(x):
    return np.maximum(0, x)


def relu6(x):
    return np.minimum(relu(x), 6)
