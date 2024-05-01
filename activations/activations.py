import numpy as np


def linear(x):
    return x


@np.vectorize
def sigmoid(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        z = np.exp(x)
        return z / (1 + z)


def tanh(x):
    return np.tanh(x)


def softmax(values: np.ndarray, axis=-1) -> np.ndarray:
    m = np.max(values, axis=axis, keepdims=True)
    e = np.exp(values - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu6(x):
    return np.minimum(np.maximum(0, x), 6)


@np.vectorize
def leaky_relu(x, alpha=0.01):
    return alpha * x if x <= 0 else x

