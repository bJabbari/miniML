import numpy as np


class Zeros:
    def __init__(self, dtype: np.dtype = np.float32):
        self.dtype = dtype

    def __call__(self, shape: tuple):
        return np.zeros(shape, dtype=self.dtype)
