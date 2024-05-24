import numpy as np
from miniML.losses import Loss


class MeanSquaredError(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        return np.mean(np.mean(np.square(y_pred - y_true), axis=-1), axis=None)

    def gradient(self):
        super().gradient()
        return (self.y_pred - self.y_true) * 2.0 / np.prod(self.y_pred.shape)
