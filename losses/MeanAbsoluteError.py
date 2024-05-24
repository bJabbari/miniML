import numpy as np
from miniML.losses import Loss


class MeanAbsoluteError(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        return np.mean(np.sum(np.abs(y_pred - y_true), axis=-1), axis=None)

    def gradient(self):
        super().gradient()
        return np.sign(self.y_pred - self.y_true) / self.n_batches
