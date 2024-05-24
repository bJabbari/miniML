import numpy as np
from miniML.losses import Loss


class BinaryCrossentropy(Loss):
    def __init__(self, from_logits=False, eps=1e-7):
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps

    @property
    def eps(self):
        return self._eps

    @eps.setter
    def eps(self, value):
        if 0 <= value < 1:
            self._eps = value
            return
        raise ValueError(f'eps must be between [0, 1). Received value: {value}')

    def __call__(self, y_true, y_pred):
        super().__call__(y_true, y_pred)
        if self.from_logits:
            self.y_pred = self._sigmoid(self.y_pred)  # convert logits to probability
            return -np.mean(np.sum(
                self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred), axis=-1),
                axis=None)
        _y_pred = np.clip(self.y_pred, self.eps, 1.0 - self.eps, dtype=np.float64)
        return -np.mean(np.sum(
            self.y_true * np.log(_y_pred + self.eps) + (1.0 - self.y_true) * np.log(1.0 - _y_pred + self.eps), axis=-1),
            axis=None)

    def gradient(self):
        super().gradient()
        if self.from_logits:
            return (self.y_pred - self.y_true) / self.n_batches
        else:
            epsilon = self.eps  # to avoid division by zero
            _y_pred = np.clip(self.y_pred, epsilon, 1.0 - epsilon)
            t = np.where(self.y_true == 0, 1.0 / (1 - _y_pred), -1.0 / _y_pred) / self.n_batches

            return t

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
