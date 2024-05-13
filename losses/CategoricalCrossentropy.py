import numpy as np
from miniML.losses import Loss


class CategoricalCrossentropy(Loss):
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
        self._evaluate_shape(y_true, y_pred)
        t = np.zeros_like(self.y_pred)
        mask = self.y_true == 1
        if self.from_logits:
            self.y_pred = self._softmax(self.y_pred)  # convert logits to probability
        else:
            self.y_pred = np.clip(self.y_pred, self.eps, 1.0, dtype=np.float64)
        t[mask] = self.y_true[mask] * np.log(self.y_pred[mask])
        return -np.mean(np.sum(t, axis=-1), axis=None)

    def gradient(self):
        super().gradient()
        if self.from_logits:
            return (self.y_pred - self.y_true) / self.n_batches
        else:
            t = np.zeros_like(self.y_pred)
            mask = self.y_true == 1
            _c = -1.0 / self.n_batches
            t[mask] = _c/(self.y_pred[mask])
            return t

    @staticmethod
    def _softmax(x: np.ndarray, axis=-1) -> np.ndarray:
        """
        Compute the softmax function element-wise.

        :param x: Input array.
        :type x: array-like

        :return: Output array of softmax function applied element-wise.
        :rtype: ndarray
        """
        m = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - m)
        return e / e.sum(axis=axis, keepdims=True)

    def _evaluate_shape(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            raise ValueError(f"y_true or y_pred can not be None.")

        # convert a vector to a two-dimensional matrix, with just one sample
        if y_true.ndim <= 1:
            self.y_true = y_true.reshape(1, -1)
        else:
            self.y_true = y_true
        if y_pred.ndim <= 1:
            self.y_pred = y_pred.reshape(1, -1)
        else:
            self.y_pred = y_pred

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(f"y_true and y_pred must have the same shape."
                             f"Received: y_true shape: {self.y_true.shape}, y_pred shape: {self.y_pred.shape}")
            # Check if dtype is numeric with precision less than float64
        if np.issubdtype(self.y_pred.dtype, np.floating) and np.finfo(self.y_pred.dtype).bits < 64:
            self.y_pred = self.y_pred.astype(np.float64)
        self.n_batches = self.y_pred.shape[0]
