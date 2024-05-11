from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.y_true = None
        self.y_pred = None
        self.n_batches = 1


    @abstractmethod
    def __call__(self, y_true, y_pred):
        if y_true is None or y_pred is None:
            raise ValueError(f"y_true or y_pred can not be None.")


        # convert vectors to a two-dimensional matrix, for backpropagation
        if y_true.ndim <= 1:
            self.y_true = y_true.reshape(-1, 1)
        else:
            self.y_true = y_true
        if y_pred.ndim <= 1:
            self.y_pred = y_pred.reshape(-1, 1)
        else:
            self.y_pred = y_pred

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(f"y_true and y_pred must have the same shape.")
        self.n_batches = self.y_pred.shape[0]
    @abstractmethod
    def gradient(self):
        if self.y_true is None or self.y_pred is None:
            raise ValueError("y_true or y_pred can not be None.")
        if self.n_batches == 0:
            raise ValueError("batch size cannot be zero. You need to provide at least one sample")

