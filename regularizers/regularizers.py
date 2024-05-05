from abc import ABC, abstractmethod
import numpy as np

from miniML.helper.validation import validate_non_negative_float


class Regularizer(ABC):
    @abstractmethod
    def __call__(self, matrix):
        pass

    @abstractmethod
    def backward(self, matrix):
        pass


class L1(Regularizer):
    def __init__(self, l1: float = 0.01):
        self.l1 = validate_non_negative_float(l1, 'l1')

    def __call__(self, matrix):
        return self.l1 * np.sum(np.abs(matrix), axis=None)

    def backward(self, matrix):
        return self.l1 * np.sign(matrix)


class L2(Regularizer):
    def __init__(self, l2: float = 0.01):
        self.l2 = validate_non_negative_float(l2, 'l2')

    def __call__(self, matrix):
        return self.l2 * np.sum(np.square(matrix), axis=None)

    def backward(self, matrix):
        return self.l2 * 2.0 * matrix


class L1L2(Regularizer):
    def __init__(self, l1: float = 0.01, l2: float = 0.01):
        self.l1 = l1
        self.l2 = l2

    def __call__(self, matrix):
        return self.l1 * np.sum(np.abs(matrix), axis=None) + self.l2 * np.sum(np.square(matrix), axis=None)

    def backward(self, matrix):
        return self.l1 * np.sign(matrix) + self.l2 * 2.0 * matrix
