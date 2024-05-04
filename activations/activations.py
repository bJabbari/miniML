import numbers

import numpy as np


from miniML.activations.Activation_function import Activation
from miniML.helper.helper import multiply_2D_with_3D
from miniML.helper.validation import validate_scalar_vector


class Linear(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return x

    def derivative(self, input_values, return_as_matrix=False, **kwargs):
        """
        Compute the derivative of the linear activation function.

        :param input_values: The input value(s) at which to compute the derivative.
        :type input_values: numeric or array-like
        :param return_as_matrix: If True, returns the derivative as a 2-D matrix.
                                 Defaults to False, returning a 1-D array when False.
        :type return_as_matrix: bool, optional

        :return: The derivative(s) of the linear activation function.
        :rtype: numeric or array-like

        :raises ValueError: If input_values is not a valid vector.
        """
        if isinstance(input_values, numbers.Number):
            return 1
        if self.check_validity:
            validate_scalar_vector(input_values)
        if not return_as_matrix:
            return np.ones(np.shape(input_values), dtype=input_values.dtype)
        else:
            return np.eye(np.size(input_values), dtype=input_values.dtype)

    def back_propagation(self, gradient, **kwargs):
        return gradient


class Sigmoid(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return self._sigmoid(x)

    @staticmethod
    def _sigmoid(x):
        """
        Compute the sigmoid function element-wise.

        :param x: Input array.
        :type x: array-like

        :return: Output array of sigmoid function applied element-wise.
        :rtype: ndarray

        """
        return 1.0 / (1.0 + np.exp(-x))

    def derivative(self, input_values, return_as_matrix=False, activation_output=False):
        if not activation_output:
            z = self(input_values)
        else:
            z = input_values

        if isinstance(z, numbers.Number):
            z = np.array(z)

        dz = z * (1.0 - z)
        if self.check_validity:
            validate_scalar_vector(input_values)

        if not return_as_matrix:
            return dz
        else:
            return np.diagflat(dz)

    def back_propagation(self, gradient, x, activation_output=True):
        super().back_propagation(gradient, x, activation_output)
        z = x
        if not activation_output:
            z = self(x)

        dz = z * (1.0 - z)
        return gradient * dz


class Tanh(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return self._tanh(x)

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    def derivative(self, input_values, return_as_matrix=False, activation_output=False):
        if not activation_output:
            z = self(input_values)
        else:
            z = input_values

        if isinstance(z, numbers.Number):
            z = np.array(z)

        dz = 1.0 - z * z

        if self.check_validity:
            validate_scalar_vector(input_values)

        if not return_as_matrix:
            return dz
        else:
            return np.diagflat(dz)

    def back_propagation(self, gradient, x, activation_output=True):
        super().back_propagation(gradient, x, activation_output)
        z = x
        if not activation_output:
            z = self(x)

        dz = 1.0 - z * z
        return gradient * dz


class ReLU(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return self._relu(x)

    @staticmethod
    def _relu(x):
        """
        Compute the Rectified Linear Unit (ReLU) function element-wise.

        :param x: Input array.
        :type x: array-like

        :return: Output array after applying ReLU function element-wise.
        :rtype: ndarray
        """
        return np.maximum(0, x)

    def derivative(self, input_values, return_as_matrix=False, **kwargs):
        z = input_values
        if isinstance(z, numbers.Number):
            z = np.array(z)

        dz = (z > 0) * 1  # np.where(z > 0, 1.0, 0.0)  # Derivative of ReLU

        if self.check_validity:
            validate_scalar_vector(input_values)

        if not return_as_matrix:
            return dz
        else:
            return np.diagflat(dz)

    def back_propagation(self, gradient, x, **kwarg):
        super().back_propagation(gradient, x, **kwarg)
        dz = (x > 0) * 1
        return gradient * dz


class ReLU6(Activation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, x):
        return self._relu6(x)

    @staticmethod
    def _relu6(x):
        """
        Compute the ReLU6 function element-wise.

        :param x: Input array.
        :type x: array-like

        :return: Output array after applying ReLU6 function element-wise.
        :rtype: ndarray
        """
        _t = np.minimum(np.maximum(0, x), 6)

        if isinstance(x, np.ndarray) and np.ndim(_t) == 0:
            _t = np.array(_t)
        return _t

    def derivative(self, input_values, return_as_matrix=False, **kwargs):
        z = input_values

        if isinstance(z, numbers.Number):
            z = np.array(z)

        dz = ((z > 0) & (z < 6)) * 1  # np.where((z > 0) & (z <= 6), 1, 0)  # Derivative of ReLU6
        if isinstance(input_values, np.ndarray) and np.ndim(z) == 0:
            return np.array(dz)

        if self.check_validity:
            validate_scalar_vector(input_values)

        if not return_as_matrix:
            return dz
        else:
            return np.diagflat(dz)

    def back_propagation(self, gradient, x, **kwarg):
        super().back_propagation(gradient, x, **kwarg)
        dz = ((x > 0) & (x < 6)) * 1
        return gradient * dz


class LeakyReLU(Activation):
    def __init__(self, *, alpha: float = 0.01, **kwargs):
        """
        Initialize the LeakyReLU activation function.

        :param alpha: Slope of the negative part, typically a small positive value.
        :type alpha: float
        """
        super().__init__(**kwargs)
        self.alpha = alpha

    def __call__(self, x):
        return self._leaky_relu(x)

    def _leaky_relu(self, x):
        """
        Compute the Leaky ReLU function element-wise.

        :param x: Input array.
        :type x: array-like

        :return: Output array after applying Leaky ReLU function element-wise.
        :rtype: ndarray
        """
        # if isinstance(x, numbers.Number):
        #     return x if x > 0 else self.alpha * x

        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, input_values, return_as_matrix=False, **kwargs):
        z = input_values
        if isinstance(z, numbers.Number):
            z = np.array(z)

        dz = np.where(z > 0, 1.0, self.alpha)  # Derivative of Leaky ReLU
        if self.check_validity:
            validate_scalar_vector(input_values)

        if not return_as_matrix:
            return dz
        else:
            return np.diagflat(dz)

    def back_propagation(self, gradient, x, **kwarg):
        super().back_propagation(gradient, x, **kwarg)
        dz = np.where(x > 0, 1.0, self.alpha)
        return gradient * dz


class Softmax(Activation):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def __call__(self, x):
        return self._softmax(x, axis=self.axis)

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

    def derivative(self, input_values, activation_output=True, return_as_matrix=False):
        if not activation_output:
            z = self(input_values)
        else:
            z = input_values       

        # Reshape z if it's 1D to 2D with a single batch
        if np.ndim(z) == 1:
            # If z is a 1D array, reshape it to a 2D array with a single batch
            z = np.expand_dims(z, axis=0)

        # Compute the Jacobian matrix of Softmax
        m, n = z.shape
        iden = np.eye(n)
        t1 = np.zeros((m, n, n), dtype=np.float64)
        t2 = np.zeros((m, n, n), dtype=np.float64)
        t1 = np.einsum('ij,jk->ijk', z, iden)
        t2 = np.einsum('ij,ik->ijk', z, z)
        return t1 - t2

    def back_propagation(self, gradient, x, activation_output=True):
        super().back_propagation(gradient, x, activation_output)
        
        g = gradient
        if np.ndim(gradient) == 1:
            # reshape it to a 2D array with a single batch
            g = np.expand_dims(gradient, axis=0)

        dz = self.derivative(x, activation_output=activation_output)
        return multiply_2D_with_3D(g, dz)
