import numpy as np
from typing import Union
from miniML.layers import Layer
from miniML.initializers import GlorotNormal
from miniML.initializers import Zeros
from miniML.helper.validation import validate_positive_integer
import miniML.activations as activations
import miniML.regularizers as regularizers


class Dense(Layer):
    def __init__(self, units: int,
                 activation: str = None,
                 input_shape: Union[int, tuple, np.ndarray] = None,
                 weight_initializer=GlorotNormal(),
                 bias_initializer=Zeros(),
                 weight_regularize=None,
                 bias_regularize=None,
                 seed=None,
                 **kwargs
                 ):

        super().__init__(**kwargs)

        self.units = units
        self.input_shape = input_shape
        self.output_shape = None

        self.activations_function = activations.get(activation)
        self._weight_initializer = weight_initializer
        self._bias_initializer = bias_initializer
        self._weight_regularize = regularizers.get(weight_regularize)
        self._bias_regularize = regularizers.get(bias_regularize)
        self._is_called = False
        self._seed = seed

        self._n_in = None
        self._n_out = None

        self.weight = None
        self.bias = None
        self.input = None
        self.z = None  # pre-activation
        self.output = None

        self.loss = 0.0

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        validate_positive_integer(value, 'units')
        self._units = value

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        if value is None:
            self._input_shape = None
        elif isinstance(value, int) and value > 0:
            self._input_shape = (1, value)
        elif hasattr(value, '__len__'):
            if len(value) > 2:
                raise ValueError(
                    f"for dense layer, input dimension can not be more than 2. input shape received is: {len(value)}")
            if isinstance(value, tuple):
                if all(isinstance(v, int) and v > 0 for v in value):
                    self._input_shape = value
                else:
                    raise ValueError(f'All elements of input_shape tuple must be positive integers. Received: {value}')
            elif isinstance(value, np.ndarray):
                if value.dtype == np.integer and np.all(value > 0):
                    self._input_shape = tuple(value.tolist())
                else:
                    raise ValueError('All elements of input_shape NumPy array must be positive integers. '
                                     f'Received: {value}')
        else:
            raise TypeError('input_shape must be an integer, a tuple, or a NumPy array of integers.')

    def build(self, input_shape):
        self.input_shape = input_shape

        self._n_in = self.input_shape[-1]
        self._n_out = self.units

        self.output_shape = (*self._input_shape[:-1], self._n_out)

        self.bias = self._bias_initializer(shape=(1, self._n_out))
        self.weight = self._weight_initializer(shape=(self._n_in, self._n_out))

    def __call__(self, input_values: np.ndarray):
        if self.input_shape and self.input_shape != np.shape(input_values):
            raise ValueError(f'this layer expects input with the shape'
                             f' of {self.input_shape} but received inputs with the shape '
                             f' of {np.shape(input_values)}')

        if not self._is_called:
            self.build(np.shape(input_values))
            self._is_called = True

        self.input = input_values
        # always convert input vector to a 2D array, to do operation like matrix transpose correctly
        if self.input.ndim == 0:
            self.input = np.expand_dims(self.input, axis=0)
        # pre - activation value
        self.z = np.matmul(input_values, self.weight) + self.bias
        self.output = self.activations_function(self.z)

        return self.output

    def update_weights(self, gradient: np.ndarray, learning_rate=1.0e-3):
        # input gradient.shape should be m * self.units; where m is number of batches
        n_batches = gradient.shape[0]
        if gradient.ndim == 0:
            n_batches = 1

        # back propagated gradient through activation function
        grad = self.activations_function.back_propagation(gradient,
                                                          self.output,
                                                          activation_output=True)

        # update weights
        delta_weight = np.matmul(self.input.T, grad) / float(n_batches)
        if self._weight_regularize is not None:
            delta_weight += self._weight_regularize.backward(self.weight)
        self.weight = self.weight - learning_rate * delta_weight
        # update biases
        delta_bias = np.mean(grad, axis=0, keepdims=True)
        if self._bias_regularize is not None:
            delta_bias += self._bias_regularize.backward(self.bias)
        self.bias = self.bias - learning_rate * delta_bias

        # Regularization loss
        if self._weight_regularize is not None:
            self.loss = self._weight_regularize(self.weight)
        if self._bias_regularize is not None:
            self.loss = self._bias_regularize(self.bias)

        # back propagated gradient to previous layer
        propagated_grad = np.matmul(grad, self.weight.T)  # grad.shape is m * self.units

        return propagated_grad