import numpy as np

from miniML.layers import Layer


class InputLayer(Layer):
    def __init__(self, shape: tuple = None, batch_size=None, **kwargs):
        super().__init__(trainable=False, batch_size=batch_size, **kwargs)
        self.input_shape = shape
        self.output_shape = self.input_shape

    @property
    def input_shape(self):
        return self._input_shape

    @input_shape.setter
    def input_shape(self, value):
        if value is None:
            raise ValueError('you can not define an input layer without providing the input shape.')
        elif isinstance(value, int) and value > 0:
            self._input_shape = (self.batch_size, value)
        elif isinstance(value, (tuple, list)):
            if len(value) == 0:
                raise ValueError(f"input shape can not be an empty sequence.")
            if not all(isinstance(v, int) and v > 0 for v in value):
                raise ValueError(f'All elements of input_shape must be positive integers. Received: {value}')
            self._input_shape = (self.batch_size,) + tuple(value)
        else:
            raise TypeError('input_shape must be an integer, a tuple, or a list of integers.')

    def __call__(self, input_values: np.ndarray,*args, **kwargs):
        return input_values
