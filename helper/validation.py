import numbers
import numpy as np


def validate_positive_integer(value, name):
    if value is None:
        raise ValueError(f'{name} cannot be None.')
    elif not isinstance(value, int):
        raise TypeError(f'{name} must be an integer.')
    elif value <= 0:
        raise ValueError(f'{name} must be a positive integer.')


def validate_scalar_vector(value):
    if isinstance(value, numbers.Number):
        return
    if not isinstance(value, np.ndarray):
        raise TypeError(f'input is not an array')
    else:
        if np.ndim(value) in [0, 1]:
            return
        elif np.ndim(value) == 2:
            if any(z == 1 for z in np.shape(value)):
                return
            else:
                raise ValueError(f'input must be a vector, not {np.shape(value)}')
        else:
            raise ValueError(f'input must be a vector, not {np.shape(value)}')


def validate_non_negative_float(value, name):
    if value is None:
        raise ValueError(f'{name} cannot be None.')
    if isinstance(value, numbers.Real):
        if value < 0:
            raise ValueError(f'{name} must be a non-negative float. Received: {name}={value}')
        else:
            return float(value)
    else:
        raise TypeError(f'{name} must be a float number, not {type(value)}')
