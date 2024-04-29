import numpy as np
from miniML.initializers.Initializer import Initializer


class HeNormal(Initializer, object):

    def __call__(self, shape: tuple):
        super().__call__(shape)

        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
