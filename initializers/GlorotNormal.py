import numpy as np
from miniML.initializers.Initializer import Initializer


class GlorotNormal(Initializer):
    """
    also called Xavier normal initializer

    ========
    Example:
    ========
    initializer = GlorotNormal(seed=11)
    print(initializer(shape=(2, 3)))
    """

    def __call__(self, shape: tuple, dtype: np.dtype = np.float32):
        super().__call__(shape)

        return np.random.randn(*shape, ) * np.sqrt(2 / np.sum(shape))