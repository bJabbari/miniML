from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):

    def __init__(self, **kwargs):
        if kwargs:
            self.name = kwargs.get('name', None)
            self.trainable = kwargs.get('trainable', True)


