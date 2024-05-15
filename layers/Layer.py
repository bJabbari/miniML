from abc import ABC


class Layer(ABC):

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.trainable = kwargs.get('trainable', True)
        self.batch_size = kwargs.get('batch_size', None)



