from abc import ABC, abstractmethod
from miniML.layers import InputLayer


class Model(ABC):

    def __init__(self):
        self._layers = []
        self.training = False
        self._is_build = False
        self.loss_function = None
        self.optimizer = None
        self.metrics = None

    @property
    def layers(self):
        return self._layers

    def get_layer(self, layer_name=None, layer_index=None):
        if layer_name is None and layer_index is None:
            raise ValueError(f'either layer_name or layer_index must be provided')
        elif layer_name is not None and layer_index is not None:
            raise ValueError(f'please only provide the layer_name or layer_index but not both. '
                             f'Received {layer_name}, {layer_index}')
        elif layer_index is not None:
            if 0 <= layer_index < len(self._layers):
                return self._layers[layer_index]
            else:
                raise ValueError(f'layer_index must be between 0 and {len(self._layers)}'
                                 f' but got {layer_index}')
        elif layer_name is not None:
            for layer in self._layers:
                if layer_name == layer.name:
                    return layer
            raise ValueError(f'layer {layer_name} does not exist. provide a valid layer name')

        raise ValueError(f'provide a valid layer name or index')

    def build(self):
        layer = self._layers[0]
        input_shape = layer.output_shape
        for layer in self._layers[1:]:
            layer.build(input_shape)
            input_shape = layer.output_shape
        self._is_build = True

    def summary(self):
        if not self._is_build:
            self.build()

        widths = [25, 20, 15, 15]
        length = sum(widths)
        total_params = 0
        total_trainable_params = 0
        print('Layer (type)'.ljust(widths[0]),
              'Output Shape'.center(widths[1]),
              'Param #'.ljust(widths[2]),
              'Trainable #'.ljust(widths[3])
              )
        print('=' * length)

        for layer in self._layers:
            layer_name = f'{layer.name} ({layer.__class__.__name__})'.ljust(widths[0])
            output_shape = str(layer.output_shape).center(widths[1])
            param = str(None).ljust(widths[2])
            trainable_param = str(None).ljust(widths[3])
            if not isinstance(layer, InputLayer):
                layer_params = layer.compute_number_of_parameters()
                param = f"{layer_params['total_params']:,}".ljust(widths[2])
                trainable_param = f"{layer_params['trainable_params']:,}".ljust(widths[3])
                total_params += layer_params['total_params']
                total_trainable_params += layer_params['trainable_params']
            print(layer_name, output_shape, param, trainable_param)

        print('=' * length)
        print(f'Total params: {total_params:,}')
        print(f'Trainable params: {total_trainable_params:,}')
        print(f'Non-trainable params:{(total_params - total_trainable_params):,}')

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        return self.__call__(x, *args, training=False, **kwargs)

    def __call__(self, *args, **kwargs):
        pass
