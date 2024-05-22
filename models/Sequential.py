import time

import numpy as np

import miniML
from helper.helper import float_formatter
from miniML.layers import Layer, InputLayer
from miniML.models import Model


class Sequential(Model):

    def __init__(self, layers: list[Layer] = None):
        super().__init__()
        if layers is not None:
            for layer in layers:
                self.add(layer)

    def add(self, layer: Layer) -> None:
        if not self._layers:
            if not isinstance(layer, InputLayer):
                if hasattr(layer, 'input_shape') and layer.input_shape is not None:
                    self.add(InputLayer(layer.input_shape[1:], batch_size=layer.batch_size))
                else:
                    raise ValueError('the first layer in sequential model must be an InputLayer or have None shape '
                                     'parameter')
        elif self._layers and isinstance(layer, InputLayer):
            raise ValueError('an InputLayer must be added as the first layer in sequential model')

        if isinstance(layer, Layer):
            if hasattr(layer, 'name') and layer.name is not None:
                if not self._is_layer_name_unique(layer):
                    raise ValueError(f"All layers added to a Sequential model "
                                     f"should have unique names. Name '{layer.name}' is already "
                                     "the name of a layer in this model. Update the `name` argument "
                                     "to pass a unique name.")
            if layer.name is None:
                self._name_layer(layer)

            self._layers.append(layer)
        else:
            raise ValueError('layer was not recognized. layer must be a subclass of Layer.')

    def pop(self):
        if len(self._layers) > 0:
            self._layers.pop()

    def __call__(self, x, *args, **kwargs):
        for layer in self._layers:
            x = layer(x, *args, **kwargs)
        return x

    def compile(self, loss='mse', optimizer='adam'):
        self.loss_function = miniML.losses.get(loss)
        self.optimizer = miniML.optimizers.get(optimizer)

    def fit(self, x, y,
            batch_size=None,
            epochs=1,
            # verbose=True,
            # validation_data=None,
            shuffle=True,
            initial_epoch=0,
            # steps_per_epoch=None,
            # validation_steps=None,
            # validation_batch_size=None,
            # validation_freq=1,
            ):
        if not self._is_build:
            self.build()

        is_verbose = True  # Todo
        bar_length = 30
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError('X and Y should be Numpy array')
        # total_samples = X.shape[0]
        for epoch in range(initial_epoch, epochs):
            epoch_start_time = time.perf_counter()
            print(f'Epoch {epoch + 1}/{epochs}')
            seen_samples = 0
            x_batches, y_batches = self.shuffle_and_batch(x, y, batch_size=batch_size, shuffle=shuffle)
            num_batches = len(x_batches)
            loss_total = 0

            for i, (xb, yb) in enumerate(zip(x_batches, y_batches)):
                y_pred = self.__call__(xb, training=True)
                loss = self.loss_function(y_true=yb, y_pred=y_pred)
                loss_regularization = 0
                delta = self.loss_function.gradient()
                grads_and_params = []
                for layer in self._layers[-1:0:-1]:
                    loss_regularization += layer.loss
                    delta, gradients_variables = layer.backward(delta)
                    grads_and_params.extend(gradients_variables)

                self.optimizer.update(grads_and_params, in_place=True)

                if is_verbose:
                    seen_samples += xb.shape[0]
                    # Calculate step loss
                    loss_total += (loss + loss_regularization) * xb.shape[0]
                    step_loss = loss_total / seen_samples

                    # Calculate step time
                    epoch_end_time = time.perf_counter()
                    epoch_time = epoch_end_time - epoch_start_time
                    step_time = epoch_time / (i + 1.0)
                    # Update progress bar
                    progress = (i + 1) / num_batches

                    block = int(round(bar_length * progress))
                    bar = '#' * block + '-' * (bar_length - block)
                    print(
                        f'\r{i + 1}/{num_batches}: |{bar}| {str(int(progress * 100)).rjust(3)}%'
                        f' - {epoch_time:.1f}s {step_time * 1000:.2f}ms/step - Loss: {float_formatter(step_loss)}',
                        end='')

            # end of one epoch
            epoch_end_time = time.perf_counter()
            epoch_time = epoch_end_time - epoch_start_time
            step_time = epoch_time / num_batches
            # Calculate loss
            epoch_loss = self.evaluate(x, y, batch_size, verbose=False)
            print(
                f'\r{num_batches}/{num_batches}:'
                f' - {epoch_time:.1f}s {step_time * 1000:.2f}ms/step - Loss: {float_formatter(epoch_loss)}')
            # Calculate metrics

    def evaluate(self, x, y, batch_size=None, verbose=True):
        x_batches, y_batches = self.shuffle_and_batch(x, y, batch_size=batch_size, shuffle=False)
        total_loss = 0

        total_samples = x.shape[0]
        for xb, yb in zip(x_batches, y_batches):
            y_pred = self.__call__(xb, training=False)
            loss = self.loss_function(y_true=yb, y_pred=y_pred) * xb.shape[0]
            total_loss += loss
        total_loss = total_loss / total_samples
        loss_regularization = 0
        for layer in self._layers[1:]:
            loss_regularization += layer.loss
        total_loss += loss_regularization
        if verbose:
            print(f'Loss: {total_loss}')
        return total_loss

    def predict(self, x, *args, **kwargs):
        super().predict(x, args, kwargs)

    def _name_layer(self, layer: Layer) -> None:
        if 'name' not in layer.__dict__ or not layer.name:
            base_name = layer.__class__.__name__
            index = sum(1 for l in self._layers if l.__class__.__name__ == base_name)
            layer.name = f"{base_name}_{index}"

    def _is_layer_name_unique(self, layer: Layer) -> bool:
        for _layer in self._layers:
            if _layer.name == layer.name and _layer is not layer:
                return False
        return True

    @staticmethod
    def shuffle_and_batch(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True, ) -> object:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            print()
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input and output arrays must have the same number of samples")

        # Shuffle the data
        if shuffle:
            permutation = np.random.permutation(len(x))
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]
        else:
            x_shuffled = x
            y_shuffled = y

        # Create batches
        if batch_size is not None:
            num_batches = len(x) // batch_size
            x_batches = np.array_split(x_shuffled, num_batches)
            y_batches = np.array_split(y_shuffled, num_batches)
        else:
            x_batches = x_shuffled
            y_batches = y_shuffled

        return x_batches, y_batches
