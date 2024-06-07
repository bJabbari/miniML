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

    def compile(self, loss='mse', optimizer='adam' , metrics=None):
        self.loss_function = miniML.losses.get(loss)
        self.optimizer = miniML.optimizers.get(optimizer)
        self.metrics = None
        if metrics is not None:
            self.metrics = []
            for metric in metrics:
                self.metrics.append(miniML.metrics.get(metric))
        if not self._is_build:
            self.build()

    def fit(self, x, y,
            batch_size=None,
            epochs=1,
            verbose=1,
            validation_data=None,
            shuffle=True,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            # validation_freq=1,
            compute_epoch_loss=False
            ):
        if not self._is_build:
            self.build()

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError('X and Y should be Numpy array')
        for epoch in range(initial_epoch, epochs):
            if verbose:
                epoch_start_time = time.perf_counter()
                print(f'Epoch {epoch + 1}/{epochs}')
            seen_samples = 0
            x_batches, y_batches = self.shuffle_and_batch(x, y, batch_size=batch_size, shuffle=shuffle)
            steps_per_epoch = steps_per_epoch if steps_per_epoch else len(x_batches)
            loss_total = 0
            if self.metrics:
                for metric in self.metrics:
                    metric.reset_state()

            for i, (xb, yb) in enumerate(zip(x_batches, y_batches)):
                if i >= steps_per_epoch:
                    break
                if i == 0:
                    grads_and_params, loss, loss_regularization = self.one_train_step(xb, yb, verbose=0)
                self.optimizer.update(grads_and_params, in_place=True)
                # Compute the new value of loss after updating weights, for the current batch
                grads_and_params, loss, loss_regularization = self.one_train_step(xb, yb, verbose)

                seen_samples += xb.shape[0]
                # Calculate step loss
                loss_total += (loss + loss_regularization) * xb.shape[0]
                if verbose == 1:
                    step_loss = loss_total / seen_samples
                    # Calculate step time
                    step_end_time = time.perf_counter()
                    total_step_time = step_end_time - epoch_start_time
                    avg_step_time = total_step_time / (i + 1.0)
                    # Update progress bar
                    progress_str = self.progress_bar(i, steps_per_epoch)
                    print(
                        f'\r{progress_str}'
                        f' - {total_step_time:.1f}s {avg_step_time * 1000:.2f}ms/step - Loss: {float_formatter(step_loss)}',
                        end='')
                    if self.metrics:
                        for metric in self.metrics:
                            print(f' - {metric.name}: {float_formatter(metric.result())}', end='')

            # end of one epoch
            if verbose:
                epoch_time = time.perf_counter() - epoch_start_time
                avg_step_time = epoch_time / steps_per_epoch
            step_loss = loss_total / seen_samples
            epoch_loss = None
            val_loss = None
            train_metrics= dict()
            val_metrics = dict()
            if compute_epoch_loss:
                # Calculate the value of loss for the  all batches
                epoch_loss = self.evaluate(x, y, batch_size, verbose=0)
                if self.metrics:
                    for metric in self.metrics:
                        train_metrics.update({metric.name: metric.result()})

            if validation_data:
                val_loss = self.evaluate(validation_data[0], validation_data[1],
                                         batch_size=validation_batch_size,
                                         steps=validation_steps,
                                         verbose=0)
                if self.metrics:
                    for metric in self.metrics:
                        val_metrics.update({metric.name: metric.result()})





            if verbose in [1, 2]:
                epoch_end_time = time.perf_counter()
                epoch_time = epoch_end_time - epoch_start_time
                print(f'\r{steps_per_epoch}/{steps_per_epoch}:'
                      f' - {epoch_time:.1f}s/epoch - {avg_step_time * 1000:.2f}ms/step', end='')
                print(f' - Loss: {float_formatter(step_loss)}', end='')
                if epoch_loss:
                    print(f' - train-Loss: {float_formatter(epoch_loss)}', end='')
                if train_metrics:
                    for metric_name, metric_value in train_metrics.items():
                        print(f' - {metric_name}: {float_formatter(metric_value)}', end='')
                elif self.metrics:
                    for metric in self.metrics:
                        print(f' - {metric.name}: {float_formatter(metric.result())}', end='')
                if val_loss:
                    print(f' - val-Loss: {float_formatter(val_loss)}', end='')
                    if val_metrics:
                        for metric_name, metric_value in train_metrics.items():
                            print(f' - val_{metric_name}: {float_formatter(metric_value)}', end='')
                print()

    def one_train_step(self, xb, yb, verbose):
        y_pred = self.__call__(xb, training=True)
        loss = self.loss_function(y_true=yb, y_pred=y_pred)
        loss_regularization = 0
        delta = self.loss_function.gradient()
        grads_and_params = []
        for layer in self._layers[-1:0:-1]:
            loss_regularization += layer.loss
            delta, gradients_variables = layer.backward(delta)
            grads_and_params.extend(gradients_variables)
        if self.metrics and verbose==1:
            for metric in self.metrics:
                metric.update_state(y_true=yb, y_pred=y_pred)
        return grads_and_params, loss, loss_regularization

    def evaluate(self, x, y, batch_size=None, verbose=1, steps=None):
        x_batches, y_batches = self.shuffle_and_batch(x, y, batch_size=batch_size, shuffle=False)
        steps = steps if steps else len(x_batches)
        loss_total = 0
        seen_samples = 0
        if verbose:
            epoch_start_time = time.perf_counter()

        if self.metrics:
            for metric in self.metrics:
                metric.reset_state()


        for i, (xb, yb) in enumerate(zip(x_batches, y_batches)):
            if i >= steps:
                break
            y_pred = self.__call__(xb, training=False)
            loss = self.loss_function(y_true=yb, y_pred=y_pred) * xb.shape[0]
            loss_total += loss
            seen_samples += xb.shape[0]
            if self.metrics:
                for metric in self.metrics:
                    metric.update_state(y_true=yb, y_pred=y_pred)

            if verbose == 1:
                step_loss = loss_total / seen_samples
                # Calculate step time
                step_end_time = time.perf_counter()
                total_step_time = step_end_time - epoch_start_time
                avg_step_time = total_step_time / (i + 1.0)
                # Update progress bar
                progress_str = self.progress_bar(i, steps)
                print(
                    f'\r{progress_str}'
                    f' - {total_step_time:.1f}s {avg_step_time * 1000:.2f}ms/step - Loss: {float_formatter(step_loss)}',
                    end='')
                if self.metrics:
                    for metric in self.metrics:
                        print(f' - {metric.name}: {float_formatter(metric.result())}', end='')

        if verbose:
            total_steps_time = time.perf_counter() - epoch_start_time
            avg_step_time = total_steps_time / steps
        loss_total = loss_total / seen_samples
        loss_regularization = 0
        for layer in self._layers[1:]:
            loss_regularization += layer.loss
        loss_total += loss_regularization
        if verbose in [1, 2]:
            epoch_end_time = time.perf_counter()
            total_time = epoch_end_time - epoch_start_time
            print(f'\r{steps}/{steps}:'
                  f' - {total_time:.1f}s {avg_step_time * 1000:.2f}ms/step - Loss: {float_formatter(loss_total)}'
                  , end='')
            if self.metrics:
                for metric in self.metrics:
                    print(f' - {metric.name}: {float_formatter(metric.result())}', end='')
            print()

        return loss_total

    def predict(self, x, *args, **kwargs):
        return super().predict(x, *args, **kwargs)

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
            x_batches = [x_shuffled]
            y_batches = [y_shuffled]

        return x_batches, y_batches

    @staticmethod
    def progress_bar(step, total_steps, bar_length=30):
        progress = (step + 1) / total_steps
        block = int(round(bar_length * progress))
        bar = '#' * block + '-' * (bar_length - block)
        progress_str = f'{step + 1}/{total_steps}: |{bar}| {str(int(progress * 100)).rjust(3)}%'
        return progress_str
