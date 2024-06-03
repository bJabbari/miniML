# miniML

miniML is a lightweight and easy-to-use machine learning library designed for educational purposes and simple machine learning tasks. Inspired by [TensorFlow](https://tensorflow.org/), miniML provides a clean and intuitive API to build, train, and evaluate neural networks. It supports a variety of activation functions, initializers, layers, loss functions, and optimizers. Currently, it only supports the sequential model architecture and relies on CPU computation.

For those interested in learning the fundamentals of machine learning, miniML offers a straightforward implementation of many features found in TensorFlow using plain Python and [NumPy](https://numpy.org/). miniML adheres to TensorFlow patterns, employing practices like utilizing row vectors and maintaining consistent coefficients within loss functions and optimizers. One of the subtle differences from TensorFlow lies in the metrics, where miniML support various input/output types like one-hot encoded vectors, multi-label, multi-class classification, class labels, and probabilities, similar to [scikit-learn](https://scikit-learn.org/)'s syntax. This minimizes the need for input/output reshaping or conversion.

[//]: # (## Installation)

[//]: # ()
[//]: # (To install miniML, you can use pip:)

[//]: # (```bash)

[//]: # (pip install miniML)

[//]: # (```)

## Features

### Activation Functions
- **Linear**
- **Sigmoid**
- **Tanh**
- **ReLU**
- **ReLU6**
- **LeakyReLU**
- **Softmax**

### Initializers
- **GlorotNormal**
- **HeNormal**
- **Zeros**

### Loss Functions
- **MeanSquaredError**
- **MeanAbsoluteError**
- **BinaryCrossentropy**
- **CategoricalCrossentropy**

### Optimizers
- **SGD**
- **Adam**

### Layers
- **Input**
- **Dense** (supports weight regularizer)

## Quick Start

Here's a simple example to get you started with miniML:

```python
import miniML

# Define model architecture
model = miniML.models.Sequential()
model.add(miniML.layers.InputLayer(shape=(2,)))
model.add(miniML.layers.Dense(units=4, activation='tanh'))
model.add(miniML.layers.Dense(units=4, activation='relu'))
model.add(miniML.layers.Dense(units=1, activation='linear'))

# Compile the model
optimizer = miniML.optimizers.Adam(learning_rate=0.001)
loss_function = miniML.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss_function)

# Train the model
x_train, y_train = ...  # your training data
x_test, y_test = ...    # your validation data

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32, verbose=1)

# Evaluate the model
val_loss = model.evaluate(x_test, y_test, verbose=1)
print(f'Validation Loss: {val_loss}')
```

## Documentation

### Sequential Model

The `Sequential` model is a linear stack of layers. You can create a `Sequential` model by passing a list of layers to the constructor or using the `add` method.

```python
model = miniML.models.Sequential()
model.add(miniML.layers.InputLayer(shape=(2,)))
model.add(miniML.layers.Dense(units=4, activation='tanh'))
model.add(miniML.layers.Dense(units=4, activation='relu'))
model.add(miniML.layers.Dense(units=1, activation='linear'))
```

### Dense Layer
The `Dense` layer is a fully connected layer that supports weight regularizers. Here is an example of how to use the `Dense` layer:
```python
from miniML.layers import Dense
from miniML.initializers import GlorotNormal, Zeros
from miniML.regularizers import L2

layer = Dense(
    units=64,
    activation='relu',
    shape=(None, 128),
    weight_initializer=GlorotNormal(),
    bias_initializer=Zeros(),
    weight_regularize=L2(0.01),
    bias_regularize=None
)
model.add(layer)
```
### Compiling the Model
Before training a model, you need to configure the learning process, which is done via the `compile` method. It requires specifying a loss function and an optimizer.
```python
optimizer = miniML.optimizers.Adam(learning_rate=0.001)
loss_function = miniML.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss_function)
```
After compilation or the first call of the model, all layers' weights and biases are initialized.
### Model Summary
To view the summary of the model, use the `summary` method. This will show the output shape, number of parameters, and the number of trainable parameters for each layer.
```
model.summary()

Layer (type)                  Output Shape     Param #         Trainable #
===========================================================================
InputLayer_0 (InputLayer)      (None, 4)       None            None
Dense_0 (Dense)                (None, 4)       20              20
Dense_1 (Dense)                (None, 4)       20              20
Dense_2 (Dense)                (None, 6)       30              30
===========================================================================
Total params: 70
Trainable params: 70
Non-trainable params: 0
```
### Training the Model
To train a model, use the `fit` method. You need to pass the training data, number of epochs, batch size, and optionally, validation data.
```python
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32, verbose=1)
```

### Evaluating the Model
To evaluate the model's performance on test data, use the `evaluate` method.
```python
val_loss = model.evaluate(x_test, y_test, verbose=1)
print(f'Validation Loss: {val_loss}')
```

### Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue on GitHub.

### License
miniML is released under the MIT License.