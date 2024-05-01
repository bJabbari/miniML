

1. **Zero Initialization**: 

   This initializes all weights and biases to zero. it's generally not recommended because it leads to symmetric weights, which can cause all neurons in a layer to behave the same way, leading to reduced learning capacity.
   
   :bulb: Zero Initialization is particularly used for biases.

2. **Glorot Initialization (Xavier Initialization)**:
   
   This initializer sets the weights using a Gaussian or uniform distribution with zero mean and variance scaled according to the number of input and output neurons. 
   
   Uniform distribution:
   $$W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}, \frac{\sqrt{6}}{\sqrt{n_{\text{in}} + n_{\text{out}}}}\right)$$

   Normal distribution:
   $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

   Where  $n_{\text{in}}$ is the number of input neurons and  $n_{\text{out}}$ is the number of output neurons.

   :bulb: Glorot initialization is particularly effective for sigmoid and tanh activation functions

3. **He Initialization**:

   Named after the researcher who introduced it, He initialization initializes the weights using a Gaussian or uniform distribution with zero mean and variance scaled according to the number of input neurons.

   For weights \( W \) sampled from a uniform or normal distribution:

   Uniform distribution:
   $$W \sim U\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)$$

   Normal distribution:
   $$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

   Where  $n_{\text{in}}$ is the number of input neurons.

   :bulb: He initialization is well-suited for activation functions like ReLU (Rectified Linear Unit) and its variants
