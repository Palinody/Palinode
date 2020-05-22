# Palinode

> C++ repository used for educational purposes in deep learning. It will potentially contain multiple neural network layer architectures as well as cpu-gpu multi-threading computation depending of the motivation and time I have to implement them.

### Compilation
Serial
```sh
$ g++-9 -std=c++17 -O3 main.cpp
$ ./a.out
```
OpenMP
```sh
$ g++-9 -std=c++17 -O3 -fopenmp main.cpp
$ ./a.out
```
### Layers
- Fully connected hidden layer
- Fully connected output layer

### Activation functions
- logit (hidden | output)
- relu (hidden)
- sigmoid (hidden | output)
- swish (hidden)
- tanh (hidden | output)
- softmax (output)

### Cost|Loss functions
- Mean Squared Error (MSE)
- Binary Cross Entropy (bCE) [stable]
- Cross Entropy (CE)

### Optimizers
- SGD
- Momentum
- NAG
- Adagrad
- RMSProp
- Adam
