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

### Optimizers
-  SGD
- Momentum
- RMSProp
- Adam
