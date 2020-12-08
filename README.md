# Palinode

> C++ repository used for educational purposes in deep learning. It will potentially contain multiple neural network layer architectures as well as cpu-gpu multi-threading computation depending of the motivation and time I have to implement them.

### TODO

fix: warnings when compiling without -fopenmp flag
fix: warnings (clang++-10) loop not vectorized when using openmp

### Compilation
Serial
```sh
$ g++-10 -std=c++20 -Wall -O3 -march=native -o main main.cpp
$ ./a.out
```
OpenMP
```sh
$ g++-10 -std=c++20 -Wall -O3 -march=native -fopenmp -o main main.cpp
$ ./a.out
```
You can also compile directly with Makefile
```sh
$ make
$ ./main
```
The program has been compiled/tested using g++-10
If the program doesn't compile on your machine 
this [link](https://linuxconfig.org/how-to-switch-between-multiple-gcc-and-g-compiler-versions-on-ubuntu-20-04-lts-focal-fossa) might help.
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
