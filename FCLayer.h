#pragma once

#include "headers/Matrix.h"
#include <algorithm>
#include <initializer_list>

/**
 * Hidden FC layer is preallocated with: 
 *      zeros by default
 *      a random distribution
 * rows: batch size
 * cols: number of hidden nodes
*/

/**
 * Dropout implementation
 * EXAMPLE:
 *      H1 = H0.dot(W)
 * WITH:
 *      H0: 2X5 (prev layer)
 *      H1: 2x3 (curr layer)
 *      ==> W = 5x3
 * H0:
 * [[., ., ., X, .],
 *  [., ., ., X, .]]
 * W:
 * [[., ., .],
 *  [., ., .],
 *  [., ., .],
 *  [X, X, X],
 *  [., ., .]]
 * H1:
 * [[., ., .],
 *  [., ., .]]
 * 
 * lets: dropout = 0.1
 * generate vector of size 1xHO.getCols() 
 * with each elem drawn from uniform(0, 1)
 * mask = (unif_vect < dropout) ? 1 : 0
 * mask.where(1) -> gives indices to iterate over
*/

enum Activation { LOGIT, RELU, SWISH, SIGMOID, TANH, SOFTMAX };
enum OptimizerName { sgd, momentum, nag, adagrad, rmsprop, adam };
// OptimizerName
template<typename T>
class Optimizer{
public:
    Optimizer(T lr) : _lr{ lr }{}

    virtual void operator()(Matrix<T>& weights, const Matrix<T>& gradients) = 0;
protected:
    T _lr;
};

template<typename T>
class SGD : public Optimizer<T>{
public:
    SGD(T lr) : Optimizer<T>(lr) {}

    void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
        weights -= gradients * this->_lr;
    }
};

template<typename T>
class Momentum : public Optimizer<T>{
public:
    Momentum(T lr, T damp, int rows, int cols) : 
        Optimizer<T>(lr),
        _damp{damp},
        _velocity{ std::make_unique<Matrix<T>>(rows, cols, 0) }{}

    void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
        (*_velocity) = (*_velocity) * _damp + gradients * this->_lr;
        weights -= (*_velocity);
    }
private:
    T _damp;
    std::unique_ptr<Matrix<T>> _velocity; // accumulator (velocity)
};

template<typename T>
void sqrt_mat(Matrix<T>& matrix){
    for(int i = 0; i < matrix.getRows(); ++i){
        for(int j = 0; j < matrix.getCols(); ++j){
            matrix(i, j) = std::sqrt(matrix(i, j));
        }
    }
}

template<typename T>
class Adagrad : public Optimizer<T>{
public:
    Adagrad(T lr, int rows, int cols) : 
        Optimizer<T>(lr),
        _running_sum{ std::make_unique<Matrix<T>>(rows, cols, 0) }{}

    void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
        
        (*_running_sum) = (*_running_sum) + (gradients*gradients);
        Matrix<T> sqrt_tmp(_running_sum->getRows(), _running_sum->getCols(), 0);
        func2D::sqrt(sqrt_tmp, *_running_sum);
        weights -= gradients / sqrt_tmp * this->_lr;
    }
protected:
    std::unique_ptr<Matrix<T>> _running_sum; // accumulator (sum of grads)
};

template<typename T>
class RMSProp : public Optimizer<T>{
public:
    RMSProp(T lr, T decay, int rows, int cols) : 
    Optimizer<T>(lr), _decay{ decay },
    _running_sum{ std::make_unique<Matrix<T>>(rows, cols, 0) }{}
    
    void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
        
        (*_running_sum) = (*_running_sum) * _decay;
        T coeff = static_cast<T>(1) * _decay;
        (*_running_sum) += (gradients*gradients) * coeff;
        Matrix<T> sqrt_tmp(_running_sum->getRows(), _running_sum->getCols(), 0);
        func2D::sqrt(sqrt_tmp, *_running_sum);
        weights -=  gradients / sqrt_tmp * this->_lr;
    }
protected:
    T _decay;
    std::unique_ptr<Matrix<T>> _running_sum; // accumulator (sum of grads)
};

template<typename T>
class Adam : public Optimizer<T>{
public:
    Adam(T lr, T beta1, T beta2, int rows, int cols) : 
    Optimizer<T>(lr),
    _beta1{ beta1 }, _beta2{ beta2 },
    _beta1_t{ beta1 }, _beta2_t{ beta2 },
    _moment1{ std::make_unique<Matrix<T>>(rows, cols, 0) },
    _moment2{ std::make_unique<Matrix<T>>(rows, cols, 0) }{}

    void operator()(Matrix<T>& weights, const Matrix<T>& gradients){
        T one = static_cast<T>(1);
        (*_moment1) = ((*_moment1) * _beta1 + gradients * (one-_beta1)) /*/ (one-_beta1)*/;
        (*_moment2) = ((*_moment2) * _beta2 + gradients*gradients * (one-_beta2)) /*/ (one-_beta2)*/;
        Matrix<T> sqrt_tmp(_moment2->getRows(), _moment2->getCols(), 0);
        func2D::sqrt(sqrt_tmp, *_moment2);
        T alpha_t = this->_lr * std::sqrt(one-_beta2_t) / (one-_beta1_t);
        weights -= (*_moment1) / sqrt_tmp * alpha_t;

        _beta1_t *= _beta1;
        _beta2_t *= _beta2;
    }
private:
    T _beta1, _beta2;
    T _beta1_t, _beta2_t;
    std::unique_ptr<Matrix<T>> _moment1;
    std::unique_ptr<Matrix<T>> _moment2;
};

template<typename T>
class FCLayer{
public:
    FCLayer(int n_batch, int n_nodes, int n_prev_nodes, T value = 0, int num_threads = 1);

    // ** setters **
    void dropout(float drop_rate);

    // ** getters **
    const Matrix<T>& getWeights();
    const std::vector<int>& getDroppedIdx();

    // ** mem. management **
    void mallocGrad();
    void optimizer(OptimizerName op, std::initializer_list<double> args);
    void freeGrad();
    void reallocBatch(int batchSize);
    void freeOptimizer();

    // ** methods **
    const Matrix<T>& logit(const Matrix<T>& prev_layer);
    const Matrix<T>& activate(Activation activation);

    const Matrix<T>& delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next, const std::vector<int>& dropped_indices_next);
    void gradients(const Matrix<T>& prev_layer, const std::vector<int>& dropped_indices_next);
    void weights_update(const std::vector<int>& dropped_indices_next);

    void gradientCheck(const Matrix<T>& prev_layer, T epsilon);
private:
    std::unique_ptr<Matrix<T>> _layer;          // f(Z)
    std::unique_ptr<Matrix<T>> _logit;          // Z = X.W + b
    std::unique_ptr<Matrix<T>> _weights;        // weights matrix
    std::unique_ptr<Matrix<T>> _biases;         // bias matrix
    std::unique_ptr<Matrix<T>> _delta;          // layer derivative: (DJ)/(Dy_hat)
    std::unique_ptr<Matrix<T>> _weights_grad;   // weights gradients: (DJ)/(DW)
    std::unique_ptr<Matrix<T>> _biases_grad;    // biases gradient

    std::unique_ptr<Optimizer<T>> _optimizer_w;
    std::unique_ptr<Optimizer<T>> _optimizer_b;

    int _w_rows, _w_cols;   // number of rows/cols of weight matrix (_w_rows == _n_prev_nodes)
    int _n_prev_nodes;      // number of nodes of previous layer
    // NEW args
    int _batch;
    int _nodes;

    float _dropout = 0;     // dropout prob [0, 1]
    std::vector<int> _dropped_indices;

    Activation _activation = LOGIT;

    int _n_threads;
};
//////////////////////////////////////////////////////////////////////
//																	//
//							 IMPLEMENTATION	                        //
//						                                            //
//////////////////////////////////////////////////////////////////////

template<typename T>
FCLayer<T>::FCLayer(int n_batch, int n_nodes, int n_prev_nodes, T value, int num_threads) : 
    _batch{ n_batch }, _nodes{ n_nodes },
    _w_rows{ n_prev_nodes }, _w_cols{ n_nodes },
    _n_prev_nodes{ n_prev_nodes },
    _dropped_indices{ -1, n_prev_nodes }, _n_threads{ num_threads },
    _layer{ std::make_unique<Matrix<T>>(n_batch, n_nodes, 0, num_threads) },
    _logit{ std::make_unique<Matrix<T>>(n_batch, n_nodes, 0, num_threads) },
    _weights{ std::make_unique<Matrix<T>>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads) },
    _biases{ std::make_unique<Matrix<T>>(1, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads) }{
}
//////////////////////////////////////////////////////////////////////
							/// SETTERS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void FCLayer<T>::dropout(float drop_rate){
    _dropout = drop_rate;
    Matrix<float> mask(1, _n_prev_nodes, UNIFORM, 0, 1);
    mask < drop_rate; // set to 1 where mask < drop_rate
    // find the indices to skip
    auto indices_vect_temp = mask.where_row(0, 1);
    // will be the final vector wrapping the dropped idx around -1 and _cols
    // starting idx is actually -1 because we add + 1 for each starting idx to skip it
    std::vector<int> indices_vect_complete;
    indices_vect_complete.reserve(indices_vect_temp.size()+2); // +2 for start/end indices
    indices_vect_complete.push_back(-1);
    indices_vect_complete.insert(std::end(indices_vect_complete), std::begin(indices_vect_temp), std::end(indices_vect_temp));
    indices_vect_complete.push_back(_n_prev_nodes);

    _dropped_indices.swap(indices_vect_complete);
}

//////////////////////////////////////////////////////////////////////
							/// GETTERS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
const Matrix<T>& FCLayer<T>::getWeights(){
    return *_weights;
}

template<typename T>
const std::vector<int>& FCLayer<T>::getDroppedIdx(){
    return _dropped_indices;
}

//////////////////////////////////////////////////////////////////////
						/// MEM. MANAGEMENT ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void FCLayer<T>::mallocGrad(){
    _delta = std::make_unique<Matrix<T>>(_batch, _nodes, 0, _n_threads);
    _weights_grad = std::make_unique<Matrix<T>>(_w_rows, _w_cols, 0, _n_threads);
    _biases_grad = std::make_unique<Matrix<T>>(1, _w_cols, 0, _n_threads);
}

template<typename T>
void FCLayer<T>::optimizer(OptimizerName op, std::initializer_list<double> args){
    int n_args = args.size();
    std::initializer_list<double>::iterator it = args.begin();
    switch (op) {
		case sgd:{
            assert(n_args == 1);
            _optimizer_w = std::make_unique<SGD<T>>(*it);
            _optimizer_b = std::make_unique<SGD<T>>(*it);
			break;
		}
        case momentum:{
            assert(n_args == 2);
            _optimizer_w = std::make_unique<Momentum<T>>(*it, *(it+1), _w_rows, _w_cols);
            _optimizer_b = std::make_unique<Momentum<T>>(*it,  *(it+1), 1, _w_cols);
            break;
        }
        case adagrad:{
            assert(n_args == 1);
            _optimizer_w = std::make_unique<Adagrad<T>>(*it, _w_rows, _w_cols);
            _optimizer_b = std::make_unique<Adagrad<T>>(*it, 1, _w_cols);
            break;
        }
        case rmsprop:{
            assert(n_args == 2);
            _optimizer_w = std::make_unique<RMSProp<T>>(*it, *(it+1), _w_rows, _w_cols);
            _optimizer_b = std::make_unique<RMSProp<T>>(*it, *(it+1), 1, _w_cols);
            break;
        }
        case adam:{
            assert(n_args == 3);
            _optimizer_w = std::make_unique<Adam<T>>(*it, *(it+1), *(it+2), _w_rows, _w_cols);
            _optimizer_b = std::make_unique<Adam<T>>(*it, *(it+1), *(it+2), 1, _w_cols);
            break;
        }
		default:{
            std::cout << "Default op used (SGD), lr = 0.001" << std::endl;
            _optimizer_w = std::make_unique<SGD<T>>(0.001);
            _optimizer_b = std::make_unique<SGD<T>>(0.001);
			break;
		}
	};
}

template<typename T>
void FCLayer<T>::freeGrad(){
    _delta.reset(nullptr);
    _weights_grad.reset(nullptr);
    _biases_grad.reset(nullptr);
}

template<typename T>
void FCLayer<T>::reallocBatch(int batch_size){
    if(batch_size == _batch) return;
    _logit.reset(nullptr);
    _logit = std::make_unique<Matrix<T>>(batch_size, _nodes, 0, _n_threads);
    _layer.reset(nullptr);
    _layer = std::make_unique<Matrix<T>>(batch_size, _nodes, 0, _n_threads);
    _batch = batch_size;
}

template<typename T>
void FCLayer<T>::freeOptimizer(){
    _optimizer_w.reset(nullptr);
    _optimizer_b.reset(nullptr);
}
//////////////////////////////////////////////////////////////////////
							/// METHODS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
const Matrix<T>& FCLayer<T>::logit(const Matrix<T>& prev_layer) {
    (*_logit) = prev_layer.dot(*_weights);
    _logit->vBroadcast(*_biases, SUM);
    return (*_logit);
}

template<typename T>
const Matrix<T>& FCLayer<T>::activate(Activation activation){
    _activation = activation;
    switch (activation) {
		case LOGIT:{
			break;
		}
        case RELU:{
            func2D::relu((*_layer), (*_logit));
            break;
        }
        case SWISH:{
            func2D::swish((*_layer), (*_logit));
            break;
        }
        case SIGMOID:{
            func2D::sigmoid((*_layer), (*_logit));
            break;
        }
        case TANH:{
            func2D::tanh((*_layer), (*_logit));
            break;
        }
		default:{
			break;
		}
	};
    return *_layer;
}

template<typename T>
const Matrix<T>& FCLayer<T>::delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next, const std::vector<int>& dropped_indices_next){
    (*_delta) = delta_next.dotTranspose(weights_next);
    
    switch (_activation) {
		case LOGIT:{
			break;
		}
        case RELU:{
            Matrix<T> der_tmp(_delta->getRows(), _delta->getCols(), 0);
            deriv2D::relu(der_tmp, *_layer);
            (*_delta) *= der_tmp;
            break;
        }
        case SWISH:{
            // swish + sigmoid(z)*(1 - swish)
            Matrix<T> sig_mat(_logit->getRows(), _logit->getCols(), 0);
            func2D::sigmoid(sig_mat, *_logit);
            (*_delta) *= (*_layer) + sig_mat * (static_cast<T>(1) - (*_layer));
            break;
        }
        case SIGMOID:{
            (*_delta) *= (*_layer) * (static_cast<T>(1) - (*_layer));
            break;
        }
        case TANH:{
            (*_delta) *= static_cast<T>(1) - (*_layer) * (*_layer);
            break;
        }
        default:{
			break;
		}
	};
    return *_delta;
}

template<typename T>
void FCLayer<T>::gradients(const Matrix<T>& prev_layer, const std::vector<int>& dropped_indices_next){
    const Matrix<T> prev_layer_T = prev_layer.transpose();
    (*_weights_grad) = prev_layer_T.dot(*_delta);
    (*_biases_grad) = _delta->vSum();

}

template<typename T>
void FCLayer<T>::weights_update(const std::vector<int>& dropped_indices_next){
    (*_optimizer_w)(*_weights, *_weights_grad);
    (*_optimizer_b)(*_biases, *_biases_grad);
}

/*
template<typename T>
void FCLayer<T>::gradientCheck(const Matrix<T>& prev_layer, T epsilon){
    Matrix<T> thetaplus = (*_weights) + epsilon;
    Matrix<T> thetaminus = (*_weights) - epsilon;

    Matrix<T> J_plus(this->_rows, this->_cols, 0);
    Matrix<T> J_minus(this->_rows, this->_cols, 0);
    #pragma omp parallel for collapse(2) num_threads(this->_n_threads)
	for(int i = 0; i < this->_rows; ++i){
		for(int j = 0; j < this->_cols; ++j){
			T& reduc_scalar = J_plus(i, j);
			#pragma omp simd reduction(+:reduc_scalar)
            for(int d = 0; d < _dropped_indices.size()-1; ++d){
                const int& idx_start = _dropped_indices[d];
                const int& idx_end = _dropped_indices[d+1];
                for(int k = idx_start+1; k < idx_end; ++k){
                    reduc_scalar += prev_layer(i, k) * thetaplus(k, j);
                }
            }
		}
	}
    #pragma omp parallel for collapse(2) num_threads(this->_n_threads)
	for(int i = 0; i < this->_rows; ++i){
		for(int j = 0; j < this->_cols; ++j){
			T& reduc_scalar = J_minus(i, j);
			#pragma omp simd reduction(+:reduc_scalar)
            for(int d = 0; d < _dropped_indices.size()-1; ++d){
                const int& idx_start = _dropped_indices[d];
                const int& idx_end = _dropped_indices[d+1];
                for(int k = idx_start+1; k < idx_end; ++k){
                    reduc_scalar += prev_layer(i, k) * thetaminus(k, j);
                }
            }
		}
	}
    Matrix<T> gradients = (*_delta);

    //J_plus.applyFunc([](T& val){ val = 1.0/(1.0+std::exp(-val)); });
    //J_minus.applyFunc([](T& val){ val = 1.0/(1.0+std::exp(-val)); });
    Matrix<T> gradapprox = (J_plus-J_minus) / (2*epsilon);
    std::cout << "HERE" << std::endl;
    std::cout << gradients.getRows() << " " << gradapprox.getRows() << " " << gradients.getCols() << " " << gradapprox.getCols() << std::endl;
    assert(gradients.getRows() == gradapprox.getRows() && gradients.getCols() == gradapprox.getCols());
    T numerator = std::sqrt(((gradients-gradapprox)*((gradients-gradapprox))).hSum().vSum()(0, 0));
    T denominator = std::sqrt((gradients*gradients).hSum().vSum()(0, 0)) + std::sqrt((gradapprox*gradapprox).hSum().vSum()(0, 0));
    T diff = numerator / denominator;
    std::cout << "gradient check diff (hid): " << diff << std::endl;
}
*/