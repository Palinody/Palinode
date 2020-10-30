#pragma once

#include "Matrix.h"
#include "Optimizers.h"
#include <algorithm>
#include <initializer_list>

enum Activation { LOGIT, RELU, SWISH, SIGMOID, TANH, SOFTMAX };


template<typename T>
class FCLayer{
public:
    FCLayer(int n_batch, int n_nodes, int n_prev_nodes, T value = 0, int num_threads = 1);

    // ** setters **
    void dropout(float drop_rate);

    // ** getters **
    const Matrix<T>& getWeights() const;
    const std::vector<int>& getDroppedIdx() const;

    // ** mem. management **
    //void mallocGrad();
    void optimizer(OptimizerName op, std::initializer_list<double> args);
    //void freeGrad();
    void reallocBatch(int batchSize);
    //void freeOptimizer();

    // ** methods **
    const Matrix<T>& logit(const Matrix<T>& prev_layer);
    const Matrix<T>& activate(Activation activation);

    const Matrix<T>& delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next);
    const Matrix<T>& delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next, const std::vector<int>& dropped_indices_next);
    void gradients(const Matrix<T>& prev_layer);
    void gradients(const Matrix<T>& prev_layer, const std::vector<int>& dropped_indices_next);
    void weights_update();
    void weights_update(const std::vector<int>& dropped_indices_next);

private:
    int _batch;
    int _nodes;
    int _w_rows, _w_cols;   // number of rows/cols of weight matrix (_w_rows == _n_prev_nodes)
    int _n_prev_nodes;      // number of nodes of previous layer
    float _dropout = 0;     // dropout prob [0, 1]
    std::vector<int> _dropped_indices;
    Activation _activation = LOGIT;
    int _n_threads;

    Matrix<T> _layer;          // f(Z)
    Matrix<T> _logit;          // Z = X.W + b
    Matrix<T> _weights;        // weights matrix
    Matrix<T> _biases;         // bias matrix
    
    Matrix<T> _delta;          // layer derivative: (DJ)/(Dy_hat)
    Matrix<T> _weights_grad;   // weights gradients: (DJ)/(DW)
    Matrix<T> _biases_grad;    // biases gradient
    std::unique_ptr<Optimizer<T>> _optimizer_w;
    std::unique_ptr<Optimizer<T>> _optimizer_b;
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
    _layer{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
    _logit{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
    _weights{ Matrix<T>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads) },
    _biases{ Matrix<T>(1, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads) },
    _delta{ Matrix<T>(_batch, _nodes, 0, _n_threads) },
    _weights_grad{ Matrix<T>(_w_rows, _w_cols, 0, _n_threads) },
    _biases_grad{ Matrix<T>(1, _w_cols, 0, _n_threads) }{
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
const Matrix<T>& FCLayer<T>::getWeights() const {
    return _weights;
}

template<typename T>
const std::vector<int>& FCLayer<T>::getDroppedIdx() const {
    return _dropped_indices;
}

//////////////////////////////////////////////////////////////////////
						/// MEM. MANAGEMENT ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void FCLayer<T>::optimizer(OptimizerName op, std::initializer_list<double> args){
    int n_args = args.size();
    std::initializer_list<double>::iterator it = args.begin();
    switch(op){
		case sgd:{
            assert(n_args == 1);
            _optimizer_w = std::make_unique<SGD<T>>(*it);
            _optimizer_b = std::make_unique<SGD<T>>(*it);
			break;
		}
        case momentum:{
            assert(n_args == 2);
            _optimizer_w = std::make_unique<Momentum<T>>(*it, *(it+1), _w_rows, _w_cols);
            _optimizer_b = std::make_unique<Momentum<T>>(*it, *(it+1), 1, _w_cols);
            break;
        }
        case nag:{
            assert(n_args == 2);
            _optimizer_w = std::make_unique<NAG<T>>(*it, *(it+1), _w_rows, _w_cols);
            _optimizer_b = std::make_unique<NAG<T>>(*it, *(it+1), 1, _w_cols);
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
void FCLayer<T>::reallocBatch(int batch_size){
    if(batch_size == _batch) return;
    _logit = Matrix<T>(batch_size, _nodes, 0, _n_threads);
    _layer = Matrix<T>(batch_size, _nodes, 0, _n_threads);
    _batch = batch_size;
}
//////////////////////////////////////////////////////////////////////
							/// METHODS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
const Matrix<T>& FCLayer<T>::logit(const Matrix<T>& prev_layer) {
    //_logit = prev_layer.dot(_weights);
    _logit.dot(prev_layer, _weights);
    _logit.vBroadcast(_biases, SUM);
    return _logit;
}

template<typename T>
const Matrix<T>& FCLayer<T>::activate(Activation activation){
    _activation = activation;
    switch(activation){
		case LOGIT:{
			break;
		}
        case RELU:{
            func2D::relu(_layer, _logit);
            break;
        }
        case SWISH:{
            func2D::swish(_layer, _logit);
            break;
        }
        case SIGMOID:{
            func2D::sigmoid(_layer, _logit);
            break;
        }
        case TANH:{
            func2D::tanh(_layer, _logit);
            break;
        }
		default:{
			break;
		}
	};
    return _layer;
}

template<typename T>
const Matrix<T>& FCLayer<T>::delta(const Matrix<T>& delta_next, const Matrix<T>& weights_next){
    //_delta = delta_next.dotTranspose(weights_next);
    _delta.dotTranspose(delta_next, weights_next);
    
    switch(_activation){
		case LOGIT:{
			break;
		}
        case RELU:{
            Matrix<T> der_tmp(_delta.getRows(), _delta.getCols(), 0);
            deriv2D::relu(der_tmp, _layer);
            _delta *= der_tmp;
            break;
        }
        case SWISH:{
            // swish + sigmoid(z) * (1 - swish)
            Matrix<T> swish_der(_logit.getRows(), _logit.getCols(), 0);
            deriv2D::swish(swish_der, _logit, _layer);
            _delta *= swish_der;
            break;
        }
        case SIGMOID:{
            Matrix<T> sig_der(_batch, _nodes);
            deriv2D::sigmoid(sig_der, _layer);
            _delta *= sig_der;//_layer * (static_cast<T>(1) - _layer);
            break;
        }
        case TANH:{
            _delta *= static_cast<T>(1) - _layer * _layer;
            break;
        }
        default:{
			break;
		}
	};
    return _delta;
}

template<typename T>
void FCLayer<T>::gradients(const Matrix<T>& prev_layer){
    const Matrix<T> prev_layer_T = prev_layer.transpose();
    //_weights_grad = prev_layer_T.dot(_delta);
    _weights_grad.dot(prev_layer_T, _delta);
    _biases_grad = _delta.vSum();
}

template<typename T>
void FCLayer<T>::weights_update(){
    (*_optimizer_w)(_weights, _weights_grad);
    (*_optimizer_b)(_biases, _biases_grad);
}