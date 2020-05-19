#pragma once

#include "FCLayer.h"

//enum Activation { LOGIT, RELU, SIGMOID, TANH, SOFTMAX };
enum CostFunc { CE, bCE, MSE }; // Cross entropy, binary-cross entropy, mean squared error

template<typename T>
class Output{
public:
    Output(int n_batch, int n_nodes, int n_prev_nodes, T value = 0, int num_threads = 1);

    // ** setters **
    void dropout(float drop_rate);
    void clip(T inf=0.1, T sup=0.9);

    // ** getters **
    float getCost(CostFunc costFunc);
    const Matrix<T>& getWeights();
    const std::vector<int>& getDroppedIdx();
    const Matrix<T> getThreasholdClip();
    

    // ** mem. management **
    void mallocGrad();
    void mallocTarget();
    void optimizer(OptimizerName op, std::initializer_list<double> args);
    void mallocThreashold();
    void freeGrad();
    void freeTarget();
    void reallocBatch(int batchSize);
    void freeThreashold();
    void freeOptimizer();

    // ** methods **
    const Matrix<T>& logit(const Matrix<T>& prev_layer);
    const Matrix<T>& activate(Activation activation);

    const Matrix<T>& delta(CostFunc costFunc, const Matrix<T>& target);
    void gradients(const Matrix<T>& prev_layer);
    void weights_update();

    const Matrix<T>& updateThreashold(const Matrix<T>& target);

    void gradientCheck(const Matrix<T>& prev_layer, T epsilon);

private:
    std::unique_ptr<Matrix<T>> _layer;          // f(Z)
    std::unique_ptr<Matrix<T>> _logit;          // Z = X.W + b
    std::unique_ptr<Matrix<T>> _weights;        // weights matrix
    std::unique_ptr<Matrix<T>> _biases;         // bias matrix

    std::unique_ptr<Matrix<T>> _delta;          // layer derivative: (DJ)/(Dy_hat)
    std::unique_ptr<Matrix<T>> _weights_grad;   // weights gradients: (DJ)/(DW)
    std::unique_ptr<Matrix<T>> _biases_grad;    // biases gradient
    std::unique_ptr<Matrix<T>> _target;         // mini-batch target layer

    std::unique_ptr<Optimizer<T>> _optimizer_w;
    std::unique_ptr<Optimizer<T>> _optimizer_b;

    std::unique_ptr<Matrix<T>> _threashold_buffer; // keep tracks of the target output frequency for threasholding

    int _w_rows, _w_cols;   // number of rows/cols of weight matrix (_w_rows == _n_prev_nodes)
    int _n_prev_nodes;      // number of nodes of previous layer
    // NEW args
    int _batch;
    int _nodes;

    float _dropout = 0;     // dropout prob [0, 1]
    std::vector<int> _dropped_indices;

    Activation _activation = LOGIT;
    CostFunc _costFunc;

    int _n_threads;
};
//////////////////////////////////////////////////////////////////////
//																	//
//							 IMPLEMENTATION	                        //
//						                                            //
//////////////////////////////////////////////////////////////////////

template<typename T>
Output<T>::Output(int n_batch, int n_nodes, int n_prev_nodes, T value, int num_threads) : 
    _batch{ n_batch }, _nodes{ n_nodes },
    _w_rows{ n_prev_nodes }, _w_cols{ n_nodes },
    _n_prev_nodes{ n_prev_nodes },
    _dropped_indices{ -1, n_prev_nodes }, _n_threads{ num_threads }{

    _layer      = std::make_unique<Matrix<T>>(n_batch, n_nodes, 0, num_threads);
    _logit      = std::make_unique<Matrix<T>>(n_batch, n_nodes, 0, num_threads);
    _weights    = std::make_unique<Matrix<T>>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads);
    _biases     = std::make_unique<Matrix<T>>(1, n_nodes, XAVIER, n_prev_nodes, 0/*n_nodes*/, num_threads);
}
//////////////////////////////////////////////////////////////////////
							/// SETTERS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Output<T>::dropout(float drop_rate){
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
/**
 * Clipping should be used solely during the training phase
 * and should be used to avoid NaNs returned by the bCE and
 * CE loss functions and their derivatives. The output should
 * not be clipped if the prediction is close to the target.
 * It should be clipped when it is very CONFIDENT AND WRONG.
*/
template<typename T>
void Output<T>::clip(T inf, T sup){
    for(int i = 0; i < _batch; ++i){
        for(int j = 0; j < _nodes; ++j){
            if((*_target)(i, j) && _layer(i, j)<inf) _layer(i, j) = inf;
            else if (!(*_target)(i, j) && _layer(i, j)>sup) _layer(i, j) = sup;
        }
    }
}

//////////////////////////////////////////////////////////////////////
						/// GETTERS ///
//////////////////////////////////////////////////////////////////////
template<typename T>
float Output<T>::getCost(CostFunc costFunc){
    float cost{ 0 };
    switch (costFunc) {
        case CE:{
            T type_min = std::numeric_limits<T>::min();
            for(int i = 0; i < _batch; ++i){
                for(int j = 0; j < _nodes; ++j){
                    const T& y_hat = (*_layer)(i, j);
                    const T& y = (*_target)(i, j);
                    cost -= y*log(std::max(type_min, y_hat));
                }
            }
            cost /= (_batch * _nodes);
            break;
        }
        case bCE:{
            // stable binary cross entropy loss
            // max(0, z) - yz + log(1 + exp(-|z|))
            Matrix<T> relu_tmp(_logit->getRows(), _logit->getCols(), 0);
            relu_tmp.copy(*_logit);
            func2D::relu(relu_tmp, *_logit);

            Matrix<T> log_tmp(_logit->getRows(), _logit->getCols(), 0);
            log_tmp.copy(*_logit);
            func2D::abs(log_tmp);
            log_tmp *= (-1);
            func2D::exp(log_tmp);
            log_tmp += 1.0;
            func2D::log(log_tmp);

            cost = (relu_tmp - (*_target) * (*_logit) + log_tmp).hSum().vSum()(0, 0);
            cost /= (_batch * _nodes);
            break;
        }
        case MSE:{
            for(int i = 0; i < _batch; ++i){
                for(int j = 0; j < _nodes; ++j){
                    cost += std::pow((*_layer)(i, j) - (*_target)(i, j), 2);
                }
            }
            cost /= (2 * _batch * _nodes);
            break;
        }
    };
    return cost;
}

template<typename T>
const Matrix<T>& Output<T>::getWeights(){
    return *_weights;
}

template<typename T>
const std::vector<int>& Output<T>::getDroppedIdx(){
    return _dropped_indices;
}

template<typename T>
const Matrix<T> Output<T>::getThreasholdClip(){
    Matrix<T> h_vect = _threashold_buffer->vSum();
    T sum = h_vect.hSum()(0, 0);
    h_vect /= sum;
    return h_vect;
}

//////////////////////////////////////////////////////////////////////
						/// MEM. MANAGEMENT ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Output<T>::mallocGrad(){
    _delta = std::make_unique<Matrix<T>>(_batch, _nodes, 0, _n_threads);
    _weights_grad = std::make_unique<Matrix<T>>(_w_rows, _w_cols, 0, _n_threads);
    _biases_grad = std::make_unique<Matrix<T>>(1, _w_cols, 0, _n_threads);
}

template<typename T>
void Output<T>::mallocTarget(){
    _target = std::make_unique<Matrix<T>>(_batch, _nodes, 0, _n_threads);
}

template<typename T>
void Output<T>::mallocThreashold(){
    _threashold_buffer = std::make_unique<Matrix<T>>(_batch, _nodes, 0, _n_threads);
}

template<typename T>
void Output<T>::freeGrad(){
    _delta.reset(nullptr);
    _weights_grad.reset(nullptr);
    _biases_grad.reset(nullptr);
}

template<typename T>
void Output<T>::freeTarget(){
    _target.reset(nullptr);
}

template<typename T>
void Output<T>::freeOptimizer(){
    _optimizer_w.reset(nullptr);
    _optimizer_b.reset(nullptr);
}

template<typename T>
void Output<T>::reallocBatch(int batch_size){
    if(batch_size == _batch) return;
    _logit.reset(nullptr);
    _logit = std::make_unique<Matrix<T>>(batch_size, _nodes, 0, _n_threads);
    _layer.reset(nullptr);
    _layer = std::make_unique<Matrix<T>>(batch_size, _nodes, 0, _n_threads);
    _batch = batch_size;
}

template<typename T>
void Output<T>::freeThreashold(){
    _threashold_buffer.reset(nullptr);
}

template<typename T>
void Output<T>::optimizer(OptimizerName op, std::initializer_list<double> args){
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
            _optimizer_b = std::make_unique<Momentum<T>>(*it, *(it+1), 1, _w_cols);
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

//////////////////////////////////////////////////////////////////////
							/// METHODS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
const Matrix<T>& Output<T>::logit(const Matrix<T>& prev_layer) {
    (*_logit) = prev_layer.dot(*_weights);
    _logit->vBroadcast(*_biases, SUM);
    return *_logit;
}

template<typename T>
const Matrix<T>& Output<T>::activate(Activation activation){
    _activation = activation;
    switch (activation) {
		case LOGIT:{
			break;
		}
        case RELU:{
            func2D::relu(*_layer, *_logit);
            break;
        }
        case SIGMOID:{
            func2D::sigmoid(*_layer, *_logit);
            break;
        }
        case TANH:{
            func2D::tanh(*_layer, *_logit);
            break;
        }
        case SOFTMAX:{
            /*
            Matrix<T> batchMax = this->vMax();                  // logit example wise shift to avoid NaN
            this->hBroadcast(batchMax, SUB);                    // shift data to get example-wise max = 0
            this->applyFunc([](T& val){ val = std::exp(val); });// apply exp element-wise
            Matrix<T> summed = this->hSum();                    // compute example-wise normalization terms
            this->hBroadcast(summed, DIV);                      // normalize
            */
            Matrix<T> batchMax = _logit->vMax();
            _logit->hBroadcast(batchMax, SUB);
            func2D::exp(*_layer, *_logit);
            Matrix<T> summed = _layer->hSum();
            _layer->hBroadcast(summed, DIV);
            break;
        }
		default:{
			break;
		}
	};
    return *_layer;
}

template<typename T>
const Matrix<T>& Output<T>::delta(CostFunc costFunc, const Matrix<T>& target){
    _costFunc = costFunc;
    _target->copy(target);
    switch (costFunc) {
        case CE:{
            (*_delta) = (*_layer) - target;
            break;
        }
        case bCE:{
            // stable binary cross entropy derivative (with sigmoid activatiion)
            (*_delta) = (*_layer) - target;
            break;
        }
        case MSE:{
            (*_delta) = (*_layer) - target;
            switch (_activation) {
                case LOGIT:{
                        break;
                    }
                    case SIGMOID:{
                        (*_delta) *= (*_layer) * (static_cast<T>(1) - (*_layer));
                        break;
                    }
                    case TANH:{
                        func2D::pow(*_layer, 2);
                        (*_delta) *= static_cast<T>(1) - (*_layer);
                        break;
                    }
                    default:{ break; }
            };
            break;
        }
        default:{ break; }
    };
    return *_delta;
}

template<typename T>
void Output<T>::gradients(const Matrix<T>& prev_layer){
    const Matrix<T> prev_layer_T = prev_layer.transpose();
    (*_weights_grad) = prev_layer_T.dot(*_delta);
    (*_biases_grad) = _delta->vSum();
}

template<typename T>
void Output<T>::weights_update(){
    (*_optimizer_w)(*_weights, *_weights_grad);
    (*_optimizer_b)(*_biases, *_biases_grad);
}

template<typename T>
const Matrix<T>& Output<T>::updateThreashold(const Matrix<T>& target){
    (*_threashold_buffer) += target;
    return (*_threashold_buffer);
}

/*
template<typename T>
void Output<T>::gradientCheck(const Matrix<T>& prev_layer, T epsilon){
    Matrix<T> weights_plus = (*_weights);
    Matrix<T> bias_plus = (*_biases);
    Matrix<T> weights_minus = (*_weights);
    Matrix<T> bias_minus = (*_biases);
    // predictions made with modified gradients
    Matrix<T> Y_hat_plus(_batch, _nodes, 0);
    Matrix<T> Y_hat_minus(_batch, _nodes, 0);
    // gradients computed with grad checking
    Matrix<T> JW_plus(_w_rows, _w_cols);
    Matrix<T> JW_minus(_w_rows, _w_cols);
    Matrix<T> Jb_plus(1, _w_cols);
    Matrix<T> Jb_minus(1, _w_cols);
    
    for(int w_i = 0; w_i < _w_rows; ++w_i){
        for(int w_j = 0; w_j < _w_cols; ++w_j){
            weights_plus(w_i, w_j) = (*_weights)(w_i, w_j) + epsilon;
            weights_minus(w_i, w_j) = (*_weights)(w_i, w_j) - epsilon;

            for(int i = 0; i < _batch; ++i){
                for(int j = 0; j < _nodes; ++j){
                    T& reduc_scalar_plus = Y_hat_plus(i, j);
                    T& reduc_scalar_minus = Y_hat_minus(i, j);
                    reduc_scalar_plus = 0;
                    reduc_scalar_minus = 0;
                    for(int k = 0; k < _w_rows; ++k){
                        reduc_scalar_plus += prev_layer(i, k) * weights_plus(k, j);
                        reduc_scalar_minus += prev_layer(i, k) * weights_minus(k, j);
                    }
                }
            }
            Y_hat_plus.vBroadcast(bias_plus, SUM);
            Y_hat_minus.vBroadcast(bias_minus, SUM);
            Y_hat_plus.applyFunc([](T& val){ val = 1.0 / (1.0 + std::exp(-val)); });
            Y_hat_minus.applyFunc([](T& val){ val = 1.0 / (1.0 + std::exp(-val)); });

            // computing error (MSE)
            for(int i = 0; i < _batch; ++i){
                for(int j = 0; j < _nodes; ++j){
                    JW_plus(w_i, w_j) += std::pow(Y_hat_plus(i, j) - (*_target)(i, j), 2);
                    JW_minus(w_i, w_j) += std::pow(Y_hat_minus(i, j) - (*_target)(i, j), 2);
                }
            }
            JW_plus(w_i, w_j) /= (2 * _nodes * _batch);
            JW_minus(w_i, w_j) /= (2 * _nodes * _batch);

            // reset corresponding weight
            weights_plus(w_i, w_j) = (*_weights)(w_i, w_j);
            weights_minus(w_i, w_j) = (*_weights)(w_i, w_j);
        }
    }
    for(int w_j = 0; w_j < _w_cols; ++w_j){
        bias_plus(0, w_j) = (*_biases)(0, w_j) + epsilon;
        bias_minus(0, w_j) = (*_biases)(0, w_j) - epsilon;

        for(int i = 0; i < _batch; ++i){
            for(int j = 0; j < _nodes; ++j){
                T& reduc_scalar_plus = Y_hat_plus(i, j);
                T& reduc_scalar_minus = Y_hat_minus(i, j);
                reduc_scalar_plus = 0;
                reduc_scalar_minus = 0;
                for(int k = 0; k < _w_rows; ++k){
                    reduc_scalar_plus += prev_layer(i, k) * weights_plus(k, j);
                    reduc_scalar_minus += prev_layer(i, k) * weights_minus(k, j);
                }
            }
        }
        Y_hat_plus.vBroadcast(bias_plus, SUM);
        Y_hat_minus.vBroadcast(bias_minus, SUM);
        // computing error (MSE)
        for(int i = 0; i < _batch; ++i){
            for(int j = 0; j < _nodes; ++j){
                Jb_plus(0, w_j) += std::pow(Y_hat_plus(i, j) - (*_target)(i, j), 2);
                Jb_minus(0, w_j) += std::pow(Y_hat_minus(i, j) - (*_target)(i, j), 2);
            }
        }
        Jb_plus(0, w_j) /= (2 * _batch * _nodes);
        Jb_minus(0, w_j) /= (2 * _batch * _nodes);
        // resetting biases
        bias_plus(0, w_j) = (*_biases)(0, w_j);
        bias_minus(0, w_j) = (*_biases)(0, w_j);
    }

    Matrix<T> gradapprox_w = (JW_plus-JW_minus) / (2*epsilon);
    Matrix<T> gradapprox_b = (Jb_plus-Jb_minus) / (2*epsilon);
    //std::cout << (*_delta).getRows() << " " << gradapprox.getRows() << " " << (*_delta).getCols() << " " << gradapprox.getCols() << std::endl;
    //assert((*_delta).getRows() == gradapprox.getRows() && (*_delta).getCols() == gradapprox.getCols());
    T numerator_w = (((*_weights_grad)-gradapprox_w)*((*_weights_grad)-gradapprox_w)).hSum().vSum()(0, 0);
    T numerator_b = (((*_biases_grad)-gradapprox_b)*((*_biases_grad)-gradapprox_b)).hSum().vSum()(0, 0);
    numerator_w = std::sqrt(numerator_w);
    numerator_b = std::sqrt(numerator_b);
    
    T denominator1_w = ((*_weights_grad)*(*_weights_grad)).hSum().vSum()(0, 0);
    T denominator2_w = (gradapprox_w*gradapprox_w).hSum().vSum()(0, 0);
    T denominator1_b = ((*_biases_grad)*(*_biases_grad)).hSum().vSum()(0, 0);
    T denominator2_b = (gradapprox_b*gradapprox_b).hSum().vSum()(0, 0);
    denominator1_w = std::sqrt(denominator1_w);
    denominator2_w = std::sqrt(denominator2_w);
    T denominator_w = denominator1_w + denominator2_w;
    denominator1_b = std::sqrt(denominator1_b);
    denominator2_b = std::sqrt(denominator2_b);
    T denominator_b = denominator1_b + denominator2_b;

    T diff_w = numerator_w / denominator_w;
    T diff_b = numerator_b / denominator_b;
    
    std::cout << "gradient check diff w (out): " << (diff_w) << std::endl;
    std::cout << "gradient check diff b (out): " << (diff_b) << std::endl;
}
*/