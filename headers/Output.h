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
    float getAccuracy(const Matrix<T>& target, const std::string& accuracy="cathegorical");
    const Matrix<T>& getWeights() const;
    const std::vector<int>& getDroppedIdx() const;
    const Matrix<T> getThreasholdClip() const;
    

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

private:
    Matrix<T> _layer;          // f(Z)
    Matrix<T> _logit;          // Z = X.W + b
    Matrix<T> _weights;        // weights matrix
    Matrix<T> _biases;         // bias matrix

    Matrix<T> _delta;          // layer derivative: (DJ)/(Dy_hat)
    Matrix<T> _weights_grad;   // weights gradients: (DJ)/(DW)
    Matrix<T> _biases_grad;    // biases gradient
    Matrix<T> _target;         // mini-batch target layer
    std::unique_ptr<Optimizer<T>> _optimizer_w;
    std::unique_ptr<Optimizer<T>> _optimizer_b;

    Matrix<T> _threashold_buffer; // keep tracks of the target output frequency for threasholding

    int _batch;
    int _nodes;
    int _w_rows, _w_cols;   // number of rows/cols of weight matrix (_w_rows == _n_prev_nodes)
    int _n_prev_nodes;      // number of nodes of previous layer

    float _dropout = 0;     // dropout prob [0, 1]
    std::vector<int> _dropped_indices;

    Activation _activation = LOGIT;
    CostFunc _costFunc;

    int _n_threads;
};
/////////////////////////////////////////////////////////////////////
//                                                                 //
//                          IMPLEMENTATION                         //
//                                                                 //
/////////////////////////////////////////////////////////////////////

template<typename T>
Output<T>::Output(int n_batch, int n_nodes, int n_prev_nodes, T value, int num_threads) : 
    _batch{ n_batch }, _nodes{ n_nodes },
    _w_rows{ n_prev_nodes }, _w_cols{ n_nodes },
    _n_prev_nodes{ n_prev_nodes }/*,
    _dropped_indices{ -1, n_prev_nodes }, _n_threads{ num_threads },
    _layer{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
    _logit{ Matrix<T>(n_batch, n_nodes, 0, num_threads) },
    _weights{ Matrix<T>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0, num_threads) },
    _biases{ Matrix<T>(1, n_nodes, XAVIER, n_prev_nodes, 0, num_threads) },
    _delta{ Matrix<T>(_batch, _nodes, 0, _n_threads) },
    _weights_grad{ Matrix<T>(_w_rows, _w_cols, 0, _n_threads) },
    _biases_grad{ Matrix<T>(1, _w_cols, 0, _n_threads) },
    _target{ Matrix<T>(_batch, _nodes, 0, _n_threads) },
    _threashold_buffer{ Matrix<T>(_batch, _nodes, 0, _n_threads) }*/{
    
    _layer        = Matrix<T>(n_batch, n_nodes, 0, num_threads);
    _logit        = Matrix<T>(n_batch, n_nodes, 0, num_threads);
    _weights      = Matrix<T>(n_prev_nodes, n_nodes, XAVIER, n_prev_nodes, 0, num_threads);
    _biases       = Matrix<T>(1, n_nodes, XAVIER, n_prev_nodes, 0, num_threads);
    _delta        = Matrix<T>(_batch, _nodes, 0, _n_threads);
    _weights_grad = Matrix<T>(_w_rows, _w_cols, 0, _n_threads);
    _biases_grad  = Matrix<T>(1, _w_cols, 0, _n_threads);
    
    _target       = Matrix<T>(_batch, _nodes, 0, _n_threads);
    _threashold_buffer = Matrix<T>(_batch, _nodes, 0, _n_threads);
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
                    const T& y_hat = _layer(i, j);
                    const T& y = _target(i, j);
                    cost -= y*log(std::max(type_min, y_hat));
                }
            }
            cost /= (_batch * _nodes);
            break;
        }
        case bCE:{
            // stable binary cross entropy loss
            // max(0, z) - yz + log(1 + exp(-|z|))
            Matrix<T> relu_tmp(_logit.getRows(), _logit.getCols(), 0);
            relu_tmp.copy(_logit);
            func2D::relu(relu_tmp, _logit);

            Matrix<T> log_tmp(_logit.getRows(), _logit.getCols(), 0);
            log_tmp.copy(_logit);
            func2D::abs(log_tmp);
            log_tmp *= (-1);
            func2D::exp(log_tmp);
            log_tmp += 1.0;
            func2D::log(log_tmp);

            cost = (relu_tmp - _target * _logit + log_tmp).hSum().vSum()(0, 0);
            cost /= (_batch * _nodes);
            break;
        }
        case MSE:{
            for(int i = 0; i < _batch; ++i){
                for(int j = 0; j < _nodes; ++j){
                    cost += std::pow(_layer(i, j) - _target(i, j), 2);
                }
            }
            cost /= (2 * _batch * _nodes);
            break;
        }
    };
    return cost;
}

template<typename T>
float Output<T>::getAccuracy(const Matrix<T>& target, const std::string& accuracy){
    Matrix<int> outputMaxIdx = _layer.vMaxIndex();
    Matrix<int> targetMaxIdx = target.vMaxIndex();
    Matrix<int> comparison = outputMaxIdx.compare(targetMaxIdx);
    return static_cast<float>(comparison.vSum()(0, 0)) / comparison.getRows();
}

template<typename T>
const Matrix<T>& Output<T>::getWeights() const {
    return _weights;
}

template<typename T>
const std::vector<int>& Output<T>::getDroppedIdx() const {
    return _dropped_indices;
}

template<typename T>
const Matrix<T> Output<T>::getThreasholdClip() const {
    Matrix<T> h_vect = _threashold_buffer.vSum();
    T sum = h_vect.hSum()(0, 0);
    h_vect /= sum;
    return h_vect;
}

//////////////////////////////////////////////////////////////////////
						/// MEM. MANAGEMENT ///
//////////////////////////////////////////////////////////////////////

template<typename T>
void Output<T>::reallocBatch(int batch_size){
    if(batch_size == _batch) return;
    _logit = Matrix<T>(batch_size, _nodes, 0, _n_threads);
    _layer = Matrix<T>(batch_size, _nodes, 0, _n_threads);
    _batch = batch_size;
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

//////////////////////////////////////////////////////////////////////
							/// METHODS ///
//////////////////////////////////////////////////////////////////////

template<typename T>
const Matrix<T>& Output<T>::logit(const Matrix<T>& prev_layer) {
    //_logit = prev_layer.dot(_weights);
    _logit.dot(prev_layer, _weights);
    _logit.vBroadcast(_biases, SUM);
    return _logit;
}

template<typename T>
const Matrix<T>& Output<T>::activate(Activation activation){
    _activation = activation;
    switch (activation) {
		case LOGIT:{
			break;
		}
        case RELU:{
            func2D::relu(_layer, _logit);
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
        case SOFTMAX:{
            Matrix<T> batchMax = _logit.vMax();     // logit example wise shift to avoid NaN
            _logit.hBroadcast(batchMax, SUB);       // shift data to get example-wise max = 0
            func2D::exp(_layer, _logit);            // apply exp element-wise
            Matrix<T> summed = _layer.hSum();       // compute example-wise normalization terms
            _layer.hBroadcast(summed, DIV);         // normalize
            break;
        }
		default:{
			break;
		}
	};
    return _layer;
}

template<typename T>
const Matrix<T>& Output<T>::delta(CostFunc costFunc, const Matrix<T>& target){
    _costFunc = costFunc;
    _target.copy(target);
    switch (costFunc) {
        case CE:{
            // Cross entropy when used with softmax
            _delta = _layer - target;
            break;
        }
        case bCE:{
            // stable binary cross entropy derivative (with sigmoid activatiion)
            _delta = _layer - target;
            break;
        }
        case MSE:{
            _delta = _layer - target;
            switch (_activation) {
                case LOGIT:{
                    break;
                }
                case SIGMOID:{
                    _delta *= _layer * (static_cast<T>(1) - _layer);
                    break;
                }
                case TANH:{
                    func2D::pow(_layer, 2);
                    _delta *= static_cast<T>(1) - _layer;
                    break;
                }
                default:{ break; }
            };
            break;
        }
        default:{ break; }
    };
    return _delta;
}

template<typename T>
void Output<T>::gradients(const Matrix<T>& prev_layer){
    const Matrix<T> prev_layer_T = prev_layer.transpose();
    //(*_weights_grad) = prev_layer_T.dot(*_delta);
    _weights_grad.dot(prev_layer_T, _delta);
    _biases_grad = _delta.vSum();
}

template<typename T>
void Output<T>::weights_update(){
    (*_optimizer_w)(_weights, _weights_grad);
    (*_optimizer_b)(_biases, _biases_grad);
}

template<typename T>
const Matrix<T>& Output<T>::updateThreashold(const Matrix<T>& target){
    _threashold_buffer += target;
    return _threashold_buffer;
}