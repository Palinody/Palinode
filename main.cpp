#include <iostream>
#include <cmath>
#include <deque>

#include "headers/Utils.h" // timer
#include "headers/DataParser.h"
#include "headers/Matrix.h"
#include "headers/FCLayer.h"
#include "headers/Output.h"

#include <unordered_map>

template<typename T>
T normalize(T val, T fromMin, T fromMax, T toMin, T toMax){
    return toMin + (val - fromMin)/(fromMax - fromMin) * (toMax - toMin);
}

template <typename T>
void printCost(const Matrix<T>& cost_buff, T inf=0, T sup=100){
    Matrix<T> cost_buff_cpy(cost_buff.getRows(), cost_buff.getCols());
    cost_buff_cpy.copy(cost_buff);
    T min_ = 0/*cost_buff.min()*/;
    T max_ = cost_buff_cpy.max();
    for(int j = 0; j < cost_buff_cpy.getCols(); j+=1){
        if(cost_buff_cpy(0, j) == 0) break;
        T val = normalize(cost_buff_cpy(0, j), min_, max_, inf, sup);
        printf("%4d: [%6f]", j, cost_buff_cpy(0, j));
        for(int t = 0; t < static_cast<int>(val); ++t) std::cout << "-";
        std::cout << ">" << std::endl;
    }
}

int main(int argc, char **argv){
    Timer<nano_t> timer;

    int batch_size = 4;
    int epochs = 10000;
    float data[] = {1, 0, 1,\
                    0, 1, 1,\
                    0, 0, 0,\
                    1, 1, 0};
    Matrix<float> DATA(data, 4, 3);
    int hidden_nodes = 8;
    FCLayer<float> H(batch_size, hidden_nodes, 2);
    Output<float> Y_hat(batch_size, 1, hidden_nodes);
    H.optimizer(adam, {0.001, 0.9, 0.999});
    //H.optimizer(sgd, {0.01});
    Y_hat.optimizer(adam, {0.001, 0.9, 0.999});
    //Y_hat.optimizer(sgd, {0.01});

    Matrix<float> INPUT(batch_size, 2);
    Matrix<float> TARGET(batch_size, 1);
    Matrix<float> COST_buff(1, epochs);
    
    std::deque<float> buffer; // buffer for early stopping
    size_t buff_max_size = 5; // max size of buffer over which we'll integrate
    float threashold = 1e-2f; // arbitrary threashold
    
    int i;
    timer.reset();
    for(i = 0; i < epochs; ++i){
        DATA.vShuffle();
        INPUT = DATA.getSlice(0, batch_size, 0, 2);
        TARGET = DATA.getSlice(0, batch_size, 2, 3);

        H.logit(INPUT);
        const Matrix<float>& hidden = H.activate(SWISH);
        Y_hat.logit(hidden);
        const Matrix<float>& output = Y_hat.activate(SIGMOID);

        const Matrix<float>& delta_out = Y_hat.delta(bCE, TARGET);
        Y_hat.gradients(hidden);
        const Matrix<float>& delta_hidden = H.delta(delta_out, Y_hat.getWeights());
        H.gradients(INPUT);
        Y_hat.weights_update();
        H.weights_update();

        // accumulate cost values for graph
        COST_buff(0, i) = Y_hat.getCost(bCE);

        // START: EARLY STOPPING CRITERION
        if(buffer.size() < buff_max_size) buffer.push_back(COST_buff(0, i));
        else{
            // integrate over buffer
            float area = 0;
            for(std::deque<float>::iterator it = buffer.begin(); it != buffer.end(); ++it)
                area += *it;
            if(area < threashold) break;
            buffer.pop_front();
        }
        /*
        Matrix<float> output_round(batch_size, 1, 0);
        output_round = output;
        for(int i = 0; i < batch_size; ++i) output_round(i, 0) = (output_round(i, 0) > 0.5f) ? 1.0f : 0.0f;
        bool isEqual = output_round.isEqual(TARGET);
        if(isEqual) break;
        */
        // END: EARLY STOPPING CRITERION
    }
    uint64_t elapsed = timer.elapsed();
    printCost<float>(COST_buff, 0, 100);
    
    // START: INFERENCE
    DATA.vShuffle();
    INPUT = DATA.getSlice(0, batch_size, 0, 2);
    TARGET = DATA.getSlice(0, batch_size, 2, 3);
    H.logit(INPUT);
    const Matrix<float>& hidden = H.activate(SWISH);
    Y_hat.logit(hidden);
    const Matrix<float>& output = Y_hat.activate(SIGMOID);
    // END: INFERENCE

    // RESULTS
    printf("Number of epochs: %d\n", i);
    printf("%10s | %10s | %10s\n", "target", "prediction", "confidence");
    for(int i = 0; i < batch_size; ++i) 
        printf("%10d | %10d | %7.1f\n", static_cast<int>(TARGET(i, 0)), static_cast<int>(std::roundf(output(i, 0))), 100*output(i, 0));

    printf("\n%f\n", (static_cast<float>(elapsed)*1e-9f));
    return 0;
}