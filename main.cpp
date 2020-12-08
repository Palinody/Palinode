#include <iostream>
#include <cmath>
#include <deque>

#include "src/headers/Timer.h" // timer
#include "src/headers/DataParser.h"
#include "src/headers/Matrix.h"
#include "src/headers/FCLayer.h"
#include "src/headers/Output.h"

#include <unordered_map>


/**
 * TODO: 
 * cleanup files management
 * rearrange dot products: use dotTranspose instead of dot during inference
 * write/read files and store model in file
 * cross validation algorithms
 * (preprocessing: PCA / Kalman filters)
 * RNN/CNN
 * DRL
*/

int count_digits(int val){
   int count = 0;
   while(val != 0){
      ++count;
      val /= 10;
   }
   return count;
}

template<typename T>
T normalize(T val, T fromMin, T fromMax, T toMin, T toMax){
    return toMin + (val - fromMin) / (fromMax - fromMin) * (toMax - toMin);
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
        printf("%*d: [%.*f]", count_digits(cost_buff_cpy.getCols()), j, 6, cost_buff_cpy(0, j));
        for(int t = 0; t < static_cast<int>(val); ++t) std::cout << "-";
        std::cout << ">" << std::endl;
    }
}

void XOR(int hiden_nodes_){
    Timer<nano_t> timer;
    
    int batch_size = 4;
    int epochs = 10000;
    float data[] = {1, 0, 1,\
                    0, 1, 1,\
                    0, 0, 0,\
                    1, 1, 0};
    Matrix<float> DATA(data, 4, 3);
    int hidden_nodes = hiden_nodes_;
    FCLayer<float> H(batch_size, hidden_nodes, 2);
    Output<float> Y_hat(batch_size, 1, hidden_nodes);
    //H.optimizer(adam, {0.01, 0.9, 0.999});
    H.optimizer(momentum, {0.01, 0.9});
    //H.optimizer(sgd, {0.1});
    //H.optimizer(nag, {0.01, 0.9});

    //Y_hat.optimizer(adam, {0.01, 0.9, 0.999});
    Y_hat.optimizer(momentum, {0.01, 0.9});
    //Y_hat.optimizer(sgd, {0.1});
    //Y_hat.optimizer(nag, {0.01, 0.9});

    Matrix<float> INPUT(batch_size, 2);
    Matrix<float> TARGET(batch_size, 1);
    Matrix<float> COST_buff(1, epochs);
    
    std::deque<float> buffer; // buffer for early stopping
    size_t buff_max_size = 15; // max size of buffer over which we'll integrate
    float threashold = 1e-1f; // arbitrary threashold
    
    int i;
    timer.reset();
    for(i = 0; i < epochs; ++i){
        DATA.vShuffle();
        INPUT = DATA.getSlice(0, batch_size, 0, 2);
        TARGET = DATA.getSlice(0, batch_size, 2, 3);

        H.logit(INPUT);
        const Matrix<float>& hidden = H.activate(SWISH);
        Y_hat.logit(hidden);
        Y_hat.activate(SIGMOID);

        const Matrix<float>& delta_out = Y_hat.delta(bCE, TARGET);
        Y_hat.gradients(hidden);
        H.delta(delta_out, Y_hat.getWeights());
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
        // END: EARLY STOPPING CRITERION
    }
    uint64_t elapsed = timer.elapsed();
    printCost<float>(COST_buff, 0, 150);
    
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
    printf("Number of epochs: (%d/%d)\n", i, epochs);
    printf("%10s | %10s | %10s\n", "target", "prediction", "confidence");
    for(int i = 0; i < batch_size; ++i) 
        printf("%10d | %10d | %7.1f\n", static_cast<int>(TARGET(i, 0)), static_cast<int>((output(i, 0)<0.5?0:1)/*std::roundf(output(i, 0))*/), 100*output(i, 0));

    printf("\ntraining time: %f\n", (static_cast<float>(elapsed)*1e-9f));
}

void XOR_softmax(int hiden_nodes_){
    Timer<nano_t> timer;
    
    int batch_size = 4;
    int epochs = 10000;
    float data[] = {1, 0, 0, 1,\
                    0, 1, 0, 1,\
                    0, 0, 1, 0,\
                    1, 1, 1, 0};
    Matrix<float> DATA(data, 4, 4);
    int hidden_nodes = hiden_nodes_;
    FCLayer<float> H(batch_size, hidden_nodes, 2);
    Output<float> Y_hat(batch_size, 2, hidden_nodes);
    //H.optimizer(adam, {0.01, 0.9, 0.999});
    //H.optimizer(momentum, {0.01, 0.9});
    H.optimizer(sgd, {0.1});
    //H.optimizer(nag, {0.01, 0.9});

    //Y_hat.optimizer(adam, {0.01, 0.9, 0.999});
    //Y_hat.optimizer(momentum, {0.01, 0.9});
    Y_hat.optimizer(sgd, {0.1});
    //Y_hat.optimizer(nag, {0.01, 0.9});

    Matrix<float> INPUT(batch_size, 2);
    Matrix<float> TARGET(batch_size, 2);
    Matrix<float> COST_buff(1, epochs);
    
    std::deque<float> buffer; // buffer for early stopping
    size_t buff_max_size = 15; // max size of buffer over which we'll integrate
    float threashold = 1e-1f; // arbitrary threashold
    
    int i;
    timer.reset();
    for(i = 0; i < epochs; ++i){
        DATA.vShuffle();
        INPUT = DATA.getSlice(0, batch_size, 0, 2);
        TARGET = DATA.getSlice(0, batch_size, 2, 4);

        H.logit(INPUT);
        const Matrix<float>& hidden = H.activate(SWISH);
        Y_hat.logit(hidden);
        Y_hat.activate(SOFTMAX);

        const Matrix<float>& delta_out = Y_hat.delta(CE, TARGET);
        Y_hat.gradients(hidden);
        H.delta(delta_out, Y_hat.getWeights());
        H.gradients(INPUT);
        Y_hat.weights_update();
        H.weights_update();

        // accumulate cost values for graph
        COST_buff(0, i) = Y_hat.getCost(CE);

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
        // END: EARLY STOPPING CRITERION
    }
    uint64_t elapsed = timer.elapsed();
    printCost<float>(COST_buff, 0, 150);
    
    // START: INFERENCE
    DATA.vShuffle();
    INPUT = DATA.getSlice(0, batch_size, 0, 2);
    TARGET = DATA.getSlice(0, batch_size, 2, 4);
    H.logit(INPUT);
    const Matrix<float>& hidden = H.activate(SWISH);
    Y_hat.logit(hidden);
    const Matrix<float>& output = Y_hat.activate(SOFTMAX);
    // END: INFERENCE

    // RESULTS
    printf("Number of epochs: (%d/%d)\n", i, epochs);
    printf("%10s | %10s | %10s\n", "target", "prediction", "confidence");
    for(int i = 0; i < batch_size; ++i){
        int max_idx_targ = (TARGET(i, 0)) ? 0 : 1;
        int max_idx_pred = (output(i, 0)>output(i, 1)) ? 0 : 1;
        float max_confidence = std::max(output(i, 0), output(i, 1));
        printf("%10d | %10d | %7.1f\n", max_idx_targ, max_idx_pred, 100*max_confidence);
    }
    printf("\ntraining time: %f\n", (static_cast<float>(elapsed)*1e-9f));
}

int main(int argc, char **argv){
    /**
     * arg: number of hidden nodes
     * N.B.: you will notice that
     *      softmax is way more consistent
     *      than the sigmoid output for
     *      2 hidden nodes.
     * N.B.2: the training is stopped
     *      if the cost function stagnates.
     *      The training might stop early
     * */


    XOR(2);
    //XOR_softmax(2);
    /*
    const std::string path = "../../../databases/image-recognition/mnist/mnist_test.csv";
    CSVParser<int> csv(path);
    printf("%d, %d\n", csv.getRows(), csv.getCols());

    Matrix<int> chunk = csv.parseChunk(0, 10);
    std::cout << chunk.col(0) << std::endl;
    */
    
    return 0;
}