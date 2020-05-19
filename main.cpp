#include <iostream>

#include "headers/Utils.h" // timer
#include "headers/DataParser.h"
#include "headers/Matrix.h"
#include "Input.h"

#include "FCLayer.h"
#include "Output.h"

#include <unordered_map>

template<typename T>
Matrix<T> toCathegoricalInput(const Matrix<int>& data){
    Matrix<T> oh_data(data.getRows(), (4+13)*5, 0);
    for(int i = 0; i < data.getRows(); ++i)
        for(int j = 0; j < 5; ++j)
            for(int c = 0; c < 2; ++c)
                oh_data(i, data(i, c+j*2)-1 + 4 * c + (j * 17)) = 1;
    return oh_data;
}

template<typename T>
Matrix<T> toCathegoricalTarget(const Matrix<int>& data, int cathegories=10){
    Matrix<T> oh_data(data.getRows(), cathegories, 0);
    for(int i = 0; i < data.getRows(); ++i){
        oh_data(i, data(i, 0)) = 1;
    }
    return oh_data;
}

/**
 * selects examples from dataset in such a way that the 
 * labels frequency remains uniformly distributed
*/
Matrix<int> selectUniformData(Matrix<int>& data){
    // counts occurences number for given class
    Matrix<int> labels_counter(1, 10, 0);
    for(int i = 0; i < data.getRows(); ++i) ++labels_counter(0, data(i, 10));
    int min_val = labels_counter.vMin()(0, 0);

    // create dataset where each class occurence < min_idx
    Matrix<int> labels_counter_2(1, 10, 0);
    Matrix<int> maj = Matrix<int>(1, 10, min_val); // min val is majorant for each class
    Matrix<int> DATABASE_filtered = data.row(0);
    ++labels_counter_2(0, data(0, 10));
    for(int i = 1; i < data.getRows(); ++i){
        if(labels_counter_2(0, data(i, 10)) < min_val){
            DATABASE_filtered.vStack(data.row(i));
            ++labels_counter_2(0, data(i, 10));
        }
        //if(labels_counter_2.isEqual(maj)) break; !!! commented because compiling with -fopenmp -> issue !!!
    }
    return DATABASE_filtered;
}

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
        T val = normalize(cost_buff_cpy(0, j), min_, max_, inf, sup);
        printf("%4d: [%6f]", j, cost_buff_cpy(0, j));
        for(int t = 0; t < static_cast<int>(val); ++t) std::cout << "-";
        std::cout << ">" << std::endl;
    }
}

void XOR(){
    Timer<nano_t> timer;

    int batch_size = 4;
    int epochs = 200;
    float data[] = {1, 0, 1,\
                    0, 1, 1,\
                    0, 0, 0,\
                    1, 1, 0};
    Matrix<float> DATA(data, 4, 3);

    int hidden_nodes = 2;
    FCLayer<float> H(batch_size, hidden_nodes, 2, 0, 6);
    Output<float> Y_hat(batch_size, 1, hidden_nodes, 0, 6);
    H.mallocGrad();
    H.optimizer(adam, {0.001, 0.9, 0.999});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(adam, {0.001, 0.9, 0.999});
    
    Matrix<float> INPUT(batch_size, 2);
    Matrix<float> TARGET(batch_size, 1);
    Matrix<float> COST_buff(1, epochs);
    for(int epoch = 0; epoch < epochs; ++epoch){
        DATA.vShuffle();
        INPUT = DATA.getSlice(0, batch_size, 0, 2);
        TARGET = DATA.col(2);
        //INPUT -= 0.5;
        //INPUT *= 2;
        
        H.logit(INPUT);
        const Matrix<float>& hidden = H.activate(SWISH);

        Y_hat.logit(hidden);
        Y_hat.activate(SIGMOID);
        //Y_hat.clip(0.01, 0.99);
        
        const Matrix<float>& delta_out = Y_hat.delta(bCE, TARGET);
        Y_hat.gradients(hidden);
        const Matrix<float>& weights_out = Y_hat.getWeights();
        const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
        
        H.delta(delta_out, weights_out, dropped_out);
        H.gradients(INPUT, dropped_out);
        
        //update
        Y_hat.weights_update();
        H.weights_update(dropped_out);

        //std::cout << "cost: " << Y_hat.getCost(bCE) << std::endl;
        COST_buff(0, epoch) = Y_hat.getCost(bCE);
    }
    printCost(COST_buff);
    DATA.vShuffle();
    INPUT = DATA.getSlice(0, batch_size, 0, 2);
    //INPUT -= 0.5;
    //INPUT *= 2;
    TARGET = DATA.col(2);
    
    H.logit(INPUT);
    const Matrix<float>& hidden = H.activate(SWISH);
    Y_hat.logit(hidden);
    const Matrix<float>& output = Y_hat.activate(SIGMOID);

    Matrix<float> output_cpy(output.getRows(),output.getCols());
    output_cpy.copy(output);
    std::cout << output_cpy << std::endl;
    output_cpy > 0.5;
    std::cout << output_cpy << std::endl;
    std::cout << std::endl;
    std::cout << TARGET << std::endl;
    

    printf("time total: %.5f\n", (timer.elapsed()*1e-9));
}

void XOR_softmax(){
    Timer<nano_t> timer;

    int batch_size = 4;
    int epochs = 100;
    float data[] = {1, 0, 0, 1,\
                    0, 1, 0, 1,\
                    1, 1, 1, 0,\
                    0, 0, 1, 0};
    Matrix<float> DATA(data, 4, 4);

    int hidden_nodes = 10;
    FCLayer<float> H(batch_size, hidden_nodes, 2, 0, 6);
    Output<float> Y_hat(batch_size, 2, hidden_nodes, 0, 6);
    H.mallocGrad();
    H.optimizer(adam, {0.001, 0.9, 0.999});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(adam, {0.001, 0.9, 0.999});
    
    Matrix<float> INPUT(batch_size, 2);
    Matrix<float> TARGET(batch_size, 2);
    Matrix<float> COST_buff(1, epochs);
    for(int epoch = 0; epoch < epochs; ++epoch){
        DATA.vShuffle();
        INPUT = DATA.getSlice(0, batch_size, 0, 2);
        TARGET = DATA.getSlice(0, batch_size, 2, 4);
        //INPUT -= 0.5;
        //INPUT *= 2;
        
        H.logit(INPUT);
        const Matrix<float>& hidden = H.activate(SWISH);

        Y_hat.logit(hidden);
        Y_hat.activate(SOFTMAX);
        //Y_hat.clip(0.01, 0.99);
        
        const Matrix<float>& delta_out = Y_hat.delta(CE, TARGET);
        Y_hat.gradients(hidden);
        const Matrix<float>& weights_out = Y_hat.getWeights();
        const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
        
        H.delta(delta_out, weights_out, dropped_out);
        H.gradients(INPUT, dropped_out);
        
        //update
        Y_hat.weights_update();
        H.weights_update(dropped_out);

        //std::cout << "cost: " << Y_hat.getCost(bCE) << std::endl;
        COST_buff(0, epoch) = Y_hat.getCost(CE);
    }
    printCost(COST_buff);
    DATA.vShuffle();
    INPUT = DATA.getSlice(0, batch_size, 0, 2);
    //INPUT -= 0.5;
    //INPUT *= 2;
    TARGET = DATA.getSlice(0, batch_size, 2, 4);
    
    H.logit(INPUT);
    const Matrix<float>& hidden = H.activate(SWISH);
    Y_hat.logit(hidden);
    const Matrix<float>& output = Y_hat.activate(SOFTMAX);

    Matrix<float> output_cpy(output.getRows(),output.getCols());
    output_cpy.copy(output);
    std::cout << output_cpy << std::endl;
    std::cout << output_cpy.vMaxIndex() << std::endl;
    std::cout << std::endl;
    std::cout << TARGET << std::endl;
    

    printf("time total: %.5f\n", (timer.elapsed()*1e-9));
}

void mnist(){
    
    std::string to_path_confusion_matrix = "data/output/confusion_matrix_test.txt";
    TXTParser<float> confusion_parser(to_path_confusion_matrix);

    
    Timer<nano_t> timer;
    std::string from_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/input/mnist/mnist_train.csv";
    std::string from_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/input/mnist/mnist_test.csv";
    CSVParser<int> csv(from_path_train);
    CSVParser<int> csv_test(from_path_test);
    std::cout << "Dataset size: " << csv.getRows() << " | " << csv.getCols() << std::endl;
    std::cout << "Dataset size: " << csv_test.getRows() << " | " << csv_test.getCols() << std::endl;

    Matrix<int> DATASET_TEST = csv_test.parseChunk(0, csv_test.getRows());
    Matrix<float> INPUT_TEST(csv_test.getRows(), 784);
    Matrix<int> target_test = DATASET_TEST.col(0);
    Matrix<float> oh_target_test = toCathegoricalTarget<float>(target_test);
    for(int i = 0; i < INPUT_TEST.getRows(); ++i)
        for(int j = 0; j < INPUT_TEST.getCols(); ++j)
            INPUT_TEST(i, j) = DATASET_TEST(i, j+1) / 255.0;
    
    std::string to_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/output/training_cost.txt";
    std::string to_path_test = "data/output/test_cost.txt";
	TXTParser<float> train_parser(to_path_train);
    TXTParser<float> test_parser(to_path_test);

    int batch_size = 64;
    int epochs = 3;
    Matrix<float> DATA(batch_size, csv.getRows());
    int hidden_nodes = 1000;
    FCLayer<float> H(batch_size, hidden_nodes, 784, 0, 6);
    Output<float> Y_hat(batch_size, 10, hidden_nodes, 0, 6);
    H.mallocGrad();
    H.optimizer(adam, {0.001, 0.9, 0.999});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(adam, {0.001, 0.9, 0.999});
    
    //Matrix<float> INPUT(batch_size, 784);
    //Matrix<float> TARGET(batch_size, 10);
    std::vector<float> COST_buff;
    std::vector<float> COST_buff_test;
    size_t train_counter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch){
        Selecter<int> selecter(0, csv.getRows(), 1);
        while(!selecter.empty()){
            std::vector<int> indices = selecter.pick_batch(batch_size);
            Matrix<int> BATCH(indices.size(), csv.getCols());
            csv.parseRows(BATCH, indices);
            Matrix<float> oh_target = toCathegoricalTarget<float>(BATCH.col(0));
            Matrix<float> INPUT(BATCH.getRows(), 784);
            for(int i = 0; i < INPUT.getRows(); ++i)
                for(int j = 0; j < INPUT.getCols(); ++j)
                    INPUT(i, j) = BATCH(i, j+1) / 255.0;
            H.reallocBatch(BATCH.getRows());
            Y_hat.reallocBatch(BATCH.getRows());
            Y_hat.freeTarget();
            Y_hat.mallocTarget();
        
            H.logit(INPUT);
            const Matrix<float>& hidden = H.activate(SWISH);

            Y_hat.logit(hidden);
            Y_hat.activate(SOFTMAX);
            //Y_hat.clip(0.01, 0.99);
            
            const Matrix<float>& delta_out = Y_hat.delta(CE, oh_target);
            Y_hat.gradients(hidden);
            const Matrix<float>& weights_out = Y_hat.getWeights();
            const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
            
            H.delta(delta_out, weights_out, dropped_out);
            H.gradients(INPUT, dropped_out);
            
            //update
            Y_hat.weights_update();
            H.weights_update(dropped_out);

            printf("%6d | (%6d/%6d) %1.12f\n", epoch, selecter.getSize(), 60000, Y_hat.getCost(CE));

            if(train_counter % 10 == 0){
                COST_buff.push_back(Y_hat.getCost(CE));

                // START: TEST PHASE
                H.reallocBatch(INPUT_TEST.getRows());
                Y_hat.reallocBatch(INPUT_TEST.getRows());
                Y_hat.freeTarget();
                Y_hat.mallocTarget();

                H.logit(INPUT_TEST);
                const Matrix<float>& hidden_test = H.activate(SWISH);
                
                Y_hat.logit(hidden_test);
                const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);

                COST_buff_test.push_back(Y_hat.getCost(CE));
                // END: TEST PHASE
                
                // save result 
                train_parser.putData(COST_buff.begin(), COST_buff.end(), 1, COST_buff.size(), to_path_train, false, ',');
                test_parser.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');
                // confusion matrix
                Matrix<int> results = output_test.vMaxIndex();
                Matrix<int> confusion_matrix(10, 10, 0);
                for(int i = 0; i < results.getRows(); ++i){
                    ++confusion_matrix(results(i, 0), target_test(i, 0));
                }

                confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                            confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                            to_path_confusion_matrix, true, ',');
            }
            ++train_counter;
        }
    }
    
    // START: TEST PHASE
    H.reallocBatch(INPUT_TEST.getRows());
    Y_hat.reallocBatch(INPUT_TEST.getRows());

    H.logit(INPUT_TEST);
    const Matrix<float>& hidden_test = H.activate(SWISH);

    Y_hat.logit(hidden_test);
    const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);
    // END: TEST PHASE

    // save result 
    train_parser.putData(COST_buff.begin(), COST_buff.end(), 1, COST_buff.size(), to_path_train, false, ',');
    test_parser.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');
    // confusion matrix
    Matrix<int> results = output_test.vMaxIndex();
    Matrix<int> confusion_matrix(10, 10, 0);
    for(int i = 0; i < results.getRows(); ++i){
        ++confusion_matrix(results(i, 0), target_test(i, 0));
    }

    confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                to_path_confusion_matrix, true, ',');
    /*
    DATA.vShuffle();
    INPUT = DATA.getSlice(0, batch_size, 0, 2);
    //INPUT -= 0.5;
    //INPUT *= 2;
    TARGET = DATA.getSlice(0, batch_size, 2, 4);
    
    H.logit(INPUT);
    const Matrix<float>& hidden = H.activate(SWISH);
    Y_hat.logit(hidden);
    const Matrix<float>& output = Y_hat.activate(SOFTMAX);

    Matrix<float> output_cpy(output.getRows(),output.getCols());
    output_cpy.copy(output);
    std::cout << output_cpy << std::endl;
    std::cout << output_cpy.vMaxIndex() << std::endl;
    std::cout << std::endl;
    std::cout << TARGET << std::endl;
    */

    printf("time total: %.5f\n", (timer.elapsed()*1e-9));
}

/**
 * Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
 * Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)
 * Hand (10 int): [Ord1, Num1, ..., Ord5, Num5]
 * 
 * Class (10 int): [0-9]
 * 
 * The purpose of this function is to construct a lexicographic
 * ordering mechanism for a pokerhand with 5 cards. Cards may 
 * come in different rearrangements in different hands and its
 * goal is to reorder them.
 * The algorithm assumes the following:
 *      The deck has no more than 52 cards 
 *      There are no card duplicates
*/
template<typename T>
void order_hand(TXTParser<int>& parser, Matrix<T>& data, Matrix<T>& target, const std::vector<int>& indices){
    Matrix<int> line(1, parser.getCols(), 0);
    for(int i = 0; i < data.getRows(); ++i){
        parser.getDataIt(line.begin(), line.end(), indices[i]);
        //std::cout << line << std::endl;
        // input data
        for(int j = 0; j < line.getCols()-1; j+=2){
            data(i, line(0, j)-1 + (line(0, j+1)-1)*4) = 1;
        }
        // target data
        target(i, line(0, line.getCols()-1)) = 1;
    }
    //std::cout << data << std::endl;
    //std::cout << "-----" << std::endl;
    //std::cout << target << std::endl;
}

template<typename T>
void order_hand(TXTParser<int>& parser, Matrix<T>& data, Matrix<T>& target, int curr_idx){
    Matrix<int> line(1, parser.getCols(), 0);
    for(int i = 0; i < data.getRows(); ++i){
        parser.getDataIt(line.begin(), line.end(), curr_idx+i);
        //std::cout << line << std::endl;
        // input data
        for(int j = 0; j < line.getCols()-1; j+=2){
            data(i, line(0, j)-1 + (line(0, j+1)-1)*4) = 1;
        }
        target(i, 0) = line(0, line.getCols()-1);
    }
}

/**
 * checks whether a dataset contains dupplicates.
 * a new dataset made of unique samples is created
 * and stored
 * @param from_path: source file location
 * @param to_path: storage location of the filtered data
 * @param chunk_size: buffer size for txt parser in terms of rows
*/
template<typename T>
void filter_dupplicates(const std::string& from_path, const std::string to_path, int chunk_size){
    TXTParser<int> input_parser(from_path);
    int data_rows  = input_parser.getRows();
    int data_cols  = input_parser.getCols();

    std::cout << "Original dataset dimension\n";
    std::cout << data_rows << " | " << data_cols << std::endl;
    /**
     * unordered set of target values (keys)
     * each key maps to an unordered set containing example data
     * this allows fast retrieval of an object mapped to
     * a particular target value
     *      std::unordered_set<std::pair::<T, Matrix<T>>>
     * or more clearly
     *      {target: Matrix, ..., other_target: other_Matrix}
     * N.B.: would be much better to store matrices of indices
     *      of the locations of 1s instead of a sparse matrix of ones...
    */
    std::unordered_map<T, std::vector<Matrix<T>>> unique_dataset_map;

    //Matrix<T> buffer(chunk_size, data_cols);
    for(int i = 0; i < data_rows; i+= chunk_size){
        std::cout << i << '\r';
	    std::cout.flush();
        
        Matrix<T> INPUT(chunk_size, 52);
        Matrix<T> target(chunk_size, 1);

        // buffer is rechecked in loop because buffer iterator might
        // go out of bouds of file if i+chunk_size > data_rows
        if(i+chunk_size > data_rows){
            Matrix<T> new_input(data_rows-i, 52);
            Matrix<T> new_target(data_rows-i, 1);
            INPUT = new_input;
            target = new_target;
        }

        order_hand(input_parser, INPUT, target, i);
        for(int i_chunk = 0; i_chunk < std::min(chunk_size, data_rows-i); ++i_chunk){
            auto curr = unique_dataset_map.find(target(i_chunk, 0));
            // if key not in unordered_map, add it with corr. sample data
            if(curr == unique_dataset_map.end()){
                unique_dataset_map[target(i_chunk, 0)] = std::vector<Matrix<T>>{};
                unique_dataset_map[target(i_chunk, 0)].push_back(INPUT.row(i_chunk));
            } else {
                // is already in vector token
                bool isIn = false;
                for(int j = 0; j < (curr->second).size(); ++j){
                    if((curr->second).size() != 0 && (curr->second)[j].isEqual(INPUT.row(i_chunk))){
                        isIn = true;
                        break;
                    }
                }
                if(!isIn) (curr->second).push_back(INPUT.row(i_chunk));
            }
        }
    }
    for(auto it = unique_dataset_map.begin(); it != unique_dataset_map.end(); ++it){
        std::cout << "target: " << it->first << " | n: " << (it->second).size() << std::endl;
    }

}

void poker_hand_momentum(){
    // timer object
    Timer<nano_t> timer;
    // training / test data parers (input)
    std::string from_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-training-true.data";
    std::string from_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-testing.data";
    TXTParser<int> input_parser_train(from_path_train);
    TXTParser<int> input_parser_test(from_path_test);
    std::cout << input_parser_train.getRows() << " | " << input_parser_train.getCols() << std::endl;
    std::cout << input_parser_test.getRows() << " | " << input_parser_test.getCols() << std::endl;
    // training / test data parers (output) -> stores training Losses
    std::string to_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_train_momentum.txt";
    std::string to_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_test_momentum.txt";
    TXTParser<int> output_parser_train(to_path_train);
    TXTParser<int> output_parser_test(to_path_test);
    // results
    std::string to_path_confusion_matrix = "data/output/poker/confusion_matrix_test_momentum.txt";
    TXTParser<int> confusion_parser(to_path_confusion_matrix);
    // model
    int batch_size = 64;
    int epochs = 30;
    int input_nodes = 52;
    int hidden_nodes_1 = 256;
    int hidden_nodes_2 = 256;
    int output_nodes = 10;
    FCLayer<float> H1(batch_size, hidden_nodes_1, input_nodes,    0);
    FCLayer<float> H2(batch_size, hidden_nodes_2, hidden_nodes_1, 0);
    Output<float> Y_hat(batch_size, output_nodes, hidden_nodes_2, 0);
    H1.mallocGrad();
    H1.optimizer(momentum, {0.0001, 0.9});
    H2.mallocGrad();
    H2.optimizer(momentum, {0.0001, 0.9});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(momentum, {0.0001, 0.9});
    Y_hat.mallocThreashold();
    
    std::vector<float> COST_buff_train;
    size_t train_counter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch){
        Selecter<int> selecter(0, input_parser_train.getRows(), 1);
        if(epoch == 1) std::cout << selecter.getSize() << std::endl;
        while(!selecter.empty()){
            std::vector<int> indices = selecter.pick_batch(batch_size);
            Matrix<float> INPUT(indices.size(), input_nodes);
            Matrix<float> oh_target(indices.size(), output_nodes, 0);
            order_hand(input_parser_train, INPUT, oh_target, indices);
            if(indices.size() != batch_size) continue;
            H1.reallocBatch(INPUT.getRows());
            H2.reallocBatch(INPUT.getRows());
            Y_hat.reallocBatch(INPUT.getRows());
            Y_hat.freeTarget();
            Y_hat.mallocTarget();
        
            H1.logit(INPUT);
            const Matrix<float>& hidden1 = H1.activate(SWISH);

            H2.logit(hidden1);
            const Matrix<float>& hidden2 = H2.activate(SWISH);

            Y_hat.logit(hidden2);
            Y_hat.activate(SOFTMAX);
            //Y_hat.clip(0.01, 0.99);
            
            const Matrix<float>& delta_out = Y_hat.delta(CE, oh_target);
            Y_hat.gradients(hidden2);
            const Matrix<float>& weights_out = Y_hat.getWeights();
            const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
            
            const Matrix<float>& delta2 = H2.delta(delta_out, weights_out, dropped_out);
            H2.gradients(hidden1, dropped_out);
            const Matrix<float>& weights2 = H2.getWeights();
            const std::vector<int>& dropped2 = H2.getDroppedIdx();

            H1.delta(delta2, weights2, dropped2);
            H1.gradients(INPUT, dropped2);
            
            //update
            Y_hat.weights_update();
            H2.weights_update(dropped_out);
            H1.weights_update(dropped2);

            // THREASHOLD UPDATE
            Y_hat.updateThreashold(oh_target);

            if(train_counter % 10 == 0) COST_buff_train.push_back(Y_hat.getCost(CE));
            ++train_counter;

            printf("%6d | (%6d/%6d) %1.12f\n", epoch, selecter.getSize(), 60000, Y_hat.getCost(CE));
        }
    }
    // free training memory
    H1.freeGrad();
    H1.freeOptimizer(),
    H2.freeGrad();
    H2.freeOptimizer();
    Y_hat.freeGrad();
    Y_hat.freeOptimizer();
    Y_hat.freeTarget();

    const Matrix<float> threasholdClip = Y_hat.getThreasholdClip();

    // save training results
    output_parser_train.putData(COST_buff_train.begin(), COST_buff_train.end(), 1, COST_buff_train.size(), to_path_train, false, ',');

    Matrix<int> confusion_matrix(10, 10, 0);
    Selecter<int> selecter(0, input_parser_test.getRows(), 1);
    size_t test_counter = 0;
    std::vector<float> COST_buff_test;
    while(!selecter.empty()){
        // START: TEST PHASE
        std::vector<int> indices = selecter.pick_batch(1000);
        Matrix<float> INPUT(indices.size(), input_nodes);
        Matrix<float> oh_target(indices.size(), output_nodes, 0);
        order_hand(input_parser_train, INPUT, oh_target, indices);

        H1.reallocBatch(INPUT.getRows());
        H2.reallocBatch(INPUT.getRows());
        Y_hat.reallocBatch(INPUT.getRows());
        Y_hat.freeTarget();
        Y_hat.mallocTarget();
        
        H1.logit(INPUT);
        const Matrix<float>& hidden1 = H1.activate(SWISH);

        H2.logit(hidden1);
        const Matrix<float>& hidden2 = H2.activate(SWISH);

        Y_hat.logit(hidden2);
        const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);
        Matrix<float> output_test_cpy = output_test;
        for(int i = 0; i < indices.size(); ++i){
            for(int j = 0; j < output_nodes; ++j){
                output_test_cpy(i, j) = (output_test(i, j)>threasholdClip(0, j)) ? 1 : 0;
            }
        }
        if(test_counter % 10 == 0) COST_buff_test.push_back(Y_hat.getCost(CE));
        ++test_counter;
        // END: TEST PHASE
        // confusion matrix
        Matrix<int> results = output_test_cpy.vMaxIndex();
        Matrix<int> target_test = oh_target.vMaxIndex();

        //std::cout << results.hStack(target_test) << std::endl;
        //std::cout << "---" << std::endl;
        //std::cout << results(0, 0) << " | " << target_test(0, 0) << std::endl;
    
        for(int i = 0; i < results.getRows(); ++i){
            ++confusion_matrix(results(i, 0), target_test(i, 0));
        }

        printf("(%6d/%6d)\n", selecter.getSize(), 1000000);
    }
    output_parser_test.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');

    confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                to_path_confusion_matrix, true, ',');
}

void poker_hand_adam(){
    // timer object
    Timer<nano_t> timer;
    // training / test data parers (input)
    std::string from_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-training-true.data";
    std::string from_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-testing.data";
    TXTParser<int> input_parser_train(from_path_train);
    TXTParser<int> input_parser_test(from_path_test);
    std::cout << input_parser_train.getRows() << " | " << input_parser_train.getCols() << std::endl;
    std::cout << input_parser_test.getRows() << " | " << input_parser_test.getCols() << std::endl;
    // training / test data parers (output) -> stores training Losses
    std::string to_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_train_adam.txt";
    std::string to_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_test_adam.txt";
    TXTParser<int> output_parser_train(to_path_train);
    TXTParser<int> output_parser_test(to_path_test);
    // results
    std::string to_path_confusion_matrix = "data/output/poker/confusion_matrix_test_adam.txt";
    TXTParser<int> confusion_parser(to_path_confusion_matrix);
    // model
    int batch_size = 64;
    int epochs = 30;
    int input_nodes = 52;
    int hidden_nodes_1 = 256;
    int hidden_nodes_2 = 256;
    int output_nodes = 10;
    FCLayer<float> H1(batch_size, hidden_nodes_1, input_nodes,    0);
    FCLayer<float> H2(batch_size, hidden_nodes_2, hidden_nodes_1, 0);
    Output<float> Y_hat(batch_size, output_nodes, hidden_nodes_2, 0);
    H1.mallocGrad();
    H1.optimizer(adam, {0.0001, 0.9, 0.999});
    H2.mallocGrad();
    H2.optimizer(adam, {0.0001, 0.9, 0.999});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(adam, {0.0001, 0.9, 0.999});
    Y_hat.mallocThreashold();
    
    std::vector<float> COST_buff_train;
    size_t train_counter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch){
        Selecter<int> selecter(0, input_parser_train.getRows(), 1);
        if(epoch == 1) std::cout << selecter.getSize() << std::endl;
        while(!selecter.empty()){
            std::vector<int> indices = selecter.pick_batch(batch_size);
            Matrix<float> INPUT(indices.size(), input_nodes);
            Matrix<float> oh_target(indices.size(), output_nodes, 0);
            order_hand(input_parser_train, INPUT, oh_target, indices);
            if(indices.size() != batch_size) continue;
            H1.reallocBatch(INPUT.getRows());
            H2.reallocBatch(INPUT.getRows());
            Y_hat.reallocBatch(INPUT.getRows());
            Y_hat.freeTarget();
            Y_hat.mallocTarget();
        
            H1.logit(INPUT);
            const Matrix<float>& hidden1 = H1.activate(SWISH);

            H2.logit(hidden1);
            const Matrix<float>& hidden2 = H2.activate(SWISH);

            Y_hat.logit(hidden2);
            Y_hat.activate(SOFTMAX);
            //Y_hat.clip(0.01, 0.99);
            
            const Matrix<float>& delta_out = Y_hat.delta(CE, oh_target);
            Y_hat.gradients(hidden2);
            const Matrix<float>& weights_out = Y_hat.getWeights();
            const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
            
            const Matrix<float>& delta2 = H2.delta(delta_out, weights_out, dropped_out);
            H2.gradients(hidden1, dropped_out);
            const Matrix<float>& weights2 = H2.getWeights();
            const std::vector<int>& dropped2 = H2.getDroppedIdx();

            H1.delta(delta2, weights2, dropped2);
            H1.gradients(INPUT, dropped2);
            
            //update
            Y_hat.weights_update();
            H2.weights_update(dropped_out);
            H1.weights_update(dropped2);

            // THREASHOLD UPDATE
            Y_hat.updateThreashold(oh_target);

            if(train_counter % 10 == 0) COST_buff_train.push_back(Y_hat.getCost(CE));
            ++train_counter;

            printf("%6d | (%6d/%6d) %1.12f\n", epoch, selecter.getSize(), 60000, Y_hat.getCost(CE));
        }
    }
    // free training memory
    H1.freeGrad();
    H1.freeOptimizer(),
    H2.freeGrad();
    H2.freeOptimizer();
    Y_hat.freeGrad();
    Y_hat.freeOptimizer();
    Y_hat.freeTarget();

    const Matrix<float> threasholdClip = Y_hat.getThreasholdClip();

    // save training results
    output_parser_train.putData(COST_buff_train.begin(), COST_buff_train.end(), 1, COST_buff_train.size(), to_path_train, false, ',');

    Matrix<int> confusion_matrix(10, 10, 0);
    Selecter<int> selecter(0, input_parser_test.getRows(), 1);
    size_t test_counter = 0;
    std::vector<float> COST_buff_test;
    while(!selecter.empty()){
        // START: TEST PHASE
        std::vector<int> indices = selecter.pick_batch(1000);
        Matrix<float> INPUT(indices.size(), input_nodes);
        Matrix<float> oh_target(indices.size(), output_nodes, 0);
        order_hand(input_parser_train, INPUT, oh_target, indices);

        H1.reallocBatch(INPUT.getRows());
        H2.reallocBatch(INPUT.getRows());
        Y_hat.reallocBatch(INPUT.getRows());
        Y_hat.freeTarget();
        Y_hat.mallocTarget();
        
        H1.logit(INPUT);
        const Matrix<float>& hidden1 = H1.activate(SWISH);

        H2.logit(hidden1);
        const Matrix<float>& hidden2 = H2.activate(SWISH);

        Y_hat.logit(hidden2);
        const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);
        Matrix<float> output_test_cpy = output_test;
        for(int i = 0; i < indices.size(); ++i){
            for(int j = 0; j < output_nodes; ++j){
                output_test_cpy(i, j) = (output_test(i, j)>threasholdClip(0, j)) ? 1 : 0;
            }
        }
        if(test_counter % 10 == 0) COST_buff_test.push_back(Y_hat.getCost(CE));
        ++test_counter;
        // END: TEST PHASE
        // confusion matrix
        Matrix<int> results = output_test_cpy.vMaxIndex();
        Matrix<int> target_test = oh_target.vMaxIndex();

        //std::cout << results.hStack(target_test) << std::endl;
        //std::cout << "---" << std::endl;
        //std::cout << results(0, 0) << " | " << target_test(0, 0) << std::endl;
    
        for(int i = 0; i < results.getRows(); ++i){
            ++confusion_matrix(results(i, 0), target_test(i, 0));
        }

        printf("(%6d/%6d)\n", selecter.getSize(), 1000000);
    }
    output_parser_test.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');

    confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                to_path_confusion_matrix, true, ',');
}

void poker_hand_rmsprop(){
    // timer object
    Timer<nano_t> timer;
    // training / test data parers (input)
    std::string from_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-training-true.data";
    std::string from_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-testing.data";
    TXTParser<int> input_parser_train(from_path_train);
    TXTParser<int> input_parser_test(from_path_test);
    std::cout << input_parser_train.getRows() << " | " << input_parser_train.getCols() << std::endl;
    std::cout << input_parser_test.getRows() << " | " << input_parser_test.getCols() << std::endl;
    // training / test data parers (output) -> stores training Losses
    std::string to_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_train_rmsprop.txt";
    std::string to_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_test_rmsprop.txt";
    TXTParser<int> output_parser_train(to_path_train);
    TXTParser<int> output_parser_test(to_path_test);
    // results
    std::string to_path_confusion_matrix = "data/output/poker/confusion_matrix_test_rmsprop.txt";
    TXTParser<int> confusion_parser(to_path_confusion_matrix);
    // model
    int batch_size = 64;
    int epochs = 30;
    int input_nodes = 52;
    int hidden_nodes_1 = 256;
    int hidden_nodes_2 = 256;
    int output_nodes = 10;
    FCLayer<float> H1(batch_size, hidden_nodes_1, input_nodes,    0);
    FCLayer<float> H2(batch_size, hidden_nodes_2, hidden_nodes_1, 0);
    Output<float> Y_hat(batch_size, output_nodes, hidden_nodes_2, 0);
    H1.mallocGrad();
    H1.optimizer(rmsprop, {0.0001, 0.9});
    H2.mallocGrad();
    H2.optimizer(rmsprop, {0.0001, 0.9});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(rmsprop, {0.0001, 0.9});
    Y_hat.mallocThreashold();
    
    std::vector<float> COST_buff_train;
    size_t train_counter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch){
        Selecter<int> selecter(0, input_parser_train.getRows(), 1);
        if(epoch == 1) std::cout << selecter.getSize() << std::endl;
        while(!selecter.empty()){
            std::vector<int> indices = selecter.pick_batch(batch_size);
            Matrix<float> INPUT(indices.size(), input_nodes);
            Matrix<float> oh_target(indices.size(), output_nodes, 0);
            order_hand(input_parser_train, INPUT, oh_target, indices);
            if(indices.size() != batch_size) continue;
            H1.reallocBatch(INPUT.getRows());
            H2.reallocBatch(INPUT.getRows());
            Y_hat.reallocBatch(INPUT.getRows());
            Y_hat.freeTarget();
            Y_hat.mallocTarget();
        
            H1.logit(INPUT);
            const Matrix<float>& hidden1 = H1.activate(SWISH);

            H2.logit(hidden1);
            const Matrix<float>& hidden2 = H2.activate(SWISH);

            Y_hat.logit(hidden2);
            Y_hat.activate(SOFTMAX);
            //Y_hat.clip(0.01, 0.99);
            
            const Matrix<float>& delta_out = Y_hat.delta(CE, oh_target);
            Y_hat.gradients(hidden2);
            const Matrix<float>& weights_out = Y_hat.getWeights();
            const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
            
            const Matrix<float>& delta2 = H2.delta(delta_out, weights_out, dropped_out);
            H2.gradients(hidden1, dropped_out);
            const Matrix<float>& weights2 = H2.getWeights();
            const std::vector<int>& dropped2 = H2.getDroppedIdx();

            H1.delta(delta2, weights2, dropped2);
            H1.gradients(INPUT, dropped2);
            
            //update
            Y_hat.weights_update();
            H2.weights_update(dropped_out);
            H1.weights_update(dropped2);

            // THREASHOLD UPDATE
            Y_hat.updateThreashold(oh_target);

            if(train_counter % 10 == 0) COST_buff_train.push_back(Y_hat.getCost(CE));
            ++train_counter;

            printf("%6d | (%6d/%6d) %1.12f\n", epoch, selecter.getSize(), 60000, Y_hat.getCost(CE));
        }
    }
    // free training memory
    H1.freeGrad();
    H1.freeOptimizer(),
    H2.freeGrad();
    H2.freeOptimizer();
    Y_hat.freeGrad();
    Y_hat.freeOptimizer();
    Y_hat.freeTarget();

    const Matrix<float> threasholdClip = Y_hat.getThreasholdClip();

    // save training results
    output_parser_train.putData(COST_buff_train.begin(), COST_buff_train.end(), 1, COST_buff_train.size(), to_path_train, false, ',');

    Matrix<int> confusion_matrix(10, 10, 0);
    Selecter<int> selecter(0, input_parser_test.getRows(), 1);
    size_t test_counter = 0;
    std::vector<float> COST_buff_test;
    while(!selecter.empty()){
        // START: TEST PHASE
        std::vector<int> indices = selecter.pick_batch(1000);
        Matrix<float> INPUT(indices.size(), input_nodes);
        Matrix<float> oh_target(indices.size(), output_nodes, 0);
        order_hand(input_parser_train, INPUT, oh_target, indices);

        H1.reallocBatch(INPUT.getRows());
        H2.reallocBatch(INPUT.getRows());
        Y_hat.reallocBatch(INPUT.getRows());
        Y_hat.freeTarget();
        Y_hat.mallocTarget();
        
        H1.logit(INPUT);
        const Matrix<float>& hidden1 = H1.activate(SWISH);

        H2.logit(hidden1);
        const Matrix<float>& hidden2 = H2.activate(SWISH);

        Y_hat.logit(hidden2);
        const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);
        Matrix<float> output_test_cpy = output_test;
        for(int i = 0; i < indices.size(); ++i){
            for(int j = 0; j < output_nodes; ++j){
                output_test_cpy(i, j) = (output_test(i, j)>threasholdClip(0, j)) ? 1 : 0;
            }
        }
        if(test_counter % 10 == 0) COST_buff_test.push_back(Y_hat.getCost(CE));
        ++test_counter;
        // END: TEST PHASE
        // confusion matrix
        Matrix<int> results = output_test_cpy.vMaxIndex();
        Matrix<int> target_test = oh_target.vMaxIndex();

        //std::cout << results.hStack(target_test) << std::endl;
        //std::cout << "---" << std::endl;
        //std::cout << results(0, 0) << " | " << target_test(0, 0) << std::endl;
    
        for(int i = 0; i < results.getRows(); ++i){
            ++confusion_matrix(results(i, 0), target_test(i, 0));
        }

        printf("(%6d/%6d)\n", selecter.getSize(), 1000000);
    }
    output_parser_test.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');

    confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                to_path_confusion_matrix, true, ',');
}

void poker_hand_sgd(){
    // timer object
    Timer<nano_t> timer;
    // training / test data parers (input)
    std::string from_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-training-true.data";
    std::string from_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/input/poker/poker-hand-testing.data";
    TXTParser<int> input_parser_train(from_path_train);
    TXTParser<int> input_parser_test(from_path_test);
    std::cout << input_parser_train.getRows() << " | " << input_parser_train.getCols() << std::endl;
    std::cout << input_parser_test.getRows() << " | " << input_parser_test.getCols() << std::endl;
    // training / test data parers (output) -> stores training Losses
    std::string to_path_train = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_train_sgd.txt";
    std::string to_path_test = "/home/lucas/vim/cpp/projects/DeepNet/data/output/poker/cost_test_sgd.txt";
    TXTParser<int> output_parser_train(to_path_train);
    TXTParser<int> output_parser_test(to_path_test);
    // results
    std::string to_path_confusion_matrix = "data/output/poker/confusion_matrix_test_sgd.txt";
    TXTParser<int> confusion_parser(to_path_confusion_matrix);
    // model
    int batch_size = 64;
    int epochs = 30;
    int input_nodes = 52;
    int hidden_nodes_1 = 256;
    int hidden_nodes_2 = 256;
    int output_nodes = 10;
    FCLayer<float> H1(batch_size, hidden_nodes_1, input_nodes,    0);
    FCLayer<float> H2(batch_size, hidden_nodes_2, hidden_nodes_1, 0);
    Output<float> Y_hat(batch_size, output_nodes, hidden_nodes_2, 0);
    H1.mallocGrad();
    H1.optimizer(sgd, {0.0001});
    H2.mallocGrad();
    H2.optimizer(sgd, {0.0001});
    Y_hat.mallocGrad();
    Y_hat.mallocTarget();
    Y_hat.optimizer(sgd, {0.0001});
    Y_hat.mallocThreashold();
    
    std::vector<float> COST_buff_train;
    size_t train_counter = 0;
    for(int epoch = 0; epoch < epochs; ++epoch){
        Selecter<int> selecter(0, input_parser_train.getRows(), 1);
        if(epoch == 1) std::cout << selecter.getSize() << std::endl;
        while(!selecter.empty()){
            std::vector<int> indices = selecter.pick_batch(batch_size);
            Matrix<float> INPUT(indices.size(), input_nodes);
            Matrix<float> oh_target(indices.size(), output_nodes, 0);
            order_hand(input_parser_train, INPUT, oh_target, indices);
            if(indices.size() != batch_size) continue;
            H1.reallocBatch(INPUT.getRows());
            H2.reallocBatch(INPUT.getRows());
            Y_hat.reallocBatch(INPUT.getRows());
            Y_hat.freeTarget();
            Y_hat.mallocTarget();
        
            H1.logit(INPUT);
            const Matrix<float>& hidden1 = H1.activate(SWISH);

            H2.logit(hidden1);
            const Matrix<float>& hidden2 = H2.activate(SWISH);

            Y_hat.logit(hidden2);
            Y_hat.activate(SOFTMAX);
            //Y_hat.clip(0.01, 0.99);
            
            const Matrix<float>& delta_out = Y_hat.delta(CE, oh_target);
            Y_hat.gradients(hidden2);
            const Matrix<float>& weights_out = Y_hat.getWeights();
            const std::vector<int>& dropped_out = Y_hat.getDroppedIdx();
            
            const Matrix<float>& delta2 = H2.delta(delta_out, weights_out, dropped_out);
            H2.gradients(hidden1, dropped_out);
            const Matrix<float>& weights2 = H2.getWeights();
            const std::vector<int>& dropped2 = H2.getDroppedIdx();

            H1.delta(delta2, weights2, dropped2);
            H1.gradients(INPUT, dropped2);
            
            //update
            Y_hat.weights_update();
            H2.weights_update(dropped_out);
            H1.weights_update(dropped2);

            // THREASHOLD UPDATE
            Y_hat.updateThreashold(oh_target);

            if(train_counter % 10 == 0) COST_buff_train.push_back(Y_hat.getCost(CE));
            ++train_counter;

            printf("%6d | (%6d/%6d) %1.12f\n", epoch, selecter.getSize(), 60000, Y_hat.getCost(CE));
        }
    }
    // free training memory
    H1.freeGrad();
    H1.freeOptimizer(),
    H2.freeGrad();
    H2.freeOptimizer();
    Y_hat.freeGrad();
    Y_hat.freeOptimizer();
    Y_hat.freeTarget();

    const Matrix<float> threasholdClip = Y_hat.getThreasholdClip();

    // save training results
    output_parser_train.putData(COST_buff_train.begin(), COST_buff_train.end(), 1, COST_buff_train.size(), to_path_train, false, ',');
    
    Matrix<int> confusion_matrix(10, 10, 0);
    Selecter<int> selecter(0, input_parser_test.getRows(), 1);
    size_t test_counter = 0;
    std::vector<float> COST_buff_test;
    while(!selecter.empty()){
        // START: TEST PHASE
        std::vector<int> indices = selecter.pick_batch(1000);
        Matrix<float> INPUT(indices.size(), input_nodes);
        Matrix<float> oh_target(indices.size(), output_nodes, 0);
        order_hand(input_parser_train, INPUT, oh_target, indices);

        H1.reallocBatch(INPUT.getRows());
        H2.reallocBatch(INPUT.getRows());
        Y_hat.reallocBatch(INPUT.getRows());
        Y_hat.freeTarget();
        Y_hat.mallocTarget();
        
        H1.logit(INPUT);
        const Matrix<float>& hidden1 = H1.activate(SWISH);

        H2.logit(hidden1);
        const Matrix<float>& hidden2 = H2.activate(SWISH);

        Y_hat.logit(hidden2);
        const Matrix<float>& output_test = Y_hat.activate(SOFTMAX);
        Matrix<float> output_test_cpy = output_test;
        for(int i = 0; i < indices.size(); ++i){
            for(int j = 0; j < output_nodes; ++j){
                output_test_cpy(i, j) = (output_test(i, j)>threasholdClip(0, j)) ? 1 : 0;
            }
        }
        if(test_counter % 10 == 0) COST_buff_test.push_back(Y_hat.getCost(CE));
        ++test_counter;
        // END: TEST PHASE
        // confusion matrix
        Matrix<int> results = output_test_cpy.vMaxIndex();
        Matrix<int> target_test = oh_target.vMaxIndex();

        //std::cout << results.hStack(target_test) << std::endl;
        //std::cout << "---" << std::endl;
        //std::cout << results(0, 0) << " | " << target_test(0, 0) << std::endl;
    
        for(int i = 0; i < results.getRows(); ++i){
            ++confusion_matrix(results(i, 0), target_test(i, 0));
        }

        printf("(%6d/%6d)\n", selecter.getSize(), 1000000);
    }
    output_parser_test.putData(COST_buff_test.begin(), COST_buff_test.end(), 1, COST_buff_test.size(), to_path_test, false, ',');

    confusion_parser.putData(   confusion_matrix.begin(),   confusion_matrix.end(), \
                                confusion_matrix.getRows(), confusion_matrix.getCols(), \
                                to_path_confusion_matrix, true, ',');
}

int main(int argc, char **argv){
    //XOR();
    //mnist();
    XOR_softmax();
    
    /*
    #pragma omp parallel
    {
        //poker_hand_momentum();
        //poker_hand_adam();
        poker_hand_rmsprop();
        poker_hand_sgd();
    }
    */

    /*
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            // work1
            poker_hand_momentum();
        }

        #pragma omp section
        {
            // work2
            poker_hand_adam();
        }
        #pragma omp section
        {
            // work3
            poker_hand_rmsprop();
        }
        #pragma omp section
        {
            // work4
            poker_hand_sgd();
        }
    }
    */

    //std::string from_path = "data/input/poker/poker-hand-training-true.data";
    //std::string to_path = "data/output/poker/cost_train_filtered.txt";
    //filter_dupplicates<int>(from_path, to_path, 1000);
    
    return 0;
}