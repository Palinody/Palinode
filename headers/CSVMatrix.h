#pragma once

#include <cstdlib> // c_str()
#include <fstream> // getline(ifstream, string)
#include <sstream> // stringstream
#include <string>
#include "Matrix.h"

template<typename T>
class CSVMatrix : public Matrix<T>{
public:
    CSVMatrix(std::string from_path);
    CSVMatrix(const Matrix<T>& other);

    int getRows() const;
    int getCols() const;

    void fillData(int n_samples=0);
    void putFile(std::string to_file, int row_idx_start, int row_idx_end, std::string del=",");


private:
    std::string _from_path = "NONE";
    std::string _to_path = "NONE";
    int _from_cursor_index = 0;
    int _to_cursor_index = 0;
};

template<typename T>
CSVMatrix<T>::CSVMatrix(std::string path) : _from_path{path}{
}

template<typename T>
CSVMatrix<T>::CSVMatrix(const Matrix<T>& other){
    *this = other;
}
/*
 * reads the entire csv file
 * the number of lines gives m_rows
 */
template<typename T>
int CSVMatrix<T>::getRows() const{
    assert(_from_path != "NONE");
    int rows = 0;
    std::ifstream file(_from_path);
    std::string line;
    while(std::getline(file, line)) ++rows;
    return rows;
}
/*
 * we read one line of the csv file
 * the number of ',' gives m_cols
 */
template<typename T>
int CSVMatrix<T>::getCols() const{
    assert(_from_path != "NONE");
    std::ifstream file(_from_path);
    std::string line;
    std::getline(file, line);
    return 1 + std::count(line.begin(), line.end(), ',');
}

/*
 * specify number of samples to store
 * object saves the previous sample index
 * if n_samples not specified by user:
 *      store entire dataset
 */
template<typename T>
void CSVMatrix<T>::fillData(int n_samples){
    // dont forget to deactivate this since it counts rows
    assert(n_samples >= 0 && n_samples <= this->getRows());

    this->_rows = n_samples;
    this->_cols = this->getCols();
    this->_matrix = std::make_unique<T[]>(this->_rows * this->_cols);
    
    std::ifstream file(this->_from_path);
    //std::stringstream file(this->_from_path);
    std::string data;
    for(int i = 0; i < this->_rows; ++i){
        for(int j = 0; j < this->_cols; ++j){
            std::getline(file, data, ',');
            std::cout << data << ", ";
            if(!(data.c_str())) continue;
            //if(!file) continue;
            this->_matrix[j+i*this->_cols] = static_cast<T>(atof(data.c_str()));
        }
    }
    std::cout << std::endl;
    std::cout << std::endl;

    //this->_from_cursor_index = this->_rows;
}

template<typename T>
void CSVMatrix<T>::putFile(std::string to_file, int row_idx_start, int row_idx_end, std::string del){
    _to_path = to_file;
    std::cout << _to_path << std::endl;

    std::fstream file;
    // Open the file in truncate mode if first line else in Append Mode
    file.open(this->_to_path, std::ios::out | (this->_to_cursor_index ? std::ios::app : std::ios::trunc));

    // Iterate over the range and add each lement to file seperated by delimeter.
    for (int i = row_idx_start; i < row_idx_end; ++i){
        for(int j = 0; j < this->_cols; ++j){
            std::string val_s = std::to_string(this->_matrix[j+i*this->_cols]);
            file << val_s;
            file << del;
        }
        file << "\n";
        this->_to_cursor_index = i;
    }
    file.close();
}