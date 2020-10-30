#include <omp.h>
#include <chrono>
#include <thread>

using sec_t   = std::chrono::seconds;
using milli_t = std::chrono::milliseconds;
using micro_t = std::chrono::microseconds;
using nano_t  = std::chrono::nanoseconds;

template<typename duration_t>
class Timer{
	using clock_t = std::chrono::system_clock;

public:
	Timer() : _start{ getNow() }{ }
	
	uint64_t getNow() {
		return std::chrono::duration_cast<duration_t>(clock_t::now().time_since_epoch()).count();
	}
	void reset(){ _start = getNow(); }
	uint64_t elapsed() { return getNow() - _start; }
    template<typename time_T>
    void sleep(uint64_t t){
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int team_id = omp_get_team_num();
        int max_tid = omp_get_num_threads() - 1;
#endif
#ifdef _OPENMP
        printf("start: (%d|%d)\n", team_id, tid);
#endif 
        std::this_thread::sleep_for(time_T(t));
#ifdef _OPENMP
        printf("end: (%d|%d)\n", team_id, tid);
#endif    
    }
	
private:
	uint64_t _start;
};

#include <unordered_set>
#include <algorithm>
#include "PRNG.h"

template<typename T>
class Selecter{
public:
	Selecter(T from, T to, T stride){
		for(int i = from; i < to; i+=stride) _data.insert(i);
	}

    template<typename Iterator>
    Selecter(Iterator Begin, Iterator End){
        while(Begin != End) _data.insert(*(Begin++));
    }

    inline bool empty(){ return _data.empty(); }
    int getSize(){ return _data.size(); }

    inline T pick_unique(){
        uniform_dist<int> r(0, _data.size()-1);
        auto it = std::begin(_data);
        std::advance(it, r());
        _data.erase(*it);
        return *it;
    }

    std::vector<T> pick_batch(int n){
        std::vector<T> res;
        res.reserve(std::min(static_cast<int>(_data.size()), n));
        int i = 0;
        while(!_data.empty() && i < n){
            res.push_back(pick_unique());
            ++i;
        }
        return res;
    }

    void print() const {
        for(auto it = std::begin(_data); it != std::end(_data); ++it) std::cout << *it << ", ";
        std::cout << std::endl;
    }

private:
    std::unordered_set<T> _data;
};

#include <fstream>
#include <string>
#include <cstring>


class CPU_profile{
public:
    CPU_profile(){
        _n_cores = n_cores();
        _proc_freq = std::make_unique<uint32_t[]>(_n_cores);
    }

    inline uint16_t getCores() const { return _n_cores; }

    void update_freq(){ parse_proc_freq("MHz"); }

    inline uint32_t operator[](uint16_t idx) { return _proc_freq[idx]; }

    inline const uint32_t operator[](uint16_t idx) const { return _proc_freq[idx]; }

    inline void show_cpu_freq(char eol='\n') const {
        for(int n = 0; n < _n_cores; ++n) std::cout << " | " << _proc_freq[n];
        std::cout << " | " << eol << std::flush;
    }
    
private:
    uint16_t n_cores(){ return parse_value<uint16_t>("cores"); }

    /**
     * Parses one value
     * Return when the value has been found
    */
    template<typename T>
    T parse_value(const std::string& key){
        std::ifstream read("/proc/cpuinfo");
        //std::string pattern = "cores";
        for(std::string line; std::getline(read, line);){
            char *token = std::strtok(&line[0], "	: ");
            while(token != NULL){
                std::string token_str{ token };
                if(token_str == key){
                    token = std::strtok(NULL, "	: ");
                    std::string token_str_2{ token };
                    return static_cast<T>(std::stoul(token_str_2)); 
                }
                token = std::strtok(NULL, "	: ");
            }
        }
        return 0;
    }
    /**
     * Parses values w.r.t. each core
     * and puts them in container
    */
    void parse_proc_freq(const std::string& key){
        std::ifstream read("/proc/cpuinfo");
        //std::string pattern = "cores";
        uint32_t curr_proc_idx{ 0 };
        for(std::string line; std::getline(read, line);){
            char *token = std::strtok(&line[0], "	: ");
            while(token != NULL){
                std::string token_str{ token };
                if(token_str == key){
                    token = std::strtok(NULL, "	: ");
                    std::string token_str_2{ token };
                    _proc_freq[curr_proc_idx++] = static_cast<uint32_t>(std::stoul(token_str_2));
                }
                token = std::strtok(NULL, "	: ");
            }
        }
    }

    uint16_t _n_cores;
    std::unique_ptr<uint32_t[]> _proc_freq;
};