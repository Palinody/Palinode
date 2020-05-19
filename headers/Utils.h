#include <chrono>

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

    void print() const{
        for(auto it=std::begin(_data); it != std::end(_data); ++it) std::cout << *it << ", ";
        std::cout << std::endl;
    }

private:
    std::unordered_set<T> _data;
};