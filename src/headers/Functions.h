#pragma once

#include "Matrix.h"

namespace func2D{
	/**
	 * Functions with 2 parameters perform:
	 * 		destination = f(source)
	 * Functions with 1 parameter perform:
	 * 		source = f(source)
	*/
	template<typename T>
	void abs(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin(); 
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = (*it_src > 0) ? (*it_src) : -(*it_src);
		}
	}

	template<typename T>
	void abs(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = (*it_src > 0) ? (*it_src) : -(*it_src);
		}
	}

	template<typename T>
	void pow(Matrix<T>& dest, const Matrix<T>& src, int power=2){
		auto it_dest = dest.begin(); 
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest)
			*(it_dest) = std::pow((*it_src), power);
		//dest = src;
		//for(int i = 0; i < power-1; ++i) dest *= src;
	}

	template<typename T>
	void pow(Matrix<T>& src, int power=2){
		Matrix<T> cpy = src;
		auto it_src = src.begin(), it_cpy = cpy.begin();
		for(; it_src != src.end(); ++it_src, ++it_cpy)
			(*it_src) *= std::pow((*it_cpy), power);
	}

	template<typename T>
	void sqrt(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = std::tanh((*it_src));
		}
	}

	template<typename T>
	void sqrt(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = std::sqrt((*it_src));
		}
	}

	template<typename T>
	void log(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin(); 
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = std::log(*it_src);
		}
	}

	template<typename T>
	void log(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = std::log(*it_src);
		}
	}

	template<typename T>
	void exp(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = std::exp(*it_src);
		}
	}

	template<typename T>
	void exp(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = std::exp(*it_src);
		}
	}

	template<typename T>
	void relu(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = std::max(static_cast<T>(0), *it_src); //(*it_src > 0) ? (*it_src) : 0;
		}
	}

	template<typename T>
	void relu(Matrix<T>& src){
		for(auto it_src = src.begin(); it_src != src.end(); ++it_src){
			(*it_src) = std::max(static_cast<T>(0), *it_src); //((*it_src) > 0) ? (*it_src) : 0;
		}
	}

	template<typename T>
	void tanh(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = std::tanh((*it_src));
		}
	}

	template<typename T>
	void tanh(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = std::tanh((*it_src));
		}
	}

	template<typename T>
	void sigmoid(Matrix<T>& dest, const Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = one / (one + std::exp(-(*it_src)));
		}
	}

	template<typename T>
	void sigmoid(Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = one / (one + std::exp(-(*it_src)));
		}
	}

	template<typename T>
	void swish(Matrix<T>& dest, const Matrix<T>& src){
		sigmoid(dest, src);
		dest *= src;
	}

	template<typename T>
	void swish(Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) *= one / (one + std::exp(-(*it_src)));
		}
	}
}

namespace deriv2D{
	/**
	 * Functions with 2 parameters perform:
	 * 		destination = f(source)
	 * Functions with 1 parameter perform:
	 * 		source = f(source)
	 * To get the derivative of a function y = f(x)
	 * one must sometimes provide the original data x
	 * or the transformed data y. Read the comments
	 * to know which one to use (x or y or both)
	*/
	// provide x
	template<typename T>
	void abs(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin(); 
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = (*it_src > 0) ? 1 : -1;
		}
	}
	// provide x
	template<typename T>
	void abs(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = (*it_src > 0) ? 1 : -1;
		}
	}
	// provide x
	template<typename T>
	void pow(Matrix<T>& dest, const Matrix<T>& src, int power=2){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		int exponent = power-1;
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = power * std::pow(*it_src, exponent);
		}
	}
	// provide x
	template<typename T>
	void pow(Matrix<T>& src, int power=2){
		auto it_src = src.begin();
		int exponent = power-1;
		for(; it_src != src.end(); ++it_src){
			(*it_src) = power * std::pow(*it_src, exponent);
		}
	}
	// provide x
	template<typename T>
	void sqrt(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = 1 / (2 * std::sqrt(*it_src) + 1e-8f);
		}
	}
	// provide x
	template<typename T>
	void sqrt(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = 1 / (2 * std::sqrt(*it_src) + 1e-8f);
		}
	}
	// provide x
	template<typename T>
	void log(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = 1 / ((*it_src) + 1e-8f);
		}
	}
	// provide x
	template<typename T>
	void log(Matrix<T>& src){
		// not checking if T is_integral because it
		// makes no sense to use the function with integers
		T type_min = std::numeric_limits<T>::min();
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = one / ((*it_src) + type_min);
		}
	}
	// provide x
	template<typename T>
	void exp(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			int sgn = (static_cast<T>(0) < (*it_src)) - ((*it_src) < static_cast<T>(0));
			(*it_dest) = sgn * std::exp(*it_src);
		}
	}
	// provide x
	template<typename T>
	void exp(Matrix<T>& src){
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			int sgn = (static_cast<T>(0) < (*it_src)) - ((*it_src) < static_cast<T>(0));
			(*it_src) = sgn * std::exp(*it_src);
		}
	}
	// provide x OR y
	template<typename T>
	void relu(Matrix<T>& dest, const Matrix<T>& src){
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = ((*it_src) > 0) ? 1 : 0;
		}
	}
	// provide x OR y
	template<typename T>
	void relu(Matrix<T>& src){
		for(auto it_src = src.begin(); it_src != src.end(); ++it_src){
			(*it_src) = ((*it_src) > 0) ? 1 : 0;
		}
	}
	// provide y
	template<typename T>
	void tanh(Matrix<T>& dest, const Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin(), it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = one - (*it_src) * (*it_src);
		}
	}
	// provide y
	template<typename T>
	void tanh(Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = one - (*it_src) * (*it_src);
		}
	}
	// provide y
	template<typename T>
	void sigmoid(Matrix<T>& dest, const Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		auto it_dest = dest.begin();
		for(; it_src != src.end(); ++it_src, ++it_dest){
			(*it_dest) = (*it_src) * (one - (*it_src));
		}
	}
	// provide y
	template<typename T>
	void sigmoid(Matrix<T>& src){
		T one = static_cast<T>(1);
		auto it_src = src.begin();
		for(; it_src != src.end(); ++it_src){
			(*it_src) = (*it_src) * (one - (*it_src));
		}
	}
	// provide x AND y
	template<typename T>
	void swish(Matrix<T>& dest, const Matrix<T>& src_x, const Matrix<T>& src_y){
		// swish + sigmoid(z)*(1 - swish)
		// src_y + func2D::sigmoid(sig_mat, src_x) * (1 - src_y)
		T one = static_cast<T>(1);
		Matrix<T> sig_mat(src_x.getRows(), src_x.getCols(), 0);
        func2D::sigmoid(sig_mat, src_x);
		sig_mat *= one + src_y * (-one);
		dest = src_y + sig_mat;
		//dest = src_y + sig_mat * (one - src_y);
	}
}