#include <random> // initialize matrix value
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

template<typename T>
class Matrix {
private:
	T *host_ptr_;
	T *device_ptr_;
	int n_row_;
	int n_col_;
public:
	Matrix(int rows, int cols): 
		n_row_(rows), 
		n_col_(cols)
	{
		host_ptr_ = (T*)malloc(n_row_ * n_col_ * sizeof(T));
	}

	T* host_ptr() {return host_ptr_;}
	T* device_ptr() {return device_ptr_;}
	
	void init(T min, T max) {
		std::random_device rd;
		std::mt19937 mt(rd());

		if (typeid(T) == typeid(int)) { 					// int
			std::uniform_int_distribution<int> dist(min, max);
			for (int i = 0; i < n_row_ * n_col_; ++i) {
				host_ptr_[i] = dist(mt);
			}
		}
		else if (typeid(T) == typeid(float)) { 				// fp32
			std::uniform_real_distribution<float> dist(min, max);
			for (int i = 0; i < n_row_ * n_col_; ++i) {
				host_ptr_[i] = dist(mt);
			}
		}
	}

	void print() {
		for (int row = 0; row < n_row_; ++row) {
			for (int col = 0; col < n_col_; ++col) {
				std::cout << host_ptr_[row * n_col_ + col] << " ";
			}
			printf("\n");
		}
	}

	void copyData(cudaMemcpyKind kind) {
		size_t byte_size = sizeof(T) * n_row_ * n_col_;
		if (kind == cudaMemcpyHostToDevice) {
			cudaError_t err = cudaMemcpy(device_ptr_, host_ptr_, byte_size, kind);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to copy from host to device (error code: %s)", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}
		else if (kind == cudaMemcpyDeviceToHost) {
			cudaError_t err = cudaMemcpy(device_ptr_, host_ptr_, byte_size, kind);
				fprintf(stderr, "Failed to copy from device to host (error code: %s)", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
		}
	}

}; // class matrix

int main() {
	Matrix<float> A(4, 4);
	A.init(0, 10);
	A.print();
	return 0;
}
