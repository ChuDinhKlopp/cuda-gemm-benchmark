#include <random> // initialize matrix value
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

template<typename T, const size_t n_row, const size_t n_col>
class BlockTile {
private:
	size_t n_row_;
	size_t n_col_;
	T* pos_;

public:
	__shared__ T block[n_row * n_col];

	__device__ BlockTile(Matrix<T> &mat, uint blockOffset):
		n_row_(n_row), 
		n_col_(n_col),
		pos_(mat.device_ptr() + blockOffset) {}

	__device__ void load() {
		
	}

	__device__ void next(uint stride) {
	
	}
};

template<typename T>
class WarpTile {
private:

public:
};


template<typename T>
class Matrix {
private:
	T *host_ptr_;
	T *device_ptr_;
	size_t n_row_;
	size_t n_col_;
	size_t byte_size_;

public:
	Matrix(int rows, int cols): 
		n_row_(rows), 
		n_col_(cols)
	{
		byte_size_ = n_row_ * n_col_ * sizeof(T);
		host_ptr_ = (T*)malloc(byte_size_);
		cudaMalloc((T **)&device_ptr_, byte_size_);
	}

	T* host_ptr() {return host_ptr_;}
	T* device_ptr() {return device_ptr_;}
	size_t byte_size() {return byte_size_;}
	
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

	void copyData(cudaMemcpyKind kind) {
		if (kind == cudaMemcpyHostToDevice) {
			cudaError_t err = cudaMemcpy(device_ptr_, host_ptr_, byte_size_, kind);
			if (err != cudaSuccess) {
				fprintf(stderr, "Failed to copy from host to device (error code: %s)", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}
		else if (kind == cudaMemcpyDeviceToHost) {
			cudaError_t err = cudaMemcpy(device_ptr_, host_ptr_, byte_size_, kind);
				fprintf(stderr, "Failed to copy from device to host (error code: %s)", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
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
}; // class Matrix



int main() {
	uint M = 4096;
	uint N = 4096;
	uint K = 4096;
	Matrix<float> A(M, K);
	Matrix<float> B(K, N);
	Matrix<float> C(M, N);
	A.init(0, 10);
	B.init(0, 10);
	A.print();
	A.copyData(cudaMemcpyHostToDevice);
	B.copyData(cudaMemcpyHostToDevice);
	// execute kernel
	// C.copyData(cudaMemcpyDeviceToHost);
	return 0;
}
