#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

template<typename T>
__global__ void kernel_naive_sgemm(T* A, T* B, T* C, int M, int N, int K) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (row < M && col < N) {
		T tmp = 0;
		for (int dotIdx = 0; dotIdx < K; dotIdx++) {
			tmp += A[row * K + dotIdx] * B[dotIdx * N + col];
		}
		C[row * N + col] = tmp;
	}
}

template<typename T>
void execute_naive_sgemm(T* A, T* B, T* C, int M, int N, int K) {
	int block_dim = 64;
	
	dim3 grid((M + block_dim - 1) / block_dim, (N + block_dim - 1) / block_dim, 1);
	dim3 block(block_dim, block_dim, 1);

	kernel_naive_sgemm<T><<<grid, block>>>(A, B, C, M, N, K);
	cudaDeviceSynchronize();
}

template<typename T, int bM, int bN, int bK>
__global__ void tiled_sgemm_kernel(T* A, T* B, T* C, int M, int N, int K) {
	
}
