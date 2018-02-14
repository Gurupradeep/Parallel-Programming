#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <math.h>

#define N 100

__global__ void sum(int a[][N], int b[][N], int c[][N]) {
	int row_index = blockDim.y * blockIdx.y + threadIdx.y;
	int col_index = blockDim.x * blockIdx.x + threadIdx.x;
	if (row_index < N && col_index < N) {
		c[row_index][col_index] = a[row_index][col_index] + b[row_index][col_index];
	}
}

void handle_error(cudaError_t error) {
	if (error != cudaSuccess) {
		std::cout << "Cuda Error. Exiting...";	
		exit(0);
	}
}

int main() {
	int a[N][N], b[N][N], c[N][N];
	int (*device_a)[N], (*device_b)[N], (*device_c)[N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			a[i][j] = i + j;
			b[i][j] = 2*i + j;
		}
	}
	handle_error(cudaMalloc((void **)&device_a, N * N * sizeof(int)));
	handle_error(cudaMalloc((void **)&device_b, N * N * sizeof(int)));
	handle_error(cudaMalloc((void **)&device_c, N * N * sizeof(int)));
	
	cudaMemcpy(device_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
	
	dim3 thread_size(8, 8);
	dim3 block_grid_size(ceil(N/8.0), ceil(N/8.0));
	
	sum<<<block_grid_size, thread_size>>>(device_a, device_b, device_c);
	
	cudaMemcpy(c, device_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			assert(c[i][j] == a[i][j] + b[i][j]);
		}
	}
	std::cout << "Successful..";
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	return 0;
}

