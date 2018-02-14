#include <iostream>
#include <math.h> 
#include <cuda.h>
#include <assert.h>

#define N 65564

__global__ void sum(float *a, float *b, float *c) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if (index < N) {
		c[index] = a[index] + b[index];
	}
}

void handle_error(cudaError_t error) {
	if (error != cudaSuccess) {
		std::cout << "Cuda Error. Exiting..";
		exit (0);
	}
}

int main() {
	float a[N], b[N], c[N];
	float *device_a, *device_b, *device_c;
	for (int i = 0; i < N; i++) {
		a[i] = (i+1) * 1.0 / 2;
		b[i] = (i+3) * 1.0 / 3;
	}
	handle_error(cudaMalloc((void **) &device_a, N * sizeof(float)));
	handle_error(cudaMalloc((void **) &device_b, N * sizeof(float)));
	handle_error(cudaMalloc((void **) &device_c, N * sizeof(float)));

	cudaMemcpy(device_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
	sum<<<ceil(N/1024.0), 1024>>>(device_a, device_b, device_c);
	cudaMemcpy(c, device_c, N * sizeof(N), cudaMemcpyDeviceToHost);
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}
	std::cout << "Successful.\n";
	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);
	return 0;
}

