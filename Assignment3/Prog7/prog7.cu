#include <bits/stdc++.h>
#include <cuda.h>

#define N 15
#define M 4096
__global__ void histogram_creation(int *A, int *hist, int no_of_threads) {
	
	int global_x = blockDim.x * blockIdx.x + threadIdx.x;
	__shared__ int local_hist[N+1];

	for(int i = threadIdx.x; i<=N; i = i + (blockDim.x ) ){
		local_hist[i] = 0;
	}
	__syncthreads();
	
	for(int i = global_x; i <= M; i = i + (blockDim.x * no_of_threads)) {
		atomicAdd(&local_hist[A[i]],1);
	}
	__syncthreads();
	
	for(int i = threadIdx.x ; i <= N; i = i + (blockDim.x) ) {
		atomicAdd(&hist[i],local_hist[i]);
		printf("%d histogram_local %d \n",local_hist[i],i);
	}
	__syncthreads();
	
}
__global__ void histogram_saturation(int *hist) {
	int index = threadIdx.x;
	if(hist[index] > 127) {
		hist[index] = 127;
	}
}
void handle_error(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("Cuda Error. Exiting....\n");
		exit(0);
	}
}

void init_matrix( int A[], long long n) {
	for (long long i = 0; i< n; i++) {
		A[i] = rand() % 4096 + 1;
	}
}

int main() {
	int A[M];
	int hist[N +1];
	
	for(int i=0; i < N + 1; i++) {
		hist[i] = 0;
	}
	init_matrix(A, M);
	int *deviceA;
	int *deviceHist;
	
	size_t size_histogram = (N + 1) * sizeof(int);
	size_t size_array = M * sizeof(int);
	
	handle_error(cudaMalloc((void**) &deviceA, size_array));
	handle_error(cudaMalloc((void**) &deviceHist, size_histogram));
	
	cudaMemcpy(A, deviceA, size_array, cudaMemcpyHostToDevice);
	cudaMemcpy(hist, deviceHist, size_histogram, cudaMemcpyHostToDevice);
	dim3 grid_dim(256,1,1);
	dim3 block_dim(256,1,1);
	
	histogram_creation<<<grid_dim, block_dim>>>(deviceA, deviceHist,256);
	
	cudaMemcpy(hist, deviceHist, size_histogram, cudaMemcpyDeviceToHost);
	histogram_saturation<<<1,4096>>>(deviceHist);
	
	cudaMemcpy(hist, deviceHist, size_histogram, cudaMemcpyDeviceToHost);
	
	for(int i=0;i<=4096; i++)
	{
		printf("%d\n", hist[i]);
	}
} 

