#include <bits/stdc++.h>
#include <cuda.h>

#define M 64
#define N 64
#define TILE_WIDTH 16

__global__ void tiled_matrix_multiplication(int *A, int *B, int *C) {

	__shared__ int As[TILE_WIDTH][TILE_WIDTH];
	__shared__ int Bs[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	
	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;
	
	int res = 0;
	
	for(int i = 0; i < M/TILE_WIDTH; i++) {
		As[ty][tx] = A[row * M + (i*TILE_WIDTH + tx)];
		Bs[ty][tx] = B[(i*TILE_WIDTH + ty)* M + col];
		
		__syncthreads();
		
		for(int j = 0; j < TILE_WIDTH; j++) {
			res += As[ty][j] + Bs[j][tx];
		}
	
		__syncthreads();
	}
	
	C[row * M + col] = res;
		
}
void handle_error(cudaError_t error) {
	if(error != cudaSuccess) {
		printf("Cuda Error. Exiting....");
		exit(0);
	}
}

void initialise_matrix(int A[])
{
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < M; j++) {
			A[i * M + j] = i * j;
		}
	}
}
void print_matrix(int A[]) {
	for(int i = 0; i < M ;i++) {
		for(int j = 0; j < M; j++) {
			printf("%d ", A[i*M + j]);
		}
		printf("\n");
	}
}
int main() {
	int A[M*M];
	int B[M*M];
	int C[M*M];
	initialise_matrix(A);
	initialise_matrix(B);
	
	int *deviceA;
	int *deviceB;
	int *deviceC;
	
	size_t size = M*M*sizeof(int);
	handle_error(cudaMalloc((void**) &deviceA, size));
	handle_error(cudaMalloc((void**) &deviceB, size));
	handle_error(cudaMalloc((void**) &deviceC, size));
	
	cudaMemcpy(deviceA, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, B, size, cudaMemcpyHostToDevice);
	
	dim3 grid_dim(M/TILE_WIDTH,M/TILE_WIDTH,1);
	dim3 block_dim(TILE_WIDTH,TILE_WIDTH,1);
	
	tiled_matrix_multiplication<<<grid_dim, block_dim>>>(deviceA, deviceB, deviceC);
	
	cudaMemcpy(C, deviceC, size, cudaMemcpyDeviceToHost);

	print_matrix(A);
	
	print_matrix(B);
	
	print_matrix(C);	
	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
}
	

