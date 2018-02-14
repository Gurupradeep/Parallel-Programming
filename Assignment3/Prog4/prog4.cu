#include <iostream>
#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <bits/stdc++.h>
using namespace std;

#define isValid(X, Y) (X >= 0 && Y>=0 && X < M && Y < N)

__global__ void image_bluring(float* a, float* b, int M, int N) {
	//__shared__ float[16][16][3];

	int global_x = blockDim.x * blockIdx.x + threadIdx.x ;
	int global_y = blockDim.y * blockIdx.y + threadIdx.y ;

	float channel1 = 0, channel2 = 0, channel3 = 0;
	int count = 0;
	for(int i = global_x - 1; i <= global_x + 1; i++) {
		for(int j = global_y - 1; j <= global_y + 1; j++){
                 	if(isValid(j,i)) {
//printf("%f\n",a[(j*N+i)*3]);
				channel1 += a[(j*N + i)*3];
				channel2 += a[(j*N + i)*3 + 1];
				channel3 += a[(j*N + i)*3 + 2];
				count++;  
			}
		}
	}

	channel1 = channel1 / count;
	channel2 = channel2 / count;
	channel3 = channel3 / count; 
//printf("%f\n",channel1);

	b[(global_y * N + global_x)*3 ] = channel1;
    b[(global_y * N + global_x)*3 + 1] = channel2;
    b[(global_y * N + global_x)*3 + 2] = channel3;
	printf("%f\n",b[(global_y*N+global_x)*3]);

}
void handle_error(cudaError_t error) {
	if (error != cudaSuccess) {
		std::cout << "Cuda Error. Exiting...";	
		exit(0);
	}
}

void initialise_matrix(int M, int N, float A[]) {
	for(int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k<3; k++) {
				A[(i*N + j )*3 + k ] = 1.0;
			}
		}
	}
}

void get_kernel(float K[][3]) {
	for(int i=0; i < 3; i++) {
		for(int j=0; j < 3;j++) {
			K[i][j] = 1.0/9.0;
		}
	}
}

void print(float a[])
{
	for(int k=0;k<3;k++)
	{
		for(int i=0;i<16;i++)
		{
			for(int j=0;j<16;j++)
				cout<<a[(i*16+j)*3+k]<< "  " ;
			cout<<endl;
		}
		cout<<endl;
	}
	cout<<endl<<endl;
}

int main() {
	float image[16*16*3];
	float result[16*16*3];
	float kernel[3][3];
	initialise_matrix(16,16,image);
	get_kernel(kernel);
	float *I, *R;
	size_t size = 16 * 16 * 3 * sizeof(float);
	handle_error(cudaMalloc((void**) &I, size));
	handle_error(cudaMalloc((void**) &R, size));

	cudaMemcpy(I,image,size,cudaMemcpyHostToDevice);

	dim3 grid_dim(1,1,1);
	dim3 block_dim(16,16,1);

	image_bluring<<<grid_dim, block_dim>>> (I, R, 16, 16);

	cudaMemcpy(result, R,size,cudaMemcpyDeviceToHost);
	print(result);
	
} 

