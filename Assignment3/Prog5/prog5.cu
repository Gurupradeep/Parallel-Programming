#include <iostream>
#include <cuda.h>
#include <time.h>
#include <math.h>

#define row 100
#define col 100

void handle_error(cudaError_t error) {
	if (error != cudaSuccess) {
		std::cout << "Error in cuda. Waiting...";
		exit(0);
	}
}

int get_rand_in_range() {	
	return rand()%256;
}

__global__ void convert(float *input_image, float *output_image, int no_of_threads) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	for(int i = index; i <= (row*col); i = i + (blockDim.x*no_of_threads)) {
		float r = input_image[3*i];
		float g = input_image[3*i + 1];
		float b = input_image[3*i + 2] ;
		
		output_image[index] = (0.21*r + 0.71*g + 0.07*b);
	}
}

void initialise_matrix(float A[], int m, int n) {	
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			for(int k = 0; k < 3; k++ ) {
				A[(i*n + j)*3 + k] = get_rand_in_range();
			}
		}
	}
}
int main() {
	srand(time(NULL));
	float image[row * col * 3], gray_image[row * col];
	float *device_image, *output_image;
	handle_error(cudaMalloc((void **)&device_image, row * col * 3 * sizeof(float)));
	handle_error(cudaMalloc((void **)&output_image, row * col * sizeof(float)));
	
	initialise_matrix(image, row, col);
	
	cudaMemcpy(device_image, image, row * col * 3 * sizeof(float), cudaMemcpyHostToDevice);
	
	dim3 grid_dim(256,1,1);
	dim3 block_dim(256,1,1);
	convert<<<grid_dim, block_dim>>>(device_image, output_image, 256);

	cudaMemcpy(gray_image, output_image, row * col * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(device_image);
	cudaFree(output_image);

	return 0;
}

