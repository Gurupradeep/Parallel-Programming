#include <iostream>
#include <stdio.h>
#define M 16
#define N 16
#define BLOCK_SIZE 16
#define BLUR_SIZE 1
using namespace std;

__global__ void add(float *cudaA, float *kernel, float *cudaResult)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int gid = idy * N + idx;

  __shared__ float blockData[BLOCK_SIZE + 2 * BLUR_SIZE][BLOCK_SIZE + 2 * BLUR_SIZE][3];

  int x = idx - BLUR_SIZE;
  int y = idy - BLUR_SIZE;

  if(x >= 0 && y >= 0)
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.x][threadIdx.y][k] = cudaA[(gid - BLUR_SIZE - BLUR_SIZE * N)*3 + k];
  else
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.x][threadIdx.y][k] = 0;

  x = idx + BLUR_SIZE;
  y = idy - BLUR_SIZE;

  if(x < N && y >= 0)
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y][threadIdx.x + 2 * BLUR_SIZE][k] = cudaA[(gid + BLUR_SIZE - BLUR_SIZE * N)*3 + k];
  else
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y][threadIdx.x + 2 * BLUR_SIZE][k] = 0;

  x = idx - BLUR_SIZE;
  y = idy + BLUR_SIZE;

  if(x >= 0 && y < N)
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x][k] = cudaA[(gid - BLUR_SIZE + BLUR_SIZE * N)*3 + k];
  else
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x][k] = 0;

  x = idx + BLUR_SIZE;
  y = idy + BLUR_SIZE;

  if(x < N && y < N)
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x + 2 * BLUR_SIZE][k] = cudaA[(gid + BLUR_SIZE + BLUR_SIZE * N)*3 + k];
  else
    for(int k = 0; k < 3; k++)
      blockData[threadIdx.y + 2 * BLUR_SIZE][threadIdx.x + 2 * BLUR_SIZE][k] = 0;

  __syncthreads();
  for(int k = 0; k < 3; k++)
  {
    for(int i = -BLUR_SIZE; i <= BLUR_SIZE; i++)
      for(int j = -BLUR_SIZE; j <= BLUR_SIZE; j++)
      {
        cudaResult[gid * 3 + k] += blockData[threadIdx.y + BLUR_SIZE + i][threadIdx.x + BLUR_SIZE + j][k] * kernel[(BLUR_SIZE - i) * (2 * BLUR_SIZE + 1) + (BLUR_SIZE - j)];
      }
  }
}

__device__ __host__ void print(float *result)
{
  for(int k = 0; k < 3; k++)
  {
    for(int i = 0; i < N; i++)
    {
      for(int j = 0; j < N; j++)
        printf("%f ",result[(i*N + j)*3 + k]);
      printf("\n");
    }
    printf("\n");
  }
}

int main()
{
  float *A = new float[N*N*3];
  float *result = new float[N*N*3];
  float *cudaKernel = new float[9];
  float kernel[9] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9};
  //float kernel[9] = {0,1,0,0,0,0,0,0,0};
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      for(int k = 0; k < 3; k++)
      {
        A[(i*N + j)*3 + k] = 1;
        result[(i*N + j)*3 + k] = 0;
      }
  float *cudaA,*cudaResult;
  cudaMalloc((void**)&cudaA,N*N*3*sizeof(float));
  cudaMalloc((void**)&cudaResult,N*N*3*sizeof(float));
  cudaMalloc((void**)&cudaKernel,9*sizeof(float));

  cudaMemcpy(cudaA,A,N*N*3*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(cudaKernel,kernel,9*sizeof(float),cudaMemcpyHostToDevice);

  dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
  dim3 blocks(N/BLOCK_SIZE,N/BLOCK_SIZE);
  add<<<blocks,block_size>>>(cudaA, cudaKernel, cudaResult);

  cudaDeviceSynchronize();

  cudaMemcpy(result,cudaResult,N*N*3*sizeof(float),cudaMemcpyDeviceToHost);
  cudaFree(cudaA);
  cudaFree(cudaResult);
  print(result);

  free(A);
  free(result);
}
