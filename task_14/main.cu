#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime_api.h>

#define BASE_TYPE float
#define INT_TYPE unsigned int

__global__ void monteKarlo(const BASE_TYPE h, INT_TYPE *result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    BASE_TYPE x = h * i;
    BASE_TYPE y = h * j;
    
    BASE_TYPE r = sqrtf(x * x + y * y);
    
    if (r < 100.0)
        atomicAdd(result, (INT_TYPE)1);
}

void cuda_init_array(INT_TYPE **dev, const INT_TYPE *host, const size_t size)
{
    cudaError_t err;
    err = cudaMalloc((void **)dev, size);
    if (err != cudaSuccess)
        throw err;

    if (host != NULL)
    {
        err = cudaMemcpy(*dev, host, size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
            throw err;
    }
}

void init_grid_and_block(dim3 *grid, dim3 *block, const int block_size, const int N)
{
    *grid = dim3(N / block_size, N / block_size, 1);
    *block = dim3(block_size, block_size, 1);
    printf("Block (%d, %d, %d)\n", block->x, block->y, block->z);
    printf("Grid  (%d, %d, %d)\n", grid->x, grid->y, grid->z);
}

int main()
{
    const int N = 1024;
    const int block_size = 16;

    dim3 blockDim, gridDim;
    init_grid_and_block(&blockDim, &gridDim, block_size, N);

    BASE_TYPE h = 100.0 / N;
    printf("%f\n", h);

    INT_TYPE result = 0;
    INT_TYPE *dev_result;

    try
    {
        cuda_init_array(&dev_result, &result, sizeof(INT_TYPE));
    }
    catch (cudaError_t err)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    monteKarlo<<<blockDim, gridDim>>>(h, dev_result);
    cudaMemcpy(&result, dev_result, sizeof(INT_TYPE), cudaMemcpyDeviceToHost);
    printf("%f\n", (double) 4 * result / (N * N));

    cudaFree(dev_result);

    return 0;
}