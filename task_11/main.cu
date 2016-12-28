#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void mult(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *c, const int N, const int M)
{
    int i = N * (blockDim.y * blockIdx.y + threadIdx.y);
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    
    BASE_TYPE sum = 0;
    
    for (int k = 0; k < N; k++)
    {
        sum += a[i + k] * b[k * M + j];
    }
    
    int id = M * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    
    c[id] = sum;
}

BASE_TYPE* gen_array(const int N)
{
    BASE_TYPE *a = new BASE_TYPE[N * N];

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            a[i * N + j] = i * N + j;
    }

    return a;
}

void print_array(BASE_TYPE *a, const int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
            printf("%5.0f ", a[i *N + j]);

        printf("\n");
    }

    printf("\n");
}

void cuda_init_array(BASE_TYPE **dev, const BASE_TYPE *host, const size_t size)
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

void cuda_init_grid_and_block(dim3 *grid, dim3 *block, const int N)
{
    *grid = dim3(1);
    *block = dim3(N, N, 1);
    printf("Block %d %d %d\n", block->x, block->y, block->z);
    printf("Grid %d %d %d\n", grid->x, grid->y, grid->z);
}

int main()
{
    const int N = 10;
    const size_t size = N * N * sizeof(BASE_TYPE);
    cudaError_t err;

    dim3 threadsPerBlock, blocksPerGrid;
    cuda_init_grid_and_block(&blocksPerGrid, &threadsPerBlock, N);

    BASE_TYPE *host_a = gen_array(N), *host_b = gen_array(N);
    BASE_TYPE *dev_a, *dev_b, *dev_c;

    if (host_a == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    print_array(host_a, N);
    print_array(host_b, N);

    try
    {
        cuda_init_array(&dev_a, host_a, size);
        cuda_init_array(&dev_b, host_b, size);
        cuda_init_array(&dev_c, NULL, size);
    }
    catch (cudaError_t err)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    mult<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N, N);

    err = cudaMemcpy(host_a, dev_c, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    print_array(host_a, N);

    cudaFree(dev_a);
    cudaFree(dev_b);

    delete[] host_a;

    return 0;
}