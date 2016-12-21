#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void init(BASE_TYPE *a, const int N)
{
    int ind = N * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    a[ind] = ind;
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
            printf("%3.0f ", a[i *N + j]);

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
    *grid = dim3(N);
    *block = dim3(N, 1, 1);
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
    cudaEvent_t start, stop;
    float h2d_cp_span, d2h_cp_span, k_span;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    BASE_TYPE *host_a = gen_array(N);
    BASE_TYPE *dev_a;

    if (host_a == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    print_array(host_a, N);

    cudaEventRecord(start, 0);

    try
    {
        cuda_init_array(&dev_a, NULL, size);
    }
    catch (cudaError_t err)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_cp_span, start, stop);

    init<<<blocksPerGrid, threadsPerBlock>>>(dev_a, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&k_span, start, stop);


    err = cudaMemcpy(host_a, dev_a, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_cp_span, start, stop);

    printf("Copy form host to device time: %.2f milliseconds\n", h2d_cp_span);
    printf("Run kernel time: %.2f milliseconds\n", k_span);
    printf("Copy form device to host time: %.2f milliseconds\n", d2h_cp_span);

    print_array(host_a, N);

    cudaFree(dev_a);

    delete[] host_a;

    return 0;
}