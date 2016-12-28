#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void add(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *result, const int N)
{
    int threads_count = blockDim.x * gridDim.x;
    int elem_per_thread = N / threads_count;
    int k = (blockIdx.x * blockDim.x + threadIdx.x) * elem_per_thread;
    
    for (int i = k; i < k + elem_per_thread; i++)
    {
        result[i] = a[i] + b[i];
    }
}

BASE_TYPE* gen_array(const int N)
{
    BASE_TYPE *a = new BASE_TYPE[N];

    for (int i = 0; i < N; i++)
        a[i] = rand() % 100;

    return a;
}

void print_array(const BASE_TYPE *a, const int N)
{
    for (int i = 0; i < N; i++)
        printf("%3.0f ", a[i]);

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

void cuda_init_grid_and_block(dim3 *grid, dim3 *block, const int N, const int threads_per_block)
{
    int blocks_count = N / threads_per_block;
    *grid = dim3(blocks_count);
    *block = dim3(threads_per_block);
    printf("Block %d %d %d\n", block->x, block->y, block->z);
    printf("Grid %d %d %d\n", grid->x, grid->y, grid->z);
}

int main()
{
    srand(time(NULL));

    const int N = 32768;
    const size_t size = N * sizeof(BASE_TYPE);
    int threads_per_block;

    scanf("%d", &threads_per_block);

    dim3 threadsPerBlock, blocksPerGrid;
    cuda_init_grid_and_block(&threadsPerBlock, &blocksPerGrid, N, threads_per_block);

    cudaEvent_t start, stop;
    float h2d_cp_span, d2h_cp_span, k_span;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    BASE_TYPE *host_a = gen_array(N), *host_b = gen_array(N), *host_c = new BASE_TYPE[N];
    BASE_TYPE *dev_a, *dev_b, *dev_c;

    if (host_a == NULL || host_b == NULL || host_c == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    cudaEventRecord(start, 0);

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

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_cp_span, start, stop);

    for(int i = 0; i < 100; i++)
        add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&k_span, start, stop);

    cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_cp_span, start, stop);

    // printf("Copy form host to device time: %.2f milliseconds\n", h2d_cp_span);
    printf("Run kernel time: %.2f milliseconds\n", (k_span - h2d_cp_span) / 100);
    // printf("Copy form device to host time: %.2f milliseconds\n", d2h_cp_span);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}