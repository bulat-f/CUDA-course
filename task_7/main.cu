#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void init(BASE_TYPE *a, const int N)
{
    int threads_count = blockDim.x * gridDim.x;
    int elem_per_thread = N / threads_count;
    int k = (blockIdx.x * blockDim.x + threadIdx.x) * elem_per_thread;
    
    for (int i = k; i < k + elem_per_thread; i++)
    {
        a[i] = __sinf((i % 360) * M_PI / 180);
        // a[i] = sin((i % 360) * M_PI / 180);
        // a[i] = sinf((i % 360) * M_PI / 180);
    }
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
    const int MAX_GRID_SIZE_X = 65535;
    int blocks_count = N / threads_per_block;
    blocks_count = (blocks_count > MAX_GRID_SIZE_X ? MAX_GRID_SIZE_X : blocks_count);
    *grid = dim3(blocks_count);
    *block = dim3(threads_per_block);
    printf("Block %d %d %d\n", block->x, block->y, block->z);
    printf("Grid %d %d %d\n", grid->x, grid->y, grid->z);
}

int main()
{
    srand(time(NULL));

    const int N = 262144;
    const size_t size = N * sizeof(BASE_TYPE);
    int threads_per_block = 512;

    dim3 threadsPerBlock, blocksPerGrid;
    cuda_init_grid_and_block(&threadsPerBlock, &blocksPerGrid, N, threads_per_block);

    cudaEvent_t start, stop;
    float h2d_cp_span, d2h_cp_span, k_span;

    double sum = 0;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    BASE_TYPE *host_a = new BASE_TYPE[N];
    BASE_TYPE *dev_a;

    if (host_a == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

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

    cudaMemcpy(host_a, dev_a, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_cp_span, start, stop);

    printf("Copy form host to device time: %.2f milliseconds\n", h2d_cp_span);
    printf("Run kernel time: %.2f milliseconds\n", k_span);
    printf("Copy form device to host time: %.2f milliseconds\n", d2h_cp_span);
    printf("Clear run time: %.2f milliseconds\n", k_span - h2d_cp_span);

    for (int i = 0; i < N; i++)
    {
        sum += fabs(sin((i % 360) * M_PI / 180) - host_a[i]);
    }

    printf("\nSum = %f\n", sum);

    cudaFree(dev_a);
    delete[] host_a;

    return 0;
}