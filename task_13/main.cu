#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void map(const BASE_TYPE *points, BASE_TYPE *result, const BASE_TYPE h)
{
    extern __shared__ BASE_TYPE s[];

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    s[threadIdx.x] = points[index] * h;
    __syncthreads();
    
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < blockDim.x; i++)
            s[0] += s[i];
        result[blockIdx.x] = s[0];
    }
}

BASE_TYPE reduce(const BASE_TYPE *dev_map, const int map_count)
{
    BASE_TYPE *host_map = new BASE_TYPE[map_count];
    BASE_TYPE result = 0;

    cudaMemcpy(host_map, dev_map, map_count * sizeof(BASE_TYPE), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < map_count; i++)
        result += host_map[i];

    return result;
}

BASE_TYPE func(BASE_TYPE x)
{
    return x;
}

BASE_TYPE* points(const BASE_TYPE a, const BASE_TYPE b, const int N)
{
    BASE_TYPE *p = new BASE_TYPE[N];
    BASE_TYPE h = (b - a) / N;

    for (int i = 0; i < N; i++)
    {
        p[i] = func(a + (i + 0.5) * h);
    }

    return p;
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

void cuda_init_grid_and_block(dim3 *grid, dim3 *block, const int threads_per_block, const int N)
{
    *grid = dim3(N / threads_per_block);
    *block = dim3(threads_per_block);
    printf("Block (%d, %d, %d)\n", block->x, block->y, block->z);
    printf("Grid  (%d, %d, %d)\n", grid->x, grid->y, grid->z);
}

int main()
{
    const int N = 20;
    const int threads_per_block = 5;
    const int block_count = N / threads_per_block;
    const size_t in_size = N * sizeof(BASE_TYPE);
    const size_t out_size = block_count * sizeof(BASE_TYPE);
    BASE_TYPE a = 0, b = 5;

    dim3 blockDim, gridDim;
    cuda_init_grid_and_block(&blockDim, &gridDim, threads_per_block, N);

    BASE_TYPE *host_a = points(a, b, N), result;
    BASE_TYPE *dev_a, *dev_result;

    try
    {
        cuda_init_array(&dev_a, host_a, in_size);
        cuda_init_array(&dev_result, NULL, out_size);
    }
    catch (cudaError_t err)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    map<<<blockDim, gridDim, threads_per_block * sizeof(BASE_TYPE)>>>(dev_a, dev_result, (b - a) / N);
    result = reduce(dev_result, block_count);
    printf("%3.2f\n", result);

    cudaFree(dev_a);
    cudaFree(dev_result);

    delete[] host_a;

    return 0;
}