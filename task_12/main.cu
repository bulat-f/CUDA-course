#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void dot(const BASE_TYPE *a, const BASE_TYPE *b, BASE_TYPE *result, const int N)
{
    extern __shared__ BASE_TYPE s[];

    int index = blockDim.x * blockIdx.x + threadIdx.x;

    s[threadIdx.x] = a[index] * b[index];
    __syncthreads();
    
    if (threadIdx.x == 0) {
        BASE_TYPE sum = 0;

        for (int i = 0; i < blockDim.x; i++)
            sum += s[i];
        result[blockIdx.x] = sum;
    }
}

BASE_TYPE* gen_array(const int N)
{
    BASE_TYPE *a = new BASE_TYPE[N];

    for (int i = 0; i < N; i++)
    {
        a[i] = i;
    }

    return a;
}

void print_vector(BASE_TYPE *a, const int N)
{
    for (int i = 0; i < N; i++)
        printf("%5.0f ", a[i]);

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

void cuda_init_grid_and_block(dim3 *grid, dim3 *block, const int threadsPerBlock, const int N)
{
    *grid = dim3(1);
    *block = dim3(N);
    printf("Block %d %d %d\n", block->x, block->y, block->z);
    printf("Grid %d %d %d\n", grid->x, grid->y, grid->z);
}

BASE_TYPE dot_produce(const BASE_TYPE *host_a, const BASE_TYPE *host_b, const int N)
{
    const int threadsPerBlock = N;
    const size_t size = N * sizeof(BASE_TYPE);
    const size_t result_size = size / threadsPerBlock;
    cudaError_t err;

    dim3 blockDim, gridDim;
    cuda_init_grid_and_block(&blockDim, &gridDim, threadsPerBlock, N);

    BASE_TYPE *host_c = new BASE_TYPE[result_size / sizeof(BASE_TYPE)];
    BASE_TYPE *dev_a, *dev_b, *dev_c;
    BASE_TYPE result = 0;

    try
    {
        cuda_init_array(&dev_a, host_a, size);
        cuda_init_array(&dev_b, host_b, size);
        cuda_init_array(&dev_c, NULL, result_size);
    }
    catch (cudaError_t err)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dot<<<blockDim, gridDim, threadsPerBlock * sizeof(BASE_TYPE)>>>(dev_a, dev_b, dev_c, N);

    err = cudaMemcpy(host_c, dev_c, result_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device (error code: %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    for (int i = 0; i < result_size / sizeof(BASE_TYPE); i++)
        result += host_c[i];

    delete[] host_c;
    return result;
}

int main()
{
    const int N = 10;
    
    BASE_TYPE *host_a = gen_array(N), *host_b = gen_array(N);
    BASE_TYPE produce = dot_produce(host_a, host_b, N);

    print_vector(host_a, N);
    print_vector(host_b, N);

    printf("%f\n", produce);;

    delete[] host_a;
    delete[] host_b;

    return 0;
}