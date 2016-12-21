#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime_api.h>

#define BASE_TYPE float

__global__ void add(BASE_TYPE *a, BASE_TYPE *b, BASE_TYPE *result, const int N)
{
    int numElemPerThread = N / blockDim.x;
    int k = threadIdx.x * numElemPerThread;
    
    for (int i = k; i < k + numElemPerThread; i++)
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

int main()
{
    srand(time(NULL));

    const int N = 8;
    const size_t size = N * sizeof(BASE_TYPE);
    const int block_size = 2;

    dim3 threadsPerBlock = dim3(block_size);
    dim3 blocksPerGrid = dim3(N / block_size);


    BASE_TYPE *host_a = gen_array(N), *host_b = gen_array(N), *host_c = new BASE_TYPE[N];
    BASE_TYPE *dev_a, *dev_b, *dev_c;

    if (host_a == NULL || host_b == NULL || host_c == NULL)
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

    add<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_c, N);

    cudaMemcpy(host_c, dev_c, size, cudaMemcpyDeviceToHost);
    print_array(host_c, N);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    delete[] host_a;
    delete[] host_b;
    delete[] host_c;

    return 0;
}