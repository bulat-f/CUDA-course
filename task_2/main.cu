#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>

__global__ void cuda_add(int a, int b, int *result)
{
    *result = a + b;
}

int main()
{
    int a = 2, b = 3;
    int h_result, *d_result;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void **)&d_result, sizeof(int));
    if (err != cudaSuccess)
        printf("fail\n");

    cuda_add<<<1, 1>>>(a, b, d_result);

    err = cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("fail\n");

    cudaFree(d_result);

    printf("a + b == %d\n", h_result);
    return 0;
}