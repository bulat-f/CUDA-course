#include <stdio.h>
#include <cuda_runtime_api.h>

__global__ void cuda_add(int a, int b, int *result)
{
    *result = a + b;
}

int main()
{
    int a, b;
    int h_result, *d_result;


    printf("Pleas, enter two ints\n");
    scanf("%d%d", &a, &b);

    cudaMalloc((void **)&d_result, sizeof(int));
    cuda_add<<<1, 1>>>(a, b, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_result);

    printf("a + b == %d\n", h_result);
    return 0;
}