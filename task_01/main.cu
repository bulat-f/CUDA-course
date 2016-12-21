#include <stdio.h>
#include <cuda_runtime_api.h>

__global__ void empty()
{
    return;
}

int main()
{
    dim3 gridSize = dim3(1, 1, 1);
    dim3 blockSize = dim3(1, 1, 1);
    empty<<<gridSize, blockSize>>>();
    printf("Hello World\n");
    return 0;
}
