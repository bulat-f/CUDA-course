/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* This sample queries the properties of the CUDA devices present in the system via CUDA Runtime API. */

// Shared Utilities (QA Testing)

// std::system includes
#include <memory>
#include <stdio.h>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

// #include <helper_cuda.h>

int main(int argc, char **argv)
{
    int deviceCount;
    cudaDeviceProp deviceProp;

    //Сколько устройств CUDA установлено на PC.
    cudaGetDeviceCount(&deviceCount);

    printf("Device count: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&deviceProp, i);

        printf("Device name: %s\n", deviceProp.name);
        printf("Total global memory: %d\n", deviceProp.totalGlobalMem);
        printf("Shared memory per block: %d\n", deviceProp.sharedMemPerBlock);
        printf("Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Memory pitch: %d\n", deviceProp.memPitch);
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

        printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
            deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);

        printf("Max grid size: x = %d, y = %d, z = %d\n",
            deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);

        printf("Clock rate: %d\n", deviceProp.clockRate);
        printf("Total constant memory: %d\n", deviceProp.totalConstMem);
        printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Texture alignment: %d\n", deviceProp.textureAlignment);
        printf("Device overlap: %d\n", deviceProp.deviceOverlap);
        printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

        printf("Kernel execution timeout enabled: %s\n",
        deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
    }

  return 0;
}
