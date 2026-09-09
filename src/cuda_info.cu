#include "cuda_info.hpp"
#include <cuda_runtime.h>
#include <iostream>

void print_cuda_device_info()
{
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        return;
    }

    if (device_count > 0)
    {
        std::cout << "Number of CUDA-capable devices: " << device_count << '\n';
    }

    else
    {
        std::cout << "No CUDA-capable device found." << '\n';
    }
}
