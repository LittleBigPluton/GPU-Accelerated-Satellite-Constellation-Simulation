#include<iostream>

void CUDA_check()
{
  int device_count = 0;
  cudaError_t error = cudaGetDeviceCount(&device_count);

  // Check DeviceCount succeed or not
  if(error != cudaSuccess)
  {
      std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
      return;
  }

  if(device_count != 0)
  {
    std::cout << "Number of CUDA-capable devices: " << device_count << std::endl;
  }
  else
  {
    std::cout << "No CUDA capable device found." << std::endl;
  }

}
