#pragma once

#include <cuda_runtime.h>

class ground_point
{
public:
    __host__ __device__
    ground_point(float x, float y, float z);

    __host__ __device__
    float x() const;

    __host__ __device__
    float y() const;

    __host__ __device__
    float z() const;

private:
    float m_x;
    float m_y;
    float m_z;
};
