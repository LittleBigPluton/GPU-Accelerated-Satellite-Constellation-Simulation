#include "ground_point.hpp"

__host__ __device__
ground_point::ground_point(float x, float y, float z):
    m_x(x),
    m_y(y),
    m_z(z)
{
}

__host__ __device__
float ground_point::x() const
{
    return m_x;
}

__host__ __device__
float ground_point::y() const
{
    return m_y;
}

__host__ __device__
float ground_point::z() const
{
    return m_z;
}
