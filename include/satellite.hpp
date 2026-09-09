#pragma once

#include <cuda_runtime.h>

class Satellite
{
public:
    __host__ __device__
    Satellite(float radius, float angular_velocity, float current_angle = 0.0f);

    __host__ __device__
    void update_position(float delta_time);

    __host__ __device__
    void compute_position(float& x, float& y, float& z) const;

private:
    float satellite_radius;
    float satellite_ang_vel;
    float satellite_cur_ang;
};
