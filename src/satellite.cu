#include "satellite.hpp"

#include <cmath>


__host__ __device__
Satellite::Satellite(float radius, float angular_velocity, float current_angle):
    satellite_radius(radius),
    satellite_ang_vel(angular_velocity),
    satellite_cur_ang(current_angle)
{
}

__host__ __device__
void Satellite::update_position(float delta_time)
{
    constexpr float TWO_PI =
        6.28318530717958647692f;

    satellite_cur_ang +=
        satellite_ang_vel * delta_time;

    satellite_cur_ang =
        fmodf(
            satellite_cur_ang,
            TWO_PI
        );

    if (satellite_cur_ang < 0.0f)
    {
        satellite_cur_ang += TWO_PI;
    }
}

__host__ __device__
void Satellite::compute_position(float& x, float& y, float& z) const
{
    x = satellite_radius * cosf(satellite_cur_ang);
    y = satellite_radius * sinf(satellite_cur_ang);
    z = 0.0f;
}
