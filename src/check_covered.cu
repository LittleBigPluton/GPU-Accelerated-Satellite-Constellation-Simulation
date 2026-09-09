#include "check_covered.hpp"

#include <cmath>

__host__ __device__
bool is_covered(const Satellite& sat, const ground_point& ground, float min_elevation_angle)
{
    float sat_x;
    float sat_y;
    float sat_z;
    sat.compute_position(sat_x, sat_y, sat_z);

    // Line-of-sight vector from ground point to satellite
    const float vx = sat_x - ground.x();
    const float vy = sat_y - ground.y();
    const float vz = sat_z - ground.z();

    // Ground-point position vector defines local "up"
    const float gx = ground.x();
    const float gy = ground.y();
    const float gz = ground.z();
    const float los_norm = sqrtf(vx * vx + vy * vy + vz * vz);
    const float ground_norm = sqrtf(gx * gx + gy * gy + gz * gz);

    if (los_norm == 0.0f || ground_norm == 0.0f)
    {
        return false;
    }

    float sin_elevation = (vx * gx + vy * gy + vz * gz) / (los_norm * ground_norm);

    // Protect asinf() against floating-point round-off.
    if (sin_elevation > 1.0f)
    {
        sin_elevation = 1.0f;
    }
    else if (sin_elevation < -1.0f)
    {
        sin_elevation = -1.0f;
    }

    const float elevation_angle = asinf(sin_elevation);
    return elevation_angle > min_elevation_angle;
}
