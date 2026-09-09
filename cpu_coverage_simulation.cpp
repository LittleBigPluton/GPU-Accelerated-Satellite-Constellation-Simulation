#include "check_covered.hpp"
#include "ground_point.hpp"
#include "satellite.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

namespace
{
constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;
}

int main()
{
    constexpr float orbital_radius = 7000.0f;
    constexpr float angular_velocity = 0.001f;
    std::vector<Satellite> satellites
    {
        Satellite(orbital_radius, angular_velocity, 0.0f),
        Satellite(orbital_radius, angular_velocity, PI / 2.0f),
        Satellite(orbital_radius, angular_velocity, PI),
        Satellite(orbital_radius, angular_velocity, 3.0f * PI / 2.0f)
    };

    constexpr float earth_radius = 6371.0f;
    const ground_point ground(earth_radius, 0.0f, 0.0f);
    constexpr float simulation_duration = 400.0f;
    constexpr float delta_time = 1.0f;
    constexpr float min_elevation_angle = 10.0f * DEG_TO_RAD;
    for (float current_time = delta_time; current_time <= simulation_duration; current_time += delta_time)
    {
        std::cout << "Time: " << current_time << " s\n";
        for (std::size_t i = 0; i < satellites.size(); ++i)
        {
            Satellite& satellite = satellites[i];
            satellite.update_position(delta_time);
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            satellite.compute_position(x, y, z);
            const bool covered = is_covered(satellite, ground, min_elevation_angle);
            std::cout << "Satellite " << i + 1 << " position: (" << x << ", " << y << ", " << z << ") "
                      << (covered ? "covers" : "does not cover") << " the ground point.\n";
        }
        std::cout << "-----------------------------------\n";
    }
    return 0;
}
