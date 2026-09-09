#include "satellite.hpp"

#include <cstddef>
#include <iostream>
#include <vector>

int main()
{
    std::vector<Satellite> satellites;
    satellites.emplace_back(7000.0f, 0.0010f);
    satellites.emplace_back(7100.0f, 0.0011f);
    satellites.emplace_back(7200.0f, 0.0009f);
    constexpr float simulation_duration = 100.0f;
    constexpr float delta_time = 1.0f;
    for (float current_time = delta_time; current_time <= simulation_duration; current_time += delta_time)
    {
        std::cout << "Time: " << current_time << " s\n";
        for (std::size_t i = 0; i < satellites.size(); ++i)
        {
            satellites[i].update_position(delta_time);
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            satellites[i].compute_position(x, y, z);
            std::cout << "Satellite " << i + 1 << ": x = " << x << ", y = " << y << ", z = " << z << '\n';
        }
        std::cout << "-----------------------------------\n";
    }
    return 0;
}
