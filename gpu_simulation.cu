#include "check_covered.hpp"
#include "cuda_info.hpp"
#include "ground_point.hpp"
#include "satellite.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>


namespace
{

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;


void check_cuda_error(
    cudaError_t error,
    const char* expression,
    const char* file,
    int line
)
{
    if (error != cudaSuccess)
    {
        std::cerr
            << "CUDA error: "
            << cudaGetErrorString(error)
            << "\nExpression: "
            << expression
            << "\nLocation: "
            << file
            << ":"
            << line
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}

} // namespace


#define CUDA_CHECK(expression)                    \
    do                                            \
    {                                             \
        check_cuda_error(                         \
            (expression),                         \
            #expression,                          \
            __FILE__,                             \
            __LINE__                              \
        );                                        \
    } while (false)


__global__
void update_satellite_positions(Satellite* satellites, int num_satellites, float delta_time)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < num_satellites)
    {
        satellites[idx].update_position(delta_time);
    }
}


__global__
void compute_coverage(const Satellite* satellites, int num_satellites, ground_point ground, float min_elevation_angle, bool* coverage_results)
{
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx < num_satellites)
    {
        coverage_results[idx] = is_covered(satellites[idx], ground, min_elevation_angle);
    }
}


int main()
{
    // Only prints information about available CUDA devices.
    // It does NOT perform CUDA error checking.
    print_cuda_device_info();
    constexpr int num_satellites = 4;
    constexpr float simulation_duration = 400.0f;
    constexpr float delta_time = 1.0f;
    constexpr float min_elevation_angle = 10.0f * DEG_TO_RAD;
    constexpr float earth_radius = 6371.0f;
    const ground_point ground(earth_radius, 0.0f, 0.0f);
    constexpr float orbital_radius = 7000.0f;
    constexpr float angular_velocity = 0.001f;
    std::vector<Satellite> host_satellites
    {
        Satellite(orbital_radius, angular_velocity, 0.0f),
        Satellite(orbital_radius, angular_velocity, PI / 2.0f),
        Satellite(orbital_radius, angular_velocity, PI),
        Satellite(orbital_radius, angular_velocity, 3.0f * PI / 2.0f)
    };

    const std::size_t satellite_bytes = host_satellites.size() * sizeof(Satellite);
    const std::size_t coverage_bytes = static_cast<std::size_t>(num_satellites) * sizeof(bool);
    Satellite* device_satellites = nullptr;
    bool* device_coverage_results = nullptr;


    // ---------------------------------------------------------
    // Allocate GPU memory
    // ---------------------------------------------------------

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&device_satellites),
            satellite_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(&device_coverage_results),
            coverage_bytes
        )
    );


    // ---------------------------------------------------------
    // Initial host -> device transfer
    // ---------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            device_satellites,
            host_satellites.data(),
            satellite_bytes,
            cudaMemcpyHostToDevice
        )
    );


    std::array<bool, num_satellites>
        host_coverage_results{};


    // ---------------------------------------------------------
    // CUDA launch configuration
    // ---------------------------------------------------------

    constexpr int block_size = 256;

    const int grid_size =
        (num_satellites + block_size - 1)
        / block_size;


    // ---------------------------------------------------------
    // Simulation
    // ---------------------------------------------------------

    for (
        float current_time = delta_time;
        current_time <= simulation_duration;
        current_time += delta_time
    )
    {
        // Update satellite states on GPU.
        update_satellite_positions
            <<<grid_size, block_size>>>(
                device_satellites,
                num_satellites,
                delta_time
            );

        CUDA_CHECK(cudaGetLastError());


        // Compute coverage on GPU.
        compute_coverage
            <<<grid_size, block_size>>>(
                device_satellites,
                num_satellites,
                ground,
                min_elevation_angle,
                device_coverage_results
            );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );


        // -----------------------------------------------------
        // Copy results back to CPU
        // -----------------------------------------------------

        CUDA_CHECK(
            cudaMemcpy(
                host_satellites.data(),
                device_satellites,
                satellite_bytes,
                cudaMemcpyDeviceToHost
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                host_coverage_results.data(),
                device_coverage_results,
                coverage_bytes,
                cudaMemcpyDeviceToHost
            )
        );


        // -----------------------------------------------------
        // Print simulation state
        // -----------------------------------------------------
        std::cout << "Time: " << current_time << " s\n";
        for (std::size_t i = 0; i < host_satellites.size(); ++i)
        {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            host_satellites[i].compute_position(x, y, z);

            std::cout
                << "Satellite "
                << i + 1
                << " position: ("
                << x
                << ", "
                << y
                << ", "
                << z
                << ") "
                << (
                    host_coverage_results[i]
                        ? "covers"
                        : "does not cover"
                )
                << " the ground point.\n";
        }
        std::cout
            << "-----------------------------------\n";
    }
    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------
    CUDA_CHECK(cudaFree(device_satellites));
    CUDA_CHECK(cudaFree(device_coverage_results));
    return 0;
}
