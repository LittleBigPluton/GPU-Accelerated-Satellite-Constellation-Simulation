#include "check_covered.hpp"
#include "ground_point.hpp"
#include "satellite.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;

constexpr int NUM_SATELLITES = 4;

constexpr float EARTH_RADIUS = 6371.0f;
constexpr float ORBITAL_RADIUS = 7000.0f;
constexpr float ANGULAR_VELOCITY = 0.001f;

constexpr float DELTA_TIME = 1.0f;
constexpr int NUM_STEPS = 300;

constexpr float MIN_ELEVATION_ANGLE =
    10.0f * DEG_TO_RAD;

constexpr float POSITION_TOLERANCE = 1e-3f;


void require(
    bool condition,
    const std::string& message
)
{
    if (!condition)
    {
        std::cerr
            << "[FAILED] "
            << message
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}


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
            << "[CUDA FAILED] "
            << cudaGetErrorString(error)
            << "\nExpression: "
            << expression
            << "\nLocation: "
            << file
            << ':'
            << line
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}


bool approximately_equal(
    float a,
    float b,
    float tolerance = POSITION_TOLERANCE
)
{
    return std::fabs(a - b) <= tolerance;
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
void update_satellites_kernel(
    Satellite* satellites,
    int num_satellites,
    float delta_time
)
{
    const int idx =
        static_cast<int>(
            blockIdx.x * blockDim.x
            + threadIdx.x
        );

    if (idx < num_satellites)
    {
        satellites[idx].update_position(
            delta_time
        );
    }
}


__global__
void compute_coverage_kernel(
    const Satellite* satellites,
    int num_satellites,
    ground_point ground,
    float min_elevation_angle,
    bool* coverage_results
)
{
    const int idx =
        static_cast<int>(
            blockIdx.x * blockDim.x
            + threadIdx.x
        );

    if (idx < num_satellites)
    {
        coverage_results[idx] =
            is_covered(
                satellites[idx],
                ground,
                min_elevation_angle
            );
    }
}


int main()
{
    // ---------------------------------------------------------
    // Check whether a CUDA device is available.
    // ---------------------------------------------------------

    int device_count = 0;

    const cudaError_t device_error =
        cudaGetDeviceCount(&device_count);

    if (
        device_error != cudaSuccess
        || device_count == 0
    )
    {
        std::cout
            << "[SKIPPED] No CUDA-capable device available.\n";

        // CTest will treat 77 as "skipped" once configured below.
        return 77;
    }


    // ---------------------------------------------------------
    // Shared initial conditions
    // ---------------------------------------------------------

    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    const std::array<Satellite, NUM_SATELLITES>
        initial_satellites{
            Satellite(
                ORBITAL_RADIUS,
                ANGULAR_VELOCITY,
                0.0f
            ),
            Satellite(
                ORBITAL_RADIUS,
                ANGULAR_VELOCITY,
                PI / 2.0f
            ),
            Satellite(
                ORBITAL_RADIUS,
                ANGULAR_VELOCITY,
                PI
            ),
            Satellite(
                ORBITAL_RADIUS,
                ANGULAR_VELOCITY,
                3.0f * PI / 2.0f
            )
        };


    // ---------------------------------------------------------
    // CPU calculation
    // ---------------------------------------------------------

    std::array<Satellite, NUM_SATELLITES>
        cpu_satellites = initial_satellites;

    std::array<bool, NUM_SATELLITES>
        cpu_coverage{};

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        for (
            int i = 0;
            i < NUM_SATELLITES;
            ++i
        )
        {
            cpu_satellites[i].update_position(
                DELTA_TIME
            );

            cpu_coverage[i] = is_covered(
                cpu_satellites[i],
                ground,
                MIN_ELEVATION_ANGLE
            );
        }
    }


    // ---------------------------------------------------------
    // GPU setup
    // ---------------------------------------------------------

    Satellite* device_satellites = nullptr;
    bool* device_coverage = nullptr;

    constexpr std::size_t satellite_bytes =
        NUM_SATELLITES * sizeof(Satellite);

    constexpr std::size_t coverage_bytes =
        NUM_SATELLITES * sizeof(bool);

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_satellites
            ),
            satellite_bytes
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &device_coverage
            ),
            coverage_bytes
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            device_satellites,
            initial_satellites.data(),
            satellite_bytes,
            cudaMemcpyHostToDevice
        )
    );


    // ---------------------------------------------------------
    // GPU calculation
    // ---------------------------------------------------------

    constexpr int block_size = 256;

    const int grid_size =
        (
            NUM_SATELLITES
            + block_size
            - 1
        )
        / block_size;

    for (int step = 0; step < NUM_STEPS; ++step)
    {
        update_satellites_kernel
            <<<grid_size, block_size>>>(
                device_satellites,
                NUM_SATELLITES,
                DELTA_TIME
            );

        CUDA_CHECK(
            cudaGetLastError()
        );

        compute_coverage_kernel
            <<<grid_size, block_size>>>(
                device_satellites,
                NUM_SATELLITES,
                ground,
                MIN_ELEVATION_ANGLE,
                device_coverage
            );

        CUDA_CHECK(
            cudaGetLastError()
        );
    }

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ---------------------------------------------------------
    // Copy GPU results back
    // ---------------------------------------------------------

    std::array<Satellite, NUM_SATELLITES>
        gpu_satellites{
            Satellite(0.0f, 0.0f),
            Satellite(0.0f, 0.0f),
            Satellite(0.0f, 0.0f),
            Satellite(0.0f, 0.0f)
        };

    std::array<bool, NUM_SATELLITES>
        gpu_coverage{};

    CUDA_CHECK(
        cudaMemcpy(
            gpu_satellites.data(),
            device_satellites,
            satellite_bytes,
            cudaMemcpyDeviceToHost
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            gpu_coverage.data(),
            device_coverage,
            coverage_bytes,
            cudaMemcpyDeviceToHost
        )
    );


    // ---------------------------------------------------------
    // Compare CPU and GPU results
    // ---------------------------------------------------------

    for (
        int i = 0;
        i < NUM_SATELLITES;
        ++i
    )
    {
        float cpu_x = 0.0f;
        float cpu_y = 0.0f;
        float cpu_z = 0.0f;

        float gpu_x = 0.0f;
        float gpu_y = 0.0f;
        float gpu_z = 0.0f;

        cpu_satellites[i].compute_position(
            cpu_x,
            cpu_y,
            cpu_z
        );

        gpu_satellites[i].compute_position(
            gpu_x,
            gpu_y,
            gpu_z
        );

        require(
            approximately_equal(
                cpu_x,
                gpu_x
            ),
            "CPU/GPU x position mismatch for satellite "
                + std::to_string(i)
        );

        require(
            approximately_equal(
                cpu_y,
                gpu_y
            ),
            "CPU/GPU y position mismatch for satellite "
                + std::to_string(i)
        );

        require(
            approximately_equal(
                cpu_z,
                gpu_z
            ),
            "CPU/GPU z position mismatch for satellite "
                + std::to_string(i)
        );

        require(
            cpu_coverage[i]
                == gpu_coverage[i],
            "CPU/GPU coverage mismatch for satellite "
                + std::to_string(i)
        );
    }


    // ---------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------

    CUDA_CHECK(
        cudaFree(device_satellites)
    );

    CUDA_CHECK(
        cudaFree(device_coverage)
    );


    std::cout
        << "[PASSED] CPU/GPU consistency test\n";

    return EXIT_SUCCESS;
}
