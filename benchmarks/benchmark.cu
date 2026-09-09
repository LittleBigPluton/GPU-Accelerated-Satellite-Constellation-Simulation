#include "check_covered.hpp"
#include "ground_point.hpp"
#include "satellite.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>


namespace
{

using Clock = std::chrono::steady_clock;

constexpr float PI =
    3.14159265358979323846f;

constexpr float DEG_TO_RAD =
    PI / 180.0f;

constexpr float EARTH_RADIUS =
    6371.0f;

constexpr float ORBITAL_RADIUS =
    7000.0f;

constexpr float ANGULAR_VELOCITY =
    0.001f;

constexpr float DELTA_TIME =
    1.0f;

constexpr float MIN_ELEVATION_ANGLE =
    10.0f * DEG_TO_RAD;

constexpr int BLOCK_SIZE =
    256;

// Single-step benchmark configuration.
constexpr int NUM_REPETITIONS =
    20;

// Resident multi-step benchmark configuration.
constexpr int RESIDENT_NUM_STEPS =
    100;

constexpr int RESIDENT_REPETITIONS =
    5;

constexpr float VALIDATION_POSITION_TOLERANCE =
    1e-2f;


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
            << ':'
            << line
            << '\n';

        std::exit(EXIT_FAILURE);
    }
}


double elapsed_ms(
    const Clock::time_point& start,
    const Clock::time_point& end
)
{
    return std::chrono::duration<
        double,
        std::milli
    >(end - start).count();
}


double median(
    std::vector<double> values
)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const std::size_t middle =
        values.size() / 2;

    if (values.size() % 2 == 0)
    {
        return (
            values[middle - 1]
            + values[middle]
        ) / 2.0;
    }

    return values[middle];
}


std::size_t count_covered(
    const std::vector<std::uint8_t>& coverage
)
{
    return static_cast<std::size_t>(
        std::count(
            coverage.begin(),
            coverage.end(),
            static_cast<std::uint8_t>(1)
        )
    );
}


/*
 * Create a deterministic constellation.
 *
 * A golden-angle phase distribution provides reproducible
 * satellite positions distributed across the full circular
 * orbit without relying on random-number generation.
 */
std::vector<Satellite> create_constellation(
    std::size_t num_satellites
)
{
    constexpr double TWO_PI_DOUBLE =
        6.28318530717958647692;

    constexpr double GOLDEN_ANGLE =
        2.39996322972865332223;

    std::vector<Satellite> satellites;

    satellites.reserve(
        num_satellites
    );

    for (
        std::size_t i = 0;
        i < num_satellites;
        ++i
    )
    {
        const double phase =
            std::fmod(
                0.123456789
                    + static_cast<double>(i)
                    * GOLDEN_ANGLE,
                TWO_PI_DOUBLE
            );

        satellites.emplace_back(
            ORBITAL_RADIUS,
            ANGULAR_VELOCITY,
            static_cast<float>(phase)
        );
    }

    return satellites;
}


/*
 * Perform one serial CPU simulation step:
 *
 * 1. update satellite position
 * 2. evaluate coverage
 */
void run_cpu_step(
    std::vector<Satellite>& satellites,
    const ground_point& ground,
    std::vector<std::uint8_t>& coverage
)
{
    const std::size_t num_satellites =
        satellites.size();

    for (
        std::size_t i = 0;
        i < num_satellites;
        ++i
    )
    {
        satellites[i].update_position(
            DELTA_TIME
        );

        coverage[i] =
            is_covered(
                satellites[i],
                ground,
                MIN_ELEVATION_ANGLE
            )
            ? static_cast<std::uint8_t>(1)
            : static_cast<std::uint8_t>(0);
    }
}


struct BenchmarkResult
{
    std::size_t num_satellites = 0;

    double cpu_ms = 0.0;

    double gpu_h2d_ms = 0.0;

    double gpu_update_ms = 0.0;
    double gpu_coverage_ms = 0.0;
    double gpu_kernel_ms = 0.0;

    double gpu_d2h_ms = 0.0;

    double gpu_total_ms = 0.0;

    double kernel_speedup = 0.0;
    double end_to_end_speedup = 0.0;

    std::size_t covered_count = 0;

    bool validation_passed = false;
};


struct ResidentBenchmarkResult
{
    std::size_t num_satellites = 0;

    int num_steps = 0;

    double cpu_total_ms = 0.0;

    double gpu_h2d_ms = 0.0;

    double gpu_kernel_total_ms = 0.0;

    double gpu_d2h_ms = 0.0;

    double gpu_total_ms = 0.0;

    double kernel_speedup = 0.0;
    double end_to_end_speedup = 0.0;

    std::size_t final_covered_count = 0;

    bool validation_passed = false;
};


/*
 * Validate a small selection of final satellite positions.
 *
 * Full coverage vectors are compared separately. Sampling the
 * satellite state avoids adding unnecessary validation cost
 * while still checking that repeated CPU/GPU position updates
 * remain consistent.
 */
bool validate_sampled_positions(
    const std::vector<Satellite>& cpu_satellites,
    const std::vector<Satellite>& gpu_satellites
)
{
    const std::size_t num_satellites =
        cpu_satellites.size();

    if (
        num_satellites == 0
        || gpu_satellites.size()
            != num_satellites
    )
    {
        return false;
    }

    const std::array<std::size_t, 5>
        sample_indices{
            0,
            num_satellites / 4,
            num_satellites / 2,
            (3 * num_satellites) / 4,
            num_satellites - 1
        };

    for (
        const std::size_t index
        : sample_indices
    )
    {
        float cpu_x = 0.0f;
        float cpu_y = 0.0f;
        float cpu_z = 0.0f;

        float gpu_x = 0.0f;
        float gpu_y = 0.0f;
        float gpu_z = 0.0f;

        cpu_satellites[index].compute_position(
            cpu_x,
            cpu_y,
            cpu_z
        );

        gpu_satellites[index].compute_position(
            gpu_x,
            gpu_y,
            gpu_z
        );

        if (
            std::fabs(cpu_x - gpu_x)
                > VALIDATION_POSITION_TOLERANCE
            ||
            std::fabs(cpu_y - gpu_y)
                > VALIDATION_POSITION_TOLERANCE
            ||
            std::fabs(cpu_z - gpu_z)
                > VALIDATION_POSITION_TOLERANCE
        )
        {
            std::cerr
                << "Final-state position mismatch "
                << "for satellite "
                << index
                << '\n'
                << "CPU: ("
                << cpu_x
                << ", "
                << cpu_y
                << ", "
                << cpu_z
                << ")\n"
                << "GPU: ("
                << gpu_x
                << ", "
                << gpu_y
                << ", "
                << gpu_z
                << ")\n";

            return false;
        }
    }

    return true;
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


/*
 * CUDA kernel: update all satellite positions.
 */
__global__
void update_satellites_kernel(
    Satellite* satellites,
    std::size_t num_satellites,
    float delta_time
)
{
    const std::size_t idx =
        static_cast<std::size_t>(
            blockIdx.x
        ) * blockDim.x
        + threadIdx.x;

    if (idx < num_satellites)
    {
        satellites[idx].update_position(
            delta_time
        );
    }
}


/*
 * CUDA kernel: evaluate coverage for every satellite.
 */
__global__
void compute_coverage_kernel(
    const Satellite* satellites,
    std::size_t num_satellites,
    ground_point ground,
    float min_elevation_angle,
    std::uint8_t* coverage_results
)
{
    const std::size_t idx =
        static_cast<std::size_t>(
            blockIdx.x
        ) * blockDim.x
        + threadIdx.x;

    if (idx < num_satellites)
    {
        coverage_results[idx] =
            is_covered(
                satellites[idx],
                ground,
                min_elevation_angle
            )
            ? static_cast<std::uint8_t>(1)
            : static_cast<std::uint8_t>(0);
    }
}


namespace
{

// ============================================================================
// SINGLE-STEP BENCHMARK
// ============================================================================

BenchmarkResult run_benchmark(
    std::size_t num_satellites
)
{
    BenchmarkResult result;

    result.num_satellites =
        num_satellites;


    // ------------------------------------------------------------------------
    // Initial conditions
    // ------------------------------------------------------------------------

    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    const std::vector<Satellite>
        initial_satellites =
            create_constellation(
                num_satellites
            );

    std::vector<Satellite>
        cpu_satellites =
            initial_satellites;

    std::vector<std::uint8_t>
        cpu_coverage(
            num_satellites,
            0
        );

    std::vector<std::uint8_t>
        gpu_coverage(
            num_satellites,
            0
        );


    const std::size_t satellite_bytes =
        num_satellites
        * sizeof(Satellite);

    const std::size_t coverage_bytes =
        num_satellites
        * sizeof(std::uint8_t);


    // ------------------------------------------------------------------------
    // Device allocation
    //
    // Allocation is intentionally excluded from benchmark timing.
    // ------------------------------------------------------------------------

    Satellite* device_satellites =
        nullptr;

    std::uint8_t* device_coverage =
        nullptr;

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


    // ------------------------------------------------------------------------
    // CUDA timing events
    // ------------------------------------------------------------------------

    cudaEvent_t update_start;
    cudaEvent_t update_stop;

    cudaEvent_t coverage_start;
    cudaEvent_t coverage_stop;

    CUDA_CHECK(
        cudaEventCreate(
            &update_start
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &update_stop
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &coverage_start
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &coverage_stop
        )
    );


    const int grid_size =
        static_cast<int>(
            (
                num_satellites
                + BLOCK_SIZE
                - 1
            )
            / BLOCK_SIZE
        );


    // ------------------------------------------------------------------------
    // GPU warm-up
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            device_satellites,
            initial_satellites.data(),
            satellite_bytes,
            cudaMemcpyHostToDevice
        )
    );

    update_satellites_kernel
        <<<grid_size, BLOCK_SIZE>>>(
            device_satellites,
            num_satellites,
            DELTA_TIME
        );

    CUDA_CHECK(
        cudaGetLastError()
    );

    compute_coverage_kernel
        <<<grid_size, BLOCK_SIZE>>>(
            device_satellites,
            num_satellites,
            ground,
            MIN_ELEVATION_ANGLE,
            device_coverage
        );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ------------------------------------------------------------------------
    // Timing samples
    // ------------------------------------------------------------------------

    std::vector<double> cpu_times;

    std::vector<double> h2d_times;

    std::vector<double> update_times;

    std::vector<double> coverage_times;

    std::vector<double> kernel_times;

    std::vector<double> d2h_times;

    std::vector<double> total_gpu_times;


    cpu_times.reserve(
        NUM_REPETITIONS
    );

    h2d_times.reserve(
        NUM_REPETITIONS
    );

    update_times.reserve(
        NUM_REPETITIONS
    );

    coverage_times.reserve(
        NUM_REPETITIONS
    );

    kernel_times.reserve(
        NUM_REPETITIONS
    );

    d2h_times.reserve(
        NUM_REPETITIONS
    );

    total_gpu_times.reserve(
        NUM_REPETITIONS
    );


    // ------------------------------------------------------------------------
    // Benchmark repetitions
    // ------------------------------------------------------------------------

    for (
        int repetition = 0;
        repetition < NUM_REPETITIONS;
        ++repetition
    )
    {
        // ====================================================================
        // CPU
        // ====================================================================

        cpu_satellites =
            initial_satellites;

        std::fill(
            cpu_coverage.begin(),
            cpu_coverage.end(),
            static_cast<std::uint8_t>(0)
        );


        const auto cpu_start =
            Clock::now();

        run_cpu_step(
            cpu_satellites,
            ground,
            cpu_coverage
        );

        const auto cpu_stop =
            Clock::now();


        cpu_times.push_back(
            elapsed_ms(
                cpu_start,
                cpu_stop
            )
        );


        // ====================================================================
        // GPU end-to-end
        // ====================================================================

        const auto gpu_total_start =
            Clock::now();


        // --------------------------------------------------------------------
        // Host -> Device
        // --------------------------------------------------------------------

        const auto h2d_start =
            Clock::now();

        CUDA_CHECK(
            cudaMemcpy(
                device_satellites,
                initial_satellites.data(),
                satellite_bytes,
                cudaMemcpyHostToDevice
            )
        );

        const auto h2d_stop =
            Clock::now();


        h2d_times.push_back(
            elapsed_ms(
                h2d_start,
                h2d_stop
            )
        );


        // --------------------------------------------------------------------
        // Update kernel
        // --------------------------------------------------------------------

        CUDA_CHECK(
            cudaEventRecord(
                update_start
            )
        );

        update_satellites_kernel
            <<<grid_size, BLOCK_SIZE>>>(
                device_satellites,
                num_satellites,
                DELTA_TIME
            );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaEventRecord(
                update_stop
            )
        );


        // --------------------------------------------------------------------
        // Coverage kernel
        // --------------------------------------------------------------------

        CUDA_CHECK(
            cudaEventRecord(
                coverage_start
            )
        );

        compute_coverage_kernel
            <<<grid_size, BLOCK_SIZE>>>(
                device_satellites,
                num_satellites,
                ground,
                MIN_ELEVATION_ANGLE,
                device_coverage
            );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaEventRecord(
                coverage_stop
            )
        );


        // Wait until the complete GPU computation is finished.
        CUDA_CHECK(
            cudaEventSynchronize(
                coverage_stop
            )
        );


        float update_ms =
            0.0f;

        float coverage_ms =
            0.0f;


        CUDA_CHECK(
            cudaEventElapsedTime(
                &update_ms,
                update_start,
                update_stop
            )
        );

        CUDA_CHECK(
            cudaEventElapsedTime(
                &coverage_ms,
                coverage_start,
                coverage_stop
            )
        );


        update_times.push_back(
            static_cast<double>(
                update_ms
            )
        );

        coverage_times.push_back(
            static_cast<double>(
                coverage_ms
            )
        );

        kernel_times.push_back(
            static_cast<double>(
                update_ms
                + coverage_ms
            )
        );


        // --------------------------------------------------------------------
        // Device -> Host
        // --------------------------------------------------------------------

        const auto d2h_start =
            Clock::now();

        CUDA_CHECK(
            cudaMemcpy(
                gpu_coverage.data(),
                device_coverage,
                coverage_bytes,
                cudaMemcpyDeviceToHost
            )
        );

        const auto d2h_stop =
            Clock::now();


        d2h_times.push_back(
            elapsed_ms(
                d2h_start,
                d2h_stop
            )
        );


        const auto gpu_total_stop =
            Clock::now();


        total_gpu_times.push_back(
            elapsed_ms(
                gpu_total_start,
                gpu_total_stop
            )
        );


        // --------------------------------------------------------------------
        // Correctness validation
        //
        // This occurs outside timed regions.
        // --------------------------------------------------------------------

        if (cpu_coverage != gpu_coverage)
        {
            std::size_t mismatch_index =
                0;

            for (
                std::size_t i = 0;
                i < num_satellites;
                ++i
            )
            {
                if (
                    cpu_coverage[i]
                    != gpu_coverage[i]
                )
                {
                    mismatch_index =
                        i;

                    break;
                }
            }

            std::cerr
                << "Single-step validation failed.\n"
                << "Satellites: "
                << num_satellites
                << '\n'
                << "First mismatch index: "
                << mismatch_index
                << '\n'
                << "CPU coverage: "
                << static_cast<int>(
                    cpu_coverage[
                        mismatch_index
                    ]
                )
                << '\n'
                << "GPU coverage: "
                << static_cast<int>(
                    gpu_coverage[
                        mismatch_index
                    ]
                )
                << '\n';

            std::exit(
                EXIT_FAILURE
            );
        }
    }


    // ------------------------------------------------------------------------
    // Aggregate statistics
    // ------------------------------------------------------------------------

    result.cpu_ms =
        median(
            cpu_times
        );

    result.gpu_h2d_ms =
        median(
            h2d_times
        );

    result.gpu_update_ms =
        median(
            update_times
        );

    result.gpu_coverage_ms =
        median(
            coverage_times
        );

    result.gpu_kernel_ms =
        median(
            kernel_times
        );

    result.gpu_d2h_ms =
        median(
            d2h_times
        );

    result.gpu_total_ms =
        median(
            total_gpu_times
        );


    if (result.gpu_kernel_ms > 0.0)
    {
        result.kernel_speedup =
            result.cpu_ms
            / result.gpu_kernel_ms;
    }


    if (result.gpu_total_ms > 0.0)
    {
        result.end_to_end_speedup =
            result.cpu_ms
            / result.gpu_total_ms;
    }


    result.covered_count =
        count_covered(
            cpu_coverage
        );

    result.validation_passed =
        true;


    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(
            update_start
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            update_stop
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            coverage_start
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            coverage_stop
        )
    );

    CUDA_CHECK(
        cudaFree(
            device_satellites
        )
    );

    CUDA_CHECK(
        cudaFree(
            device_coverage
        )
    );


    return result;
}


// ============================================================================
// RESIDENT MULTI-STEP BENCHMARK
// ============================================================================

ResidentBenchmarkResult run_resident_benchmark(
    std::size_t num_satellites,
    int num_steps
)
{
    ResidentBenchmarkResult result;

    result.num_satellites =
        num_satellites;

    result.num_steps =
        num_steps;


    // ------------------------------------------------------------------------
    // Initial conditions
    // ------------------------------------------------------------------------

    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    const std::vector<Satellite>
        initial_satellites =
            create_constellation(
                num_satellites
            );

    std::vector<Satellite>
        cpu_satellites =
            initial_satellites;

    std::vector<Satellite>
        gpu_satellites =
            initial_satellites;

    std::vector<std::uint8_t>
        cpu_coverage(
            num_satellites,
            0
        );

    std::vector<std::uint8_t>
        gpu_coverage(
            num_satellites,
            0
        );


    const std::size_t satellite_bytes =
        num_satellites
        * sizeof(Satellite);

    const std::size_t coverage_bytes =
        num_satellites
        * sizeof(std::uint8_t);


    // ------------------------------------------------------------------------
    // Device allocation
    //
    // Allocation is excluded from benchmark timing.
    // ------------------------------------------------------------------------

    Satellite* device_satellites =
        nullptr;

    std::uint8_t* device_coverage =
        nullptr;


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


    // ------------------------------------------------------------------------
    // CUDA timing events
    //
    // One event pair surrounds the entire resident kernel sequence.
    // ------------------------------------------------------------------------

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;

    CUDA_CHECK(
        cudaEventCreate(
            &kernel_start
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &kernel_stop
        )
    );


    const int grid_size =
        static_cast<int>(
            (
                num_satellites
                + BLOCK_SIZE
                - 1
            )
            / BLOCK_SIZE
        );


    // ------------------------------------------------------------------------
    // GPU warm-up
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            device_satellites,
            initial_satellites.data(),
            satellite_bytes,
            cudaMemcpyHostToDevice
        )
    );


    update_satellites_kernel
        <<<grid_size, BLOCK_SIZE>>>(
            device_satellites,
            num_satellites,
            DELTA_TIME
        );

    CUDA_CHECK(
        cudaGetLastError()
    );


    compute_coverage_kernel
        <<<grid_size, BLOCK_SIZE>>>(
            device_satellites,
            num_satellites,
            ground,
            MIN_ELEVATION_ANGLE,
            device_coverage
        );

    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ------------------------------------------------------------------------
    // Timing samples
    // ------------------------------------------------------------------------

    std::vector<double>
        cpu_times;

    std::vector<double>
        h2d_times;

    std::vector<double>
        kernel_times;

    std::vector<double>
        d2h_times;

    std::vector<double>
        total_gpu_times;


    cpu_times.reserve(
        RESIDENT_REPETITIONS
    );

    h2d_times.reserve(
        RESIDENT_REPETITIONS
    );

    kernel_times.reserve(
        RESIDENT_REPETITIONS
    );

    d2h_times.reserve(
        RESIDENT_REPETITIONS
    );

    total_gpu_times.reserve(
        RESIDENT_REPETITIONS
    );


    // ------------------------------------------------------------------------
    // Benchmark repetitions
    // ------------------------------------------------------------------------

    for (
        int repetition = 0;
        repetition < RESIDENT_REPETITIONS;
        ++repetition
    )
    {
        // ====================================================================
        // CPU
        // ====================================================================

        cpu_satellites =
            initial_satellites;

        std::fill(
            cpu_coverage.begin(),
            cpu_coverage.end(),
            static_cast<std::uint8_t>(0)
        );


        const auto cpu_start =
            Clock::now();


        for (
            int step = 0;
            step < num_steps;
            ++step
        )
        {
            run_cpu_step(
                cpu_satellites,
                ground,
                cpu_coverage
            );
        }


        const auto cpu_stop =
            Clock::now();


        cpu_times.push_back(
            elapsed_ms(
                cpu_start,
                cpu_stop
            )
        );


        // ====================================================================
        // GPU resident simulation
        // ====================================================================

        const auto gpu_total_start =
            Clock::now();


        // --------------------------------------------------------------------
        // Initial Host -> Device transfer
        // --------------------------------------------------------------------

        const auto h2d_start =
            Clock::now();


        CUDA_CHECK(
            cudaMemcpy(
                device_satellites,
                initial_satellites.data(),
                satellite_bytes,
                cudaMemcpyHostToDevice
            )
        );


        const auto h2d_stop =
            Clock::now();


        h2d_times.push_back(
            elapsed_ms(
                h2d_start,
                h2d_stop
            )
        );


        // --------------------------------------------------------------------
        // Resident kernel sequence
        // --------------------------------------------------------------------

        CUDA_CHECK(
            cudaEventRecord(
                kernel_start
            )
        );


        for (
            int step = 0;
            step < num_steps;
            ++step
        )
        {
            update_satellites_kernel
                <<<grid_size, BLOCK_SIZE>>>(
                    device_satellites,
                    num_satellites,
                    DELTA_TIME
                );

            compute_coverage_kernel
                <<<grid_size, BLOCK_SIZE>>>(
                    device_satellites,
                    num_satellites,
                    ground,
                    MIN_ELEVATION_ANGLE,
                    device_coverage
                );
        }


        /*
         * Avoid host-side checks between individual launches
         * because they would alter resident-loop timing.
         *
         * Asynchronous execution errors are also caught by the
         * subsequent event synchronization.
         */
        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaEventRecord(
                kernel_stop
            )
        );


        CUDA_CHECK(
            cudaEventSynchronize(
                kernel_stop
            )
        );


        float kernel_ms =
            0.0f;


        CUDA_CHECK(
            cudaEventElapsedTime(
                &kernel_ms,
                kernel_start,
                kernel_stop
            )
        );


        kernel_times.push_back(
            static_cast<double>(
                kernel_ms
            )
        );


        // --------------------------------------------------------------------
        // Final Device -> Host coverage transfer
        // --------------------------------------------------------------------

        const auto d2h_start =
            Clock::now();


        CUDA_CHECK(
            cudaMemcpy(
                gpu_coverage.data(),
                device_coverage,
                coverage_bytes,
                cudaMemcpyDeviceToHost
            )
        );


        const auto d2h_stop =
            Clock::now();


        d2h_times.push_back(
            elapsed_ms(
                d2h_start,
                d2h_stop
            )
        );


        const auto gpu_total_stop =
            Clock::now();


        total_gpu_times.push_back(
            elapsed_ms(
                gpu_total_start,
                gpu_total_stop
            )
        );


        // --------------------------------------------------------------------
        // Full final-coverage validation
        //
        // Validation is outside timed regions.
        // --------------------------------------------------------------------

        if (cpu_coverage != gpu_coverage)
        {
            std::size_t mismatch_index =
                0;


            for (
                std::size_t i = 0;
                i < num_satellites;
                ++i
            )
            {
                if (
                    cpu_coverage[i]
                    != gpu_coverage[i]
                )
                {
                    mismatch_index =
                        i;

                    break;
                }
            }


            std::cerr
                << "Resident benchmark coverage "
                << "validation failed.\n"
                << "Satellites: "
                << num_satellites
                << '\n'
                << "Steps: "
                << num_steps
                << '\n'
                << "First mismatch index: "
                << mismatch_index
                << '\n'
                << "CPU coverage: "
                << static_cast<int>(
                    cpu_coverage[
                        mismatch_index
                    ]
                )
                << '\n'
                << "GPU coverage: "
                << static_cast<int>(
                    gpu_coverage[
                        mismatch_index
                    ]
                )
                << '\n';

            std::exit(
                EXIT_FAILURE
            );
        }


        // --------------------------------------------------------------------
        // Final satellite-state validation
        //
        // This transfer is validation-only and is intentionally
        // excluded from the benchmark's end-to-end timing.
        // --------------------------------------------------------------------

        CUDA_CHECK(
            cudaMemcpy(
                gpu_satellites.data(),
                device_satellites,
                satellite_bytes,
                cudaMemcpyDeviceToHost
            )
        );


        if (
            !validate_sampled_positions(
                cpu_satellites,
                gpu_satellites
            )
        )
        {
            std::cerr
                << "Resident benchmark final-state "
                << "validation failed.\n";

            std::exit(
                EXIT_FAILURE
            );
        }
    }


    // ------------------------------------------------------------------------
    // Aggregate statistics
    // ------------------------------------------------------------------------

    result.cpu_total_ms =
        median(
            cpu_times
        );

    result.gpu_h2d_ms =
        median(
            h2d_times
        );

    result.gpu_kernel_total_ms =
        median(
            kernel_times
        );

    result.gpu_d2h_ms =
        median(
            d2h_times
        );

    result.gpu_total_ms =
        median(
            total_gpu_times
        );


    if (result.gpu_kernel_total_ms > 0.0)
    {
        result.kernel_speedup =
            result.cpu_total_ms
            / result.gpu_kernel_total_ms;
    }


    if (result.gpu_total_ms > 0.0)
    {
        result.end_to_end_speedup =
            result.cpu_total_ms
            / result.gpu_total_ms;
    }


    result.final_covered_count =
        count_covered(
            cpu_coverage
        );

    result.validation_passed =
        true;


    // ------------------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------------------

    CUDA_CHECK(
        cudaEventDestroy(
            kernel_start
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            kernel_stop
        )
    );

    CUDA_CHECK(
        cudaFree(
            device_satellites
        )
    );

    CUDA_CHECK(
        cudaFree(
            device_coverage
        )
    );


    return result;
}


// ============================================================================
// OUTPUT
// ============================================================================

void print_result(
    const BenchmarkResult& result
)
{
    std::cout
        << "\nSatellites: "
        << result.num_satellites
        << '\n';


    std::cout
        << std::fixed
        << std::setprecision(4);


    std::cout
        << "CPU compute:         "
        << result.cpu_ms
        << " ms\n";

    std::cout
        << "GPU H2D:             "
        << result.gpu_h2d_ms
        << " ms\n";

    std::cout
        << "GPU update kernel:   "
        << result.gpu_update_ms
        << " ms\n";

    std::cout
        << "GPU coverage kernel: "
        << result.gpu_coverage_ms
        << " ms\n";

    std::cout
        << "GPU kernels total:   "
        << result.gpu_kernel_ms
        << " ms\n";

    std::cout
        << "GPU D2H:             "
        << result.gpu_d2h_ms
        << " ms\n";

    std::cout
        << "GPU end-to-end:      "
        << result.gpu_total_ms
        << " ms\n";

    std::cout
        << "Kernel speedup:      "
        << result.kernel_speedup
        << "x\n";

    std::cout
        << "End-to-end speedup:  "
        << result.end_to_end_speedup
        << "x\n";

    std::cout
        << "Covered satellites:  "
        << result.covered_count
        << '\n';

    std::cout
        << "Validation:          "
        << (
            result.validation_passed
                ? "PASS"
                : "FAIL"
        )
        << '\n';
}


void print_resident_result(
    const ResidentBenchmarkResult& result
)
{
    const double cpu_per_step =
        result.cpu_total_ms
        / static_cast<double>(
            result.num_steps
        );

    const double gpu_kernel_per_step =
        result.gpu_kernel_total_ms
        / static_cast<double>(
            result.num_steps
        );

    const double gpu_total_per_step =
        result.gpu_total_ms
        / static_cast<double>(
            result.num_steps
        );


    std::cout
        << "\nSatellites: "
        << result.num_satellites
        << '\n';

    std::cout
        << "Timesteps:  "
        << result.num_steps
        << '\n';

    std::cout
        << "Satellite-step evaluations: "
        << (
            result.num_satellites
            * static_cast<std::size_t>(
                result.num_steps
            )
        )
        << '\n';


    std::cout
        << std::fixed
        << std::setprecision(4);


    std::cout
        << "CPU total:           "
        << result.cpu_total_ms
        << " ms\n";

    std::cout
        << "CPU per step:        "
        << cpu_per_step
        << " ms\n";

    std::cout
        << "GPU H2D once:        "
        << result.gpu_h2d_ms
        << " ms\n";

    std::cout
        << "GPU kernels total:   "
        << result.gpu_kernel_total_ms
        << " ms\n";

    std::cout
        << "GPU kernel per step: "
        << gpu_kernel_per_step
        << " ms\n";

    std::cout
        << "GPU D2H once:        "
        << result.gpu_d2h_ms
        << " ms\n";

    std::cout
        << "GPU end-to-end:      "
        << result.gpu_total_ms
        << " ms\n";

    std::cout
        << "GPU total per step:  "
        << gpu_total_per_step
        << " ms\n";

    std::cout
        << "Kernel speedup:      "
        << result.kernel_speedup
        << "x\n";

    std::cout
        << "End-to-end speedup:  "
        << result.end_to_end_speedup
        << "x\n";

    std::cout
        << "Final covered:       "
        << result.final_covered_count
        << '\n';

    std::cout
        << "Validation:          "
        << (
            result.validation_passed
                ? "PASS"
                : "FAIL"
        )
        << '\n';
}

} // namespace


int main()
{
    // ------------------------------------------------------------------------
    // CUDA device check
    // ------------------------------------------------------------------------

    int device_count =
        0;


    const cudaError_t device_error =
        cudaGetDeviceCount(
            &device_count
        );


    if (
        device_error != cudaSuccess
        || device_count == 0
    )
    {
        std::cerr
            << "No CUDA-capable device available.\n";

        return EXIT_FAILURE;
    }


    CUDA_CHECK(
        cudaSetDevice(0)
    );


    cudaDeviceProp device_properties{};


    CUDA_CHECK(
        cudaGetDeviceProperties(
            &device_properties,
            0
        )
    );


    /*
     * Force CUDA context initialization before any benchmark
     * measurements are performed.
     */
    CUDA_CHECK(
        cudaFree(nullptr)
    );


    // ------------------------------------------------------------------------
    // Environment
    // ------------------------------------------------------------------------

    std::cout
        << "GPU-Accelerated Satellite Simulation Benchmark\n"
        << "==============================================\n";


    std::cout
        << "GPU: "
        << device_properties.name
        << '\n';


    std::cout
        << "Compute capability: "
        << device_properties.major
        << '.'
        << device_properties.minor
        << '\n';


    std::cout
        << "Global memory: "
        << (
            device_properties.totalGlobalMem
            / (1024.0 * 1024.0)
        )
        << " MiB\n";


    std::cout
        << "CPU baseline: single-threaded\n";


    // ========================================================================
    // SINGLE-STEP BENCHMARK
    // ========================================================================

    std::cout
        << "\nSingle-Step Benchmark\n"
        << "=====================\n";


    std::cout
        << "Repetitions per size: "
        << NUM_REPETITIONS
        << '\n';


    std::cout
        << "Timed workload: one position update + "
        << "one coverage evaluation per satellite\n";


    std::cout
        << "GPU allocation time: excluded\n";


    constexpr std::array<std::size_t, 4>
        problem_sizes{
            1'000,
            10'000,
            100'000,
            1'000'000
        };


    for (
        const std::size_t size
        : problem_sizes
    )
    {
        const BenchmarkResult result =
            run_benchmark(
                size
            );


        print_result(
            result
        );
    }


    // ========================================================================
    // RESIDENT MULTI-STEP BENCHMARK
    // ========================================================================

    std::cout
        << "\n\n"
        << "Resident Multi-Step Benchmark\n"
        << "=============================\n";


    std::cout
        << "Satellite state remains on the GPU across "
        << RESIDENT_NUM_STEPS
        << " timesteps.\n";


    std::cout
        << "One H2D satellite-state transfer before simulation "
        << "and one final coverage D2H transfer after simulation.\n";


    std::cout
        << "Validation-only satellite-state D2H transfer: "
        << "excluded from timing.\n";


    std::cout
        << "GPU allocation time: excluded\n";


    std::cout
        << "Resident repetitions per size: "
        << RESIDENT_REPETITIONS
        << '\n';


    constexpr std::array<std::size_t, 2>
        resident_problem_sizes{
            100'000,
            1'000'000
        };


    for (
        const std::size_t size
        : resident_problem_sizes
    )
    {
        const ResidentBenchmarkResult result =
            run_resident_benchmark(
                size,
                RESIDENT_NUM_STEPS
            );


        print_resident_result(
            result
        );
    }


    return EXIT_SUCCESS;
}
