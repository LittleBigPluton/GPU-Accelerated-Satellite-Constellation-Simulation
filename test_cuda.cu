#include <iostream>
#include <vector>
#include <fstream>
#include "satellite.h"
#include "groundpoint.h"
#include "check_covered.h"
#include "cudainfo.h"   // Assumes this prints CUDA device info
#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to update satellite positions
__global__ void update_satellite_positions(Satellite* d_satellites, int num_sats, float delta_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sats) {
        d_satellites[idx].update_position(delta_time);
    }
}

// CUDA kernel to compute coverage status for each satellite
__global__ void compute_coverage(const Satellite* d_satellites, int num_sats,
                       const ground_point ground, float min_elevation_angle,
                       bool* d_coverage_results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_sats) {
        float satX, satY, satZ;
        d_satellites[idx].compute_position(satX, satY, satZ);
        float vx = satX - ground.x();
        float vy = satY - ground.y();
        float vz = satZ - ground.z();
        float distance = sqrtf(vx * vx + vy * vy + vz * vz);
        float elevation_angle = asinf(vz / distance);
        d_coverage_results[idx] = (elevation_angle > min_elevation_angle);
    }
}

int main() {
    CUDA_check();  // Check CUDA device info if applicable

    const int num_sats = 4;
    const float delta_time = 1.0f;
    const float simulation_duration = 100.0f;  // Total simulation time in seconds
    const float min_elevation_angle = 10.0f * M_PI / 180.0f;

    // Define a ground point
    ground_point ground(7000.0f, 0.0f, -100.0f);

    // Create host satellites with initial positions
    std::vector<Satellite> h_satellites;
    h_satellites.push_back(Satellite(7000.0f, 0.001f, 0.0f));
    h_satellites.push_back(Satellite(7000.0f, 0.001f, M_PI / 2));
    h_satellites.push_back(Satellite(7000.0f, 0.001f, M_PI));
    h_satellites.push_back(Satellite(7000.0f, 0.001f, 3 * M_PI / 2));

    size_t sat_size = num_sats * sizeof(Satellite);

    // Allocate device memory for satellites
    Satellite* d_satellites;
    cudaMalloc((void**)&d_satellites, sat_size);
    cudaMemcpy(d_satellites, h_satellites.data(), sat_size, cudaMemcpyHostToDevice);

    // Allocate device memory for coverage results
    bool* d_coverage_results;
    cudaMalloc((void**)&d_coverage_results, num_sats * sizeof(bool));

    // Prepare host buffer for coverage results
    std::vector<unsigned char> h_coverage_results(num_sats);

    int block_size = 256;
    int grid_size = (num_sats + block_size - 1) / block_size;

    // Simulation loop
    for (float t = 0.0f; t < simulation_duration; t += delta_time) {
        std::cout << "Time: " << t << "s\n";

        // Update satellite positions on the GPU
        update_satellite_positions<<<grid_size, block_size>>>(d_satellites, num_sats, delta_time);
        cudaDeviceSynchronize();

        // Compute coverage status on the GPU
        compute_coverage<<<grid_size, block_size>>>(d_satellites, num_sats, ground, min_elevation_angle, d_coverage_results);
        cudaDeviceSynchronize();

        // Copy updated satellite data and coverage results back to the host
        cudaMemcpy(h_satellites.data(), d_satellites, sat_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_coverage_results.data(), d_coverage_results, num_sats * sizeof(bool), cudaMemcpyDeviceToHost);

        // Print satellite positions and coverage status for this time step
        for (size_t i = 0; i < h_satellites.size(); ++i) {
            float x, y, z;
            h_satellites[i].compute_position(x, y, z);
            std::cout << "Satellite " << i
                      << " position: (" << x << ", " << y << ", " << z << ") "
                      << (h_coverage_results[i] ? "covers" : "does not cover")
                      << " the ground point.\n";
        } // End for loop to print
        std::cout << "-----------------------------------\n";
    } // End simulation loop

    // Free device memory
    cudaFree(d_satellites);
    cudaFree(d_coverage_results);

    return 0;
} // End main
