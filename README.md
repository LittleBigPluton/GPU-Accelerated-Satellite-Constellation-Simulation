# GPU-Accelerated Satellite Constellation Simulation

A C++17/CUDA simulation project for evaluating satellite motion and ground-point coverage on both CPU and GPU.

The project focuses on parallelizing independent satellite updates and elevation-angle coverage calculations with CUDA while preserving a shared CPU/GPU implementation for correctness. On the tested hardware, the GPU-resident simulation achieved a median **46.84× end-to-end speedup** over the single-threaded CPU baseline for **1,000,000 satellites across 100 timesteps**.

## Highlights

- C++17 and CUDA implementation
- Shared host/device satellite and coverage logic
- CPU and GPU simulation paths
- Elevation-angle based ground coverage evaluation
- Deterministic benchmark workloads
- Automated correctness tests with CTest
- Explicit CPU/GPU consistency validation
- Separate kernel-level and end-to-end performance measurements
- GPU-resident multi-step benchmark to measure transfer amortization
- Median **46.84× end-to-end acceleration** for the largest tested resident workload

---

## Simulation Model

The project uses a deliberately simplified orbital model designed to isolate the computational structure of constellation simulation and GPU acceleration.

Each satellite is represented by:

- orbital radius
- angular velocity
- current orbital angle

The satellite follows a circular orbit in the Cartesian `x-y` plane:

```text
x = r cos(theta)
y = r sin(theta)
z = 0
```

The angle evolves according to:

```text
theta(t + dt) = theta(t) + omega * dt
```

and is wrapped into the interval `[0, 2π)`.

This is **not a high-fidelity orbital propagator**. Effects such as orbital perturbations, inclination changes, Earth rotation, atmospheric drag, and multi-body dynamics are outside the current scope.

---

## Ground Coverage Model

Coverage is determined using the satellite-to-ground line-of-sight vector and the ground point's radial vector.

For satellite position `s` and ground position `g`:

```text
LOS = s - g
```

The ground radial vector acts as the local upward direction.

The elevation angle is computed from:

```text
sin(elevation) =
    dot(LOS, g) /
    (|LOS| |g|)
```

A satellite is considered visible when:

```text
elevation > minimum_elevation_angle
```

The benchmark uses a minimum elevation threshold of **10°**.

---

## CPU and GPU Execution

The same satellite and coverage implementations are available to both host and device code.

The CPU execution path processes satellites sequentially:

```text
Satellite 0 -> update -> coverage
Satellite 1 -> update -> coverage
Satellite 2 -> update -> coverage
...
```

The CUDA implementation assigns independent satellites to GPU threads:

```text
Thread 0 -> Satellite 0
Thread 1 -> Satellite 1
Thread 2 -> Satellite 2
...
```

Two CUDA kernels are used:

```text
update_satellites_kernel
        |
        v
compute_coverage_kernel
```

This structure maps naturally to GPU execution because satellite updates and coverage evaluations are independent for each satellite.

---

## Project Structure

```text
GPU-Accelerated-Satellite-Constellation-Simulation/
├── include/
│   ├── satellite.hpp
│   ├── ground_point.hpp
│   ├── check_covered.hpp
│   └── cuda_info.hpp
│
├── src/
│   ├── satellite.cu
│   ├── ground_point.cu
│   ├── check_covered.cu
│   └── cuda_info.cu
│
├── tests/
│   ├── test_satellite.cpp
│   ├── test_ground_point.cpp
│   ├── test_coverage.cpp
│   └── test_cpu_gpu_consistency.cu
│
├── benchmarks/
│   └── benchmark.cu
│
├── cpu_simulation.cpp
├── cpu_coverage_simulation.cpp
├── gpu_simulation.cu
└── CMakeLists.txt
```

The core simulation logic is separated from executable programs, tests, and benchmarks.

---

## Requirements

- C++17 compatible compiler
- CMake 3.20 or newer
- NVIDIA CUDA Toolkit
- CUDA-capable NVIDIA GPU for GPU execution
- NVIDIA driver compatible with the installed CUDA Toolkit

The benchmark results below were produced with:

| Component | Configuration |
|---|---|
| CPU | Intel Core i5-8250U @ 1.60 GHz |
| CPU baseline | Single-threaded |
| GPU | NVIDIA GeForce MX150 |
| Compute capability | 6.1 |
| GPU memory | ~1994 MiB |
| CUDA Toolkit | 12.4 |
| NVCC | 12.4.131 |
| C++ compiler | G++ 13.4.0 |
| NVIDIA driver | 580.159.03 |
| Build type | Release |
| CUDA architecture | `sm_61` |

> The MX150 is a Pascal-generation GPU. CUDA architecture `61` is therefore used for this test system. Use the appropriate architecture for your own GPU.

---

## Build

Create an optimized build:

```bash
cmake -S . -B build-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=61
```

Then compile:

```bash
cmake --build build-release -j
```

On systems where an explicit host compiler is required:

```bash
cmake -S . -B build-release \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_ARCHITECTURES=61
```

---

## Run

### CPU position simulation

```bash
./build-release/cpu_simulation
```

### CPU coverage simulation

```bash
./build-release/cpu_coverage_simulation
```

### GPU simulation

```bash
./build-release/gpu_simulation
```

### Benchmark

```bash
./build-release/satellite_benchmark
```

---

## Tests

The project includes four automated tests:

| Test | Purpose |
|---|---|
| `satellite` | Initial position, position updates, and angle wrapping |
| `ground_point` | Ground-point construction and coordinate access |
| `coverage` | Covered and non-covered elevation-angle cases |
| `cpu_gpu_consistency` | CPU/GPU position and coverage agreement |

Run all tests with:

```bash
cd build-release
ctest --output-on-failure
```

Expected result:

```text
100% tests passed, 0 tests failed
```

The CPU/GPU consistency test compares both implementations using the same initial constellation and simulation parameters.

---

# Benchmark

The benchmark evaluates the same workload on the serial CPU implementation and the CUDA implementation.

Two scenarios are measured:

1. **Single-step execution**
2. **GPU-resident multi-step execution**

GPU memory allocation is excluded from timing in both cases.

Correctness validation is performed outside the timed regions.

---

## Benchmark Methodology

### Single-Step Benchmark

One simulation step consists of:

```text
Host -> Device satellite transfer
            |
            v
      Position update
            |
            v
     Coverage evaluation
            |
            v
Device -> Host coverage transfer
```

Each workload size is executed **20 times** and the median is reported for each independent benchmark execution.

End-to-end timing includes:

- host-to-device satellite transfer
- kernel execution
- synchronization/runtime overhead
- device-to-host coverage transfer

It therefore represents a conservative case where input data begins on the host and the result is required back on the host after one step.

---

## Single-Step Results

Final values are the median across **three independent benchmark executions**.

| Satellites | CPU Compute | GPU Kernels | GPU End-to-End | Kernel Speedup | End-to-End Speedup |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 0.0826 ms | 0.0102 ms | 0.0417 ms | 8.06× | 1.98× |
| 10,000 | 0.8548 ms | 0.0288 ms | 0.1632 ms | 30.66× | 5.24× |
| 100,000 | 9.5530 ms | 0.2020 ms | 1.2694 ms | 47.41× | 7.55× |
| 1,000,000 | 93.0149 ms | 1.9140 ms | 11.0439 ms | 48.55× | **8.73×** |

For large workloads, the CUDA kernels provide substantial acceleration, while host-device transfer and synchronization overhead reduce the end-to-end speedup.

---

## GPU-Resident Multi-Step Benchmark

A simulation normally performs multiple timesteps over the same satellite state.

The resident benchmark therefore keeps the constellation on the GPU:

```text
Host -> Device
      |
      v
+----------------------+
| Position update      |
| Coverage evaluation  | x 100 timesteps
+----------------------+
      |
      v
Device -> Host
```

Only one initial satellite-state transfer is performed.

Satellite state then remains on the GPU for all **100 timesteps**, followed by one final coverage transfer back to the host.

Each workload is executed **5 times** per benchmark execution.

A separate satellite-state transfer used only for correctness validation is excluded from timing.

---

## Resident Multi-Step Results

Final values are the median across **three independent benchmark executions**.

| Workload | CPU Total | GPU Kernel Loop | GPU End-to-End | Kernel Speedup | End-to-End Speedup |
|---:|---:|---:|---:|---:|---:|
| 100,000 × 100 steps | 878.12 ms | 19.72 ms | 20.78 ms | 44.52× | **42.20×** |
| 1,000,000 × 100 steps | 8740.33 ms | 177.36 ms | 186.61 ms | 49.25× | **46.84×** |

For the largest workload:

```text
100,000,000 satellite-step evaluations

CPU:             8740.33 ms
GPU kernel loop:  177.36 ms
GPU end-to-end:   186.61 ms

Kernel speedup:      49.25x
End-to-end speedup:  46.84x
```

Keeping simulation state resident on the GPU allows transfer overhead to be amortized across repeated timesteps.

The end-to-end speedup therefore approaches the kernel-level speedup as the amount of computation performed per transfer increases.

---

## Benchmark Validation

Every benchmark workload validates the final CPU and GPU coverage results.

The resident benchmark additionally checks sampled final satellite positions to verify that repeated CPU and GPU state updates remain consistent.

All workloads passed validation in all three final benchmark executions.

---

## Interpreting the Results

The benchmark demonstrates two different GPU-computing regimes.

### Transfer-dominated execution

For a single simulation step:

```text
CPU                         GPU
----                        ----
Compute                     H2D transfer
                            Kernels
                            D2H transfer
```

The GPU computation itself is substantially faster, but transfer and runtime overhead reduce the overall benefit.

At 1,000,000 satellites:

```text
Kernel speedup:      48.55x
End-to-end speedup:   8.73x
```

### Compute-dominated resident execution

When satellite state stays on the device across many timesteps:

```text
H2D once
   |
   v
GPU computation x 100
   |
   v
D2H once
```

transfer cost is spread across much more computation.

For 1,000,000 satellites across 100 timesteps:

```text
Kernel speedup:      49.25x
End-to-end speedup:  46.84x
```

This illustrates why data residency and transfer strategy are important parts of GPU application design, not only kernel parallelization.

---

## Benchmark Reproducibility

The published results use:

```text
3 independent benchmark executions
```

For each execution:

```text
Single-step:
    20 repetitions per workload
    median reported

Resident multi-step:
    5 repetitions per workload
    100 timesteps
    median reported
```

The values shown in this README are the median of the three independent execution medians.

The benchmark uses deterministic satellite initialization based on a golden-angle phase distribution, allowing repeatable CPU/GPU comparisons without relying on random initialization.

Performance results are hardware- and system-dependent and should not be interpreted as universal CUDA speedups.

---

## Scope and Limitations

This project is intended as a CUDA parallel-computing and simulation exercise rather than a production astrodynamics package.

Current simplifications include:

- circular planar satellite orbits
- constant angular velocity
- fixed orbital radius
- fixed Cartesian ground point
- spherical Earth-centered coverage geometry
- no Earth rotation
- no orbital inclination model
- no atmospheric drag
- no gravitational perturbations
- no high-fidelity orbital propagator
- single-threaded CPU reference implementation

The benchmark therefore measures acceleration of the implemented computational workload, not the performance of a complete operational satellite constellation simulator.

---

## Possible Extensions

Potential future improvements include:

- inclined and three-dimensional orbital planes
- multiple ground stations
- time-dependent Earth rotation
- more realistic orbital propagation
- multi-ground-point coverage kernels
- CUDA streams and asynchronous transfers
- pinned host memory
- fused update-and-coverage kernels
- multi-threaded CPU comparison
- profiling with NVIDIA Nsight
- larger-scale constellation experiments

---

## Summary

This project demonstrates how an initially serial satellite-coverage workload can be structured for CUDA execution while maintaining CPU/GPU correctness.

The main result is not only faster kernel execution, but the effect of data movement on real application performance:

```text
1,000,000 satellites, single timestep:
    48.55x kernel speedup
     8.73x end-to-end speedup

1,000,000 satellites, 100 GPU-resident timesteps:
    49.25x kernel speedup
    46.84x end-to-end speedup
```

The comparison highlights a central GPU-programming principle: **keeping repeatedly used simulation state on the device can be as important as parallelizing the computation itself.**
