#pragma once

#include <cuda_runtime.h>

#include "satellite.hpp"
#include "ground_point.hpp"

__host__ __device__
bool is_covered(const Satellite& sat, const ground_point& ground, float min_elevation_angle);
