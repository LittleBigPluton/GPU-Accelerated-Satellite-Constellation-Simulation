#include<iostream>
#include<vector>
#include"cudainfo.h"
#include"satellite.h"
#include"groundpoint.h"
#include"check_covered.h"

int main(void){
  // Check CUDA availability
  CUDA_check();

  // Create test satellites:
  Satellite sat1(7000.0f, 0.001f, 0.0f);
  Satellite sat2(7000.0f, 0.001f, M_PI / 2);
  Satellite sat3(7000.0f, 0.001f, M_PI);
  Satellite sat4(7000.0f, 0.001f, 3 * M_PI / 2);

  // Store satellites in a vector for testing
  std::vector<Satellite> satellites = {sat1, sat2, sat3, sat4};

  // Create a ground point; for simplicity, assume it's at (0, 0, 0)
  ground_point ground(7000.0f, 0.0f, -100.0f);

  // Define simulation parameters
  const float simulation_duration = 100.0f; // total simulation time in seconds
  const float delta_time = 1.0f;            // time step (seconds)

  // Minimum elevation angle (e.g., 10 degrees in radians)
  const float minElevationAngle = 10.0f * M_PI / 180.0f;

  // Simulation loop
  for (float t = 0.0f; t < simulation_duration; t += delta_time)
  {
    std::cout << "Time: " << t << "s\n";
    // For each satellite, update its position and compute its coordinates.
    for (size_t i = 0; i < satellites.size(); ++i)
    {
      satellites[i].update_position(delta_time);
      float x, y, z;
      satellites[i].compute_position(x, y, z);
      bool coverage = is_covered(satellites[i], ground, minElevationAngle);
      std::cout << "Satellite " << i+1 << " position: (" << x << ", " << y << ", " << z << ") "
                << (coverage ? "covers" : "does not cover") << " the ground point.\n";
    } //For loop to update satellites' positions
    std::cout << "-----------------------------------\n";
  } // For loop to pass time
  return 0;
}
