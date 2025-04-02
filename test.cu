#include<iostream>
#include<vector>
#include"cudainfo.h"
#include"satellite.h"

int main(void){
  CUDA_check();
  std::vector<Satellite> satellites;
  satellites.emplace_back(7000.0f, 0.001f); // Satellite #1
  satellites.emplace_back(7100.0f, 0.0011f);// Satellite #2
  satellites.emplace_back(7200.0f, 0.0009f);// Satellite #3

  // Define simulation parameters
  const float simulation_duration = 100.0f; // total simulation time in seconds
  const float delta_time = 1.0f;            // time step (seconds)

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
      // Output the satellite's computed position.
      std::cout << "Satellite " << i+1 << ": x = " << x << ", y = " << y << ", z = " << z << "\n";
    } //For loop to update satellites' positions
    std::cout << "-----------------------------------\n";
  } // For loop to pass time
  return 0;
}
