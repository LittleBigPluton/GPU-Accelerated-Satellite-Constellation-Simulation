/**
 * @file satellite.h
 * @brief This file contains the declaration of the Satellite class.
 */

#ifndef SATELLITE_H
#define SATELLITE_H
#include <cmath>

/**
 * @class Satellite
 * @brief Class to represent a satellite, update and compute its position.
 */
class Satellite
{
public:
  /**
   * @brief Constructor with parameters.
   * @param radius The radius of the satellite's orbit.
   * @param angular_velocity The angular velocity of the satellite.
   * @param current_angle The current angle of the satellite (default is 0.0).
   */
  Satellite(float radius, float angular_velocity, float current_angle = 0.0f)
    : satellite_radius(radius), satellite_ang_vel(angular_velocity), satellite_cur_ang(current_angle) {}

  /**
   * @brief Updates the satellite's position based on elapsed time.
   * @param delta_time Elapsed time in seconds.
   */
  __host__ __device__
  void update_position(float delta_time)
  {
    // Update the current angle
    satellite_cur_ang += satellite_ang_vel * delta_time;
    // Normalize the angle to stay within 0 to 2π range
    if (satellite_cur_ang > 2 * M_PI) satellite_cur_ang -= 2 * M_PI;
  }

  /**
   * @brief Computes the satellite's position in (x, y, z) coordinates on a circular orbit (z = 0).
   * @param x Reference to the x-coordinate of the satellite.
   * @param y Reference to the y-coordinate of the satellite.
   * @param z Reference to the z-coordinate of the satellite.
   */
  __host__ __device__
  void compute_position(float& x, float& y, float& z) const
  {
    x = satellite_radius * cos(satellite_cur_ang);
    y = satellite_radius * sin(satellite_cur_ang);
    z = 0;
  }

private:
  float satellite_radius;
  float satellite_ang_vel;
  float satellite_cur_ang;
};

#endif
