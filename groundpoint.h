/**
 * @file groundpoint.h
 * @brief This file contains the declaration of the GroundPoint class.
 */

#ifndef GROUND_POINT_H
#define GROUND_POINT_H

/**
 * @class ground_point
 * @brief Represents a point on the ground in a 3D coordinate system.
 *
 * This class provides simple getter methods to access the x, y, and z coordinates.
 */
class ground_point
{
public:
    /**
     * @brief Constructor to initialize a ground point with coordinates.
     * @param x The x-coordinate of the ground point.
     * @param y The y-coordinate of the ground point.
     * @param z The z-coordinate of the ground point.
     */
    ground_point(float x, float y, float z)
        : m_x(x), m_y(y), m_z(z) {}

    /**
     * @brief Gets the x-coordinate of the ground point.
     * @return The x-coordinate as a float.
     */
    float x() const { return m_x; }

    /**
     * @brief Gets the y-coordinate of the ground point.
     * @return The y-coordinate as a float.
     */
    float y() const { return m_y; }

    /**
     * @brief Gets the z-coordinate of the ground point.
     * @return The z-coordinate as a float.
     */
    float z() const { return m_z; }

private:
    float m_x; ///< The x-coordinate of the ground point.
    float m_y; ///< The y-coordinate of the ground point.
    float m_z; ///< The z-coordinate of the ground point.
}; // End of the ground_point class

#endif // GROUNDPOINT_H
