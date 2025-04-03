
#ifndef CHECK_COVERED_H
#define CHECK_COVERED_H

bool is_covered(const Satellite& sat, const ground_point& ground, float minElevationAngle) {
    // Compute satellite position
    float satX, satY, satZ;
    sat.compute_position(satX, satY, satZ);

    // Calculate the vector from ground point to satellite
    float vx = satX - ground.x();
    float vy = satY - ground.y();
    float vz = satZ - ground.z();

    // Calculate the distance between satellite and ground point
    float distance = std::sqrt(vx * vx + vy * vy + vz * vz);

    // Calculate the elevation angle using arcsin(vz / distance)
    float elevationAngle = std::asin(vz / distance); // in radians

    // Determine if the satellite is covering the ground point
    return elevationAngle > minElevationAngle;
}

#endif
