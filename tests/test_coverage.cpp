#include "check_covered.hpp"
#include "ground_point.hpp"
#include "satellite.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

constexpr float PI = 3.14159265358979323846f;
constexpr float DEG_TO_RAD = PI / 180.0f;

constexpr float EARTH_RADIUS = 6371.0f;
constexpr float ORBITAL_RADIUS = 7000.0f;

constexpr float MIN_ELEVATION_ANGLE =
    10.0f * DEG_TO_RAD;

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

void test_directly_overhead_is_covered()
{
    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    const Satellite satellite(
        ORBITAL_RADIUS,
        0.0f,
        0.0f
    );

    const bool covered = is_covered(
        satellite,
        ground,
        MIN_ELEVATION_ANGLE
    );

    require(
        covered,
        "Satellite directly overhead should cover the ground point."
    );
}

void test_opposite_side_is_not_covered()
{
    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    const Satellite satellite(
        ORBITAL_RADIUS,
        0.0f,
        PI
    );

    const bool covered = is_covered(
        satellite,
        ground,
        MIN_ELEVATION_ANGLE
    );

    require(
        !covered,
        "Satellite on the opposite side should not cover the ground point."
    );
}

void test_above_minimum_elevation_is_covered()
{
    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    // At 0.25 rad orbital angle this geometry gives
    // an elevation comfortably above 10 degrees.
    const Satellite satellite(
        ORBITAL_RADIUS,
        0.0f,
        0.25f
    );

    const bool covered = is_covered(
        satellite,
        ground,
        MIN_ELEVATION_ANGLE
    );

    require(
        covered,
        "Satellite above minimum elevation should provide coverage."
    );
}

void test_below_minimum_elevation_is_not_covered()
{
    const ground_point ground(
        EARTH_RADIUS,
        0.0f,
        0.0f
    );

    // At 0.35 rad orbital angle this geometry gives
    // an elevation comfortably below 10 degrees.
    const Satellite satellite(
        ORBITAL_RADIUS,
        0.0f,
        0.35f
    );

    const bool covered = is_covered(
        satellite,
        ground,
        MIN_ELEVATION_ANGLE
    );

    require(
        !covered,
        "Satellite below minimum elevation should not provide coverage."
    );
}

} // namespace

int main()
{
    test_directly_overhead_is_covered();
    test_opposite_side_is_not_covered();
    test_above_minimum_elevation_is_covered();
    test_below_minimum_elevation_is_not_covered();

    std::cout
        << "[PASSED] Coverage tests\n";

    return EXIT_SUCCESS;
}
