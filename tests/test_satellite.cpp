#include "satellite.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float TOLERANCE = 1e-3f;

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

bool approximately_equal(
    float actual,
    float expected,
    float tolerance = TOLERANCE
)
{
    return std::fabs(actual - expected) <= tolerance;
}

void test_initial_position()
{
    const Satellite satellite(
        7000.0f,
        0.001f,
        0.0f
    );

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    satellite.compute_position(x, y, z);

    require(
        approximately_equal(x, 7000.0f),
        "Initial x position should equal orbital radius."
    );

    require(
        approximately_equal(y, 0.0f),
        "Initial y position should be zero."
    );

    require(
        approximately_equal(z, 0.0f),
        "Initial z position should be zero."
    );
}

void test_position_after_update()
{
    constexpr float radius = 7000.0f;
    constexpr float angular_velocity = 0.001f;
    constexpr float delta_time = 10.0f;

    Satellite satellite(
        radius,
        angular_velocity,
        0.0f
    );

    satellite.update_position(delta_time);

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    satellite.compute_position(x, y, z);

    const float expected_angle =
        angular_velocity * delta_time;

    const float expected_x =
        radius * std::cos(expected_angle);

    const float expected_y =
        radius * std::sin(expected_angle);

    require(
        approximately_equal(x, expected_x),
        "Updated x position is incorrect."
    );

    require(
        approximately_equal(y, expected_y),
        "Updated y position is incorrect."
    );

    require(
        approximately_equal(z, 0.0f),
        "Updated z position should remain zero."
    );
}

void test_angle_wrapping()
{
    // Radius 1 avoids amplifying tiny floating-point angular
    // differences into larger position differences.
    constexpr float radius = 1.0f;

    const float initial_angle =
        TWO_PI - 0.05f;

    constexpr float angular_velocity = 0.10f;
    constexpr float delta_time = 1.0f;

    Satellite satellite(
        radius,
        angular_velocity,
        initial_angle
    );

    satellite.update_position(delta_time);

    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    satellite.compute_position(x, y, z);

    constexpr float expected_angle = 0.05f;

    const float expected_x =
        radius * std::cos(expected_angle);

    const float expected_y =
        radius * std::sin(expected_angle);

    require(
        approximately_equal(
            x,
            expected_x,
            1e-5f
        ),
        "Wrapped x position is incorrect."
    );

    require(
        approximately_equal(
            y,
            expected_y,
            1e-5f
        ),
        "Wrapped y position is incorrect."
    );

    require(
        approximately_equal(
            z,
            0.0f,
            1e-6f
        ),
        "Wrapped z position should remain zero."
    );
}

} // namespace

int main()
{
    test_initial_position();
    test_position_after_update();
    test_angle_wrapping();

    std::cout
        << "[PASSED] Satellite tests\n";

    return EXIT_SUCCESS;
}
