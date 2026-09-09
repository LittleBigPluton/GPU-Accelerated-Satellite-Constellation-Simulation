#include "ground_point.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace
{

constexpr float TOLERANCE = 1e-6f;

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
    float expected
)
{
    return std::fabs(actual - expected) <= TOLERANCE;
}

void test_constructor_and_getters()
{
    const ground_point ground(
        6371.0f,
        100.0f,
        -50.0f
    );

    require(
        approximately_equal(
            ground.x(),
            6371.0f
        ),
        "Ground-point x coordinate is incorrect."
    );

    require(
        approximately_equal(
            ground.y(),
            100.0f
        ),
        "Ground-point y coordinate is incorrect."
    );

    require(
        approximately_equal(
            ground.z(),
            -50.0f
        ),
        "Ground-point z coordinate is incorrect."
    );
}

} // namespace

int main()
{
    test_constructor_and_getters();

    std::cout
        << "[PASSED] Ground-point tests\n";

    return EXIT_SUCCESS;
}
