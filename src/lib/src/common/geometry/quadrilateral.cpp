#include <pybind11/pybind11.h>
#include <array>
#include <xtgeo/geometry.hpp>

namespace py = pybind11;

namespace xtgeo::geometry {

// helper function to determine if a point is to the left of a line
static bool
is_left(const std::array<double, 2> &p0,
        const std::array<double, 2> &p1,
        const std::array<double, 2> &p2)
{
    return ((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])) > 0;
}

/*
 * Function to find if a point is within a quadrilateral. The vertices must be clockwise
 * or counter-clockwise ordered.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param p1 A vector of doubles, length 3
 * @param p2 A vector of doubles, length 3
 * @param p3 A vector of doubles, length 3
 * @param p4 A vector of doubles, length 3
 * @return Boolean
 */
bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const std::array<double, 3> &p1,
                             const std::array<double, 3> &p2,
                             const std::array<double, 3> &p3,
                             const std::array<double, 3> &p4)
{
    // Create an array of points
    std::array<std::array<double, 3>, 4> points = { p1, p2, p3, p4 };

    // Convert 3D points to 2D points
    std::array<double, 2> p1_2d = { points[0][0], points[0][1] };
    std::array<double, 2> p2_2d = { points[1][0], points[1][1] };
    std::array<double, 2> p3_2d = { points[2][0], points[2][1] };
    std::array<double, 2> p4_2d = { points[3][0], points[3][1] };
    std::array<double, 2> p = { x, y };

    // Check if the point is inside the quadrilateral using the winding number
    int winding_number = 0;

    if (is_left(p1_2d, p2_2d, p))
        winding_number++;
    else
        winding_number--;

    if (is_left(p2_2d, p3_2d, p))
        winding_number++;
    else
        winding_number--;

    if (is_left(p3_2d, p4_2d, p))
        winding_number++;
    else
        winding_number--;

    if (is_left(p4_2d, p1_2d, p))
        winding_number++;
    else
        winding_number--;

    return winding_number == 4 || winding_number == -4;
}  // is_xy_point_in_quadrilateral

}  // namespace xtgeo::geometry
