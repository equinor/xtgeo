#include <pybind11/pybind11.h>
#include <array>
#include <cmath>
#include <xtgeo/geometry.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::geometry {

using xyz::Point;

/*
 * Function to find if a point is within a quadrilateral. The vertices must be clockwise
 * or counter-clockwise ordered.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param p1 Point 1 instance
 * @param p2 Point 2 instance
 * @param p3 Point 3 instance
 * @param p4 Point 4 instance
 * @param tolerance Tolerance for considering a point on the line
 * @return Boolean
 */

bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const Point &p1,
                             const Point &p2,
                             const Point &p3,
                             const Point &p4,
                             const double tolerance)
{
    // anonymous lambda function to check if a point is to the left of a line or on the
    // line
    auto is_left_or_on = [tolerance](const Point &a, const Point &b,
                                     const std::array<double, 2> &c) {
        return ((b.x() - a.x()) * (c[1] - a.y()) - (c[0] - a.x()) * (b.y() - a.y())) >=
               -tolerance;
    };

    std::array<double, 2> p = { x, y };

    int winding_number = 0;

    if (is_left_or_on(p1, p2, p))
        winding_number++;
    else
        winding_number--;

    if (is_left_or_on(p2, p3, p))
        winding_number++;
    else
        winding_number--;

    if (is_left_or_on(p3, p4, p))
        winding_number++;
    else
        winding_number--;

    if (is_left_or_on(p4, p1, p))
        winding_number++;
    else
        winding_number--;

    return winding_number == 4 || winding_number == -4;
}

}  // namespace xtgeo::geometry
