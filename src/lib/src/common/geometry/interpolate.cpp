#include <xtgeo/constants.hpp>
#include <xtgeo/geometry.hpp>
#include <xtgeo/numerics.hpp>

namespace xtgeo::geometry {

// local helper function to interpolate Z in a triangle
static double
interpolate_z_triangles(const double x,
                        const double y,
                        const std::array<double, 3> &p1,
                        const std::array<double, 3> &p2,
                        const std::array<double, 3> &p3,
                        const std::array<double, 3> &p4)
{
    // Convert 3D points to 2D points for area calculation, generic
    std::array<double, 2> p1_2d = { p1[0], p1[1] };
    std::array<double, 2> p2_2d = { p2[0], p2[1] };
    std::array<double, 2> p3_2d = { p3[0], p3[1] };
    std::array<double, 2> p4_2d = { p4[0], p4[1] };
    std::array<double, 2> p = { x, y };

    // Compute areas of the triangles
    double area1 = triangle_area(p, p2_2d, p3_2d);
    double area2 = triangle_area(p1_2d, p, p3_2d);
    double area3 = triangle_area(p1_2d, p2_2d, p);
    double total_area = triangle_area(p1_2d, p2_2d, p3_2d);

    double z_estimated = std::numeric_limits<double>::quiet_NaN();

    // Check if the point is in the triangle
    if (std::abs(total_area - (area1 + area2 + area3)) < numerics::TOLERANCE) {
        double w1 = area1 / total_area;
        double w2 = area2 / total_area;
        double w3 = area3 / total_area;
        z_estimated = w1 * p1[2] + w2 * p2[2] + w3 * p3[2];
    }
    // check that z_estimated is not NaN
    if (!std::isnan(z_estimated)) {
        return z_estimated;
    }
    // if still NaN, check the other triangle
    area1 = triangle_area(p, p3_2d, p4_2d);
    area2 = triangle_area(p1_2d, p, p4_2d);
    area3 = triangle_area(p1_2d, p3_2d, p);
    total_area = triangle_area(p1_2d, p3_2d, p4_2d);

    // Check if the point is in the triangle
    if (std::abs(total_area - (area1 + area2 + area3)) < numerics::TOLERANCE) {
        double w1 = area1 / total_area;
        double w2 = area2 / total_area;
        double w3 = area3 / total_area;
        z_estimated = w1 * p1[2] + w2 * p3[2] + w3 * p4[2];
    }

    // If the point is not in any triangles, return NaN
    return z_estimated;

}  // _interpolate_z_triangle

/*
 * A generic function to estimate Z within 4 corners in a regular XYZ space.
 * 3                   4
 *  x-----------------x
 *  |                 |
 *  |                 |  (or flipped)
 *  |                 |
 *  x-----------------x
 * 1                   2
 *
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param p1 A vector of doubles, length 3
 * @param p2 A vector of doubles, length 3
 * @param p3 A vector of doubles, length 3
 * @param p4 A vector of doubles, length 3
 * @return double
 */
double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const std::array<double, 3> &p1,
                         const std::array<double, 3> &p2,
                         const std::array<double, 3> &p3,
                         const std::array<double, 3> &p4)
{

    // Quick check if the point is inside the quadrilateral; note ordering of points
    if (!is_xy_point_in_quadrilateral(x, y, p1, p2, p4, p3)) {
        return numerics::QUIET_NAN;
    }

    // Extract coordinates
    double x1 = p1[0], y1 = p1[1], z1 = p1[2];
    double x2 = p2[0], y2 = p2[1], z2 = p2[2];
    double x3 = p3[0], y3 = p3[1], z3 = p3[2];
    double x4 = p4[0], y4 = p4[1], z4 = p4[2];

    // Bilinear interpolation
    double denom = (x2 - x1) * (y3 - y1);
    double z_estimated = ((x2 - x) * (y3 - y) * z1 + (x - x1) * (y3 - y) * z2 +
                          (x2 - x) * (y - y1) * z3 + (x - x1) * (y - y1) * z4) /
                         denom;

    return z_estimated;
}  // interpolate_z_4p_regular

/*
 * Function to interpolate Z by averaging the results from two alternative arrangements
 * in a quadrilateral
 * Function to interpolate Z using barycentric coordinates (non regular)
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param p1 A vector of doubles, length 3
 * @param p2 A vector of doubles, length 3
 * @param p3 A vector of doubles, length 3
 * @param p4 A vector of doubles, length 3
 * @return double
 */

double
interpolate_z_4p(const double x,
                 const double y,
                 const std::array<double, 3> &p1,
                 const std::array<double, 3> &p2,
                 const std::array<double, 3> &p3,
                 const std::array<double, 3> &p4)
{
    // Quick check if the point is inside the quadrilateral
    if (!is_xy_point_in_quadrilateral(x, y, p1, p2, p4, p3)) {
        return numerics::QUIET_NAN;
    }

    double z1 = interpolate_z_triangles(x, y, p1, p2, p3, p4);
    double z2 = interpolate_z_triangles(x, y, p2, p1, p4, p3);

    // If either interpolation returns NaN, return the other value
    if (std::isnan(z1))
        return z2;
    if (std::isnan(z2))
        return z1;

    // Return the average of the two interpolated Z-values
    return (z1 + z2) / 2.0;

}  // interpolate_z_4p

}  // namespace xtgeo::geometry
