#include <xtgeo/geometry.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using xyz::Point;

// local helper function to interpolate Z in a triangle
static double
interpolate_z_triangles(const double x,
                        const double y,
                        const Point &p1,
                        const Point &p2,
                        const Point &p3,
                        const Point &p4,
                        const double tolerance = numerics::TOLERANCE)
{
    Point apply_point = { x, y, 0.0 };

    double area1 = triangle_area(apply_point, p2, p3);
    double area2 = triangle_area(p1, apply_point, p3);
    double area3 = triangle_area(p1, p2, apply_point);
    double total_area = triangle_area(p1, p2, p3);

    double z_estimated = std::numeric_limits<double>::quiet_NaN();

    // Check if the point is in the triangle
    if (std::abs(total_area - (area1 + area2 + area3)) < tolerance) {
        double w1 = area1 / total_area;
        double w2 = area2 / total_area;
        double w3 = area3 / total_area;
        z_estimated = w1 * p1.z + w2 * p2.z + w3 * p3.z;
    }
    // check that z_estimated is not NaN
    if (!std::isnan(z_estimated)) {
        return z_estimated;
    }
    // if still NaN, check the other triangle
    area1 = triangle_area(apply_point, p3, p4);
    area2 = triangle_area(p1, apply_point, p4);
    area3 = triangle_area(p1, p3, apply_point);
    total_area = triangle_area(p1, p3, p4);

    // Check if the point is in the triangle
    if (std::abs(total_area - (area1 + area2 + area3)) < tolerance) {
        double w1 = area1 / total_area;
        double w2 = area2 / total_area;
        double w3 = area3 / total_area;
        z_estimated = w1 * p1.z + w2 * p3.z + w3 * p4.z;
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
 * @param p1 Point 1 instance
 * @param p2 Point 2 instance
 * @param p3 Point 3 instance
 * @param p4 Point 4 instance
 * @return double
 */
double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const Point &p1,
                         const Point &p2,
                         const Point &p3,
                         const Point &p4,
                         const double tolerance)
{
    // Quick check if the point is inside the quadrilateral; note ordering of points
    if (!is_xy_point_in_quadrilateral(x, y, p1, p2, p4, p3, tolerance)) {
        return numerics::QUIET_NAN;
    }

    // Bilinear interpolation
    double denom = (p2.x - p1.x) * (p3.y - p1.y);
    double z_estimated =
      ((p2.x - x) * (p3.y - y) * p1.z + (x - p1.x) * (p3.y - y) * p2.z +
       (p2.x - x) * (y - p1.y) * p3.z + (x - p1.x) * (y - p1.y) * p4.z) /
      denom;

    return z_estimated;
}  // interpolate_z_4p_regular

/*
 * Function to interpolate Z by averaging the results from two alternative
 * arrangements in a quadrilateral Function to interpolate Z using barycentric
 * coordinates (non regular)
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
                 const Point &p1,
                 const Point &p2,
                 const Point &p3,
                 const Point &p4,
                 const double tolerance)
{
    // Quick check if the point is inside the quadrilateral (note order)
    if (!is_xy_point_in_quadrilateral(x, y, p1, p2, p4, p3, tolerance)) {
        return numerics::QUIET_NAN;
    }

    double z1 = interpolate_z_triangles(x, y, p1, p2, p3, p4, tolerance);
    double z2 = interpolate_z_triangles(x, y, p2, p1, p4, p3, tolerance);

    // If either interpolation returns NaN, return the other value
    if (std::isnan(z1))
        return z2;
    if (std::isnan(z2))
        return z1;

    // Return the average of the two interpolated Z-values
    return (z1 + z2) / 2.0;

}  // interpolate_z_4p

/*
 * Function that given a midpoint in a rectangular cell with known increments and
 * rotation, returns the XY corners of the cell
 * @param x X coordinate of the midpoint
 * @param y Y coordinate of the midpoint
 * @param xinc X increment
 * @param yinc Y increment
 * @param rot Rotation in degrees
 * @return std::array<double, 8>
 */
std::array<double, 8>
find_rect_corners_from_center(const double x,
                              const double y,
                              const double xinc,
                              const double yinc,
                              const double rot)
{
    double rrot = rot * M_PI / 180.0;

    double cv = cos(rrot);
    double sv = sin(rrot);

    double r1x = -0.5 * xinc * cv - 0.5 * yinc * sv;
    double r1y = -0.5 * xinc * sv + 0.5 * yinc * cv;
    double r2x = 0.5 * xinc * cv - 0.5 * yinc * sv;
    double r2y = 0.5 * xinc * sv + 0.5 * yinc * cv;

    return { x + r1x, y + r1y, x + r2x, y + r2y, x - r1x, y - r1y, x - r2x, y - r2y };
}  // find_rect_corners_from_center

}  // namespace xtgeo::geometry
