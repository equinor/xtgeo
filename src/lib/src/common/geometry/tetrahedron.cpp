#include <algorithm>  // For std::min and std::max
#include <array>      // For std::array
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using geometry::matrix::invert_matrix3x3;
using geometry::matrix::Matrix3x3;
using geometry::matrix::multiply_matrix_vector;
using geometry::point::cross_product;
using geometry::point::dot_product;
using geometry::point::subtract;
using xyz::Point;

/**
 * Check if a point is inside a triangle in 2D space.
 */
static bool
is_point_in_triangle_2d(const Point &p, const Point &a, const Point &b, const Point &c)
{
    // Calculate barycentric coordinates in 2D
    double area =
      0.5 * (-b.y * c.x + a.y * (-b.x + c.x) + a.x * (b.y - c.y) + b.x * c.y);
    double s = 1.0 / (2.0 * area) *
               (a.y * c.x - a.x * c.y + (c.y - a.y) * p.x + (a.x - c.x) * p.y);
    double t = 1.0 / (2.0 * area) *
               (a.x * b.y - a.y * b.x + (a.y - b.y) * p.x + (b.x - a.x) * p.y);

    // Check if point is inside the triangle
    return s >= 0 && t >= 0 && (s + t) <= 1;
}

/**
 * Calculate the bounding box of a tetrahedron defined by four points.
 * Returns a pair of points representing the minimum and maximum corners of the bounding
 * box.
 */
static std::pair<Point, Point>
get_tetrahedron_bounding_box(const Point &a,
                             const Point &b,
                             const Point &c,
                             const Point &d)
{
    // Initialize min and max points
    Point min_pt = { std::min({ a.x, b.x, c.x, d.x }), std::min({ a.y, b.y, c.y, d.y }),
                     std::min({ a.z, b.z, c.z, d.z }) };

    Point max_pt = { std::max({ a.x, b.x, c.x, d.x }), std::max({ a.y, b.y, c.y, d.y }),
                     std::max({ a.z, b.z, c.z, d.z }) };

    return { min_pt, max_pt };
}

double
signed_tetrahedron_volume(const Point &a,
                          const Point &b,
                          const Point &c,
                          const Point &d)
{
    Point ab = subtract(b, a);
    Point ac = subtract(c, a);
    Point ad = subtract(d, a);
    return dot_product(cross_product(ab, ac), ad) / 6.0;
}

static bool
is_tetrahedron_degenerate(const Point &a,
                          const Point &b,
                          const Point &c,
                          const Point &d,
                          double tolerance = 1e-6)
{

    double volume = std::abs(signed_tetrahedron_volume(a, b, c, d));

    return volume < tolerance;
}

static bool
is_tetrahedron_degenerate_from_volume(double volume, double tolerance = 1e-6)
{
    return volume < tolerance;
}

/**
 * Helper function to calculate relative tolerance based on volumes of tetrahedra.
 * Currently inactive, but kept for reference
 */
static double
calculate_max_volume(const std::array<double, 5> &volumes)
{
    // Find the maximum absolute volume
    double max_volume = 0.0;
    for (const auto &volume : volumes) {
        max_volume = std::max(max_volume, std::abs(volume));
    }

    return max_volume;
}

static double
calculate_bbox_diagonal(const Point &min_pt, const Point &max_pt)
{
    // Calculate the diagonal length of the bounding box
    double diagonal =
      std::sqrt(std::pow(max_pt.x - min_pt.x, 2) + std::pow(max_pt.y - min_pt.y, 2) +
                std::pow(max_pt.z - min_pt.z, 2));

    return diagonal;
}

static bool
is_point_in_tetrahedron_bounding_box(const Point &p,
                                     const Point &a,
                                     const Point &b,
                                     const Point &c,
                                     const Point &d)
{
    // Get the bounding box of the tetrahedron
    auto [min_pt, max_pt] = get_tetrahedron_bounding_box(a, b, c, d);

    // Check if the point is within the bounding box
    return (p.x >= min_pt.x && p.x <= max_pt.x && p.y >= min_pt.y && p.y <= max_pt.y &&
            p.z >= min_pt.z && p.z <= max_pt.z);
}

static bool
is_point_in_tetrahedron_bounding_box_using_minmax_pt(const Point &p,
                                                     const Point &min_pt,
                                                     const Point &max_pt)
{
    // Check if the point is within the bounding box
    return (p.x >= min_pt.x && p.x <= max_pt.x && p.y >= min_pt.y && p.y <= max_pt.y &&
            p.z >= min_pt.z && p.z <= max_pt.z);
}

// Determines if a point is inside or on the boundary of a tetrahedron
// using barycentric coordinates.
// Returns:
//  0 if outside
//  1 if inside
//  2 if on boundary
// -1 if tetrahedron is degenerate
int
is_point_in_tetrahedron(const Point &p,
                        const Point &v0,
                        const Point &v1,
                        const Point &v2,
                        const Point &v3)
{

    // Check if the point is within the bounding box of the tetrahedron
    auto [min_pt, max_pt] = get_tetrahedron_bounding_box(v0, v1, v2, v3);

    if (!is_point_in_tetrahedron_bounding_box_using_minmax_pt(p, min_pt, max_pt)) {
        return 0;  // Outside
    }

    double d = calculate_bbox_diagonal(min_pt, max_pt);
    double epsilon = 1e-6 * d;

    // Calculate vectors from v0
    Point v0v1 = subtract(v1, v0);
    Point v0v2 = subtract(v2, v0);
    Point v0v3 = subtract(v3, v0);
    Point v0p = subtract(p, v0);

    // Matrix M = [v0v1 | v0v2 | v0v3]
    Matrix3x3 M = { { { v0v1.x, v0v2.x, v0v3.x },
                      { v0v1.y, v0v2.y, v0v3.y },
                      { v0v1.z, v0v2.z, v0v3.z } } };

    Matrix3x3 inv_M;
    if (!invert_matrix3x3(M, inv_M)) {
        return -1;  // Degenerate tetrahedron (zero volume)
    }

    // Solve M * [b1, b2, b3]^T = v0p for barycentric coordinates b1, b2, b3
    Point bary_coords_123 = multiply_matrix_vector(inv_M, v0p);
    double b1 = bary_coords_123.x;
    double b2 = bary_coords_123.y;
    double b3 = bary_coords_123.z;
    double b0 = 1.0 - b1 - b2 - b3;

    std::array<double, 4> b_coords = { b0, b1, b2, b3 };

    bool all_non_negative = true;
    bool any_zero = false;
    bool all_positive_and_sum_to_one = true;

    double sum_coords = 0;
    for (double coord : b_coords) {
        sum_coords += coord;
        if (coord < -epsilon) {  // Allow for small negative due to precision
            all_non_negative = false;
            all_positive_and_sum_to_one = false;
            break;
        }
        if (coord < epsilon) {  // Effectively zero or very small positive
            any_zero = true;
        }
        if (coord <= epsilon) {  // Not strictly positive
            all_positive_and_sum_to_one = false;
        }
    }
    if (std::abs(sum_coords - 1.0) > epsilon * 4) {  // Sum should be 1
        all_positive_and_sum_to_one = false;
        all_non_negative = false;  // If sum is not 1, it's problematic
    }

    if (all_non_negative) {  // All coords are >= 0 (within EPSILON)
        if (any_zero) {
            return 2;  // On boundary (at least one coord is approx zero)
        } else if (all_positive_and_sum_to_one) {  // All coords are > EPSILON and sum
                                                   // is 1
            return 1;                              // Strictly inside
        } else {  // All non-negative, sum is 1, but some might be very close to zero
                  // (handled by any_zero) This case implies it's on boundary or inside.
                  // If not strictly inside, it's on boundary.
            return 2;  // On boundary or very close
        }
    }

    return 0;  // Outside
}

/**
 * @brief This function mimics the former x_point_in_tetrahedron in the old C base
 */
bool
is_point_in_tetrahedron_legacy(const Point &p,
                               const Point &a,
                               const Point &b,
                               const Point &c,
                               const Point &d)
{
    // Check if the point is within the bounding box of the tetrahedron
    if (!is_point_in_tetrahedron_bounding_box(p, a, b, c, d)) {
        return false;  // Outside
    }

    // Calculate the absolute volume of the tetrahedron
    double true_vol = std::abs(signed_tetrahedron_volume(a, b, c, d));
    double const FLOATEPS = 1e-5;

    if (is_tetrahedron_degenerate_from_volume(true_vol, FLOATEPS)) {
        // Degenerate tetrahedron, assume point is not inside
        return false;
    }

    // Calculate the absolute volumes of the sub-tetrahedra
    double v1 = std::abs(signed_tetrahedron_volume(p, b, c, d));
    double v2 = std::abs(signed_tetrahedron_volume(a, p, c, d));
    double v3 = std::abs(signed_tetrahedron_volume(a, b, p, d));
    double v4 = std::abs(signed_tetrahedron_volume(a, b, c, p));

    double sumvol = v1 + v2 + v3 + v4;
    double rel_error = true_vol * 0.001;
    double diff = sumvol - true_vol;

    if (diff < -rel_error) {
        throw std::runtime_error(
          "Impossible... sumvol < true_vol. Check the input points.");
    }

    // Check if the point is inside the tetrahedron
    if (diff > rel_error) {
        return false;
    }
    return true;
}

}  // namespace xtgeo::geometry
