#include <cmath>   // Required for fabs
#include <limits>  // Required for numeric_limits
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

/**
 * Computing if a point is inside a hexahedron is a non-trivial task, and
 * several methods exist. Some methods are implemented in this file, that or not based
 * on tetrahedrons or isoparametric methods. The methods are:
 * 1. Using planes
 * 2. Using ray casting
 */

namespace xtgeo::geometry {

using xyz::Point;

// =====================================================================================
// Ray casting calculations for a point inside a hexahedron
// =====================================================================================
/**
 * @brief Function to determine if a ray intersects a triangle in 3D space.
 * Uses Möller–Trumbore intersection algorithm. This implementation is robust against
 * both left-handed and right-handed coordinate systems.
 *
 * @param origin Origin point of the ray
 * @param direction Direction of the ray (doesn't need to be normalized)
 * @param v0 First vertex of the triangle
 * @param v1 Second vertex of the triangle
 * @param v2 Third vertex of the triangle
 * @return bool True if the ray intersects the triangle, false otherwise
 */
static bool
ray_intersects_triangle(const Point &origin,
                        const Point &direction,
                        const Point &v0,
                        const Point &v1,
                        const Point &v2)
{
    const double epsilon = 1e-5;

    // Calculate edge vectors
    Point edge1 = { v1.x() - v0.x(), v1.y() - v0.y(), v1.z() - v0.z() };
    Point edge2 = { v2.x() - v0.x(), v2.y() - v0.y(), v2.z() - v0.z() };

    // Calculate the cross product (pvec = direction × edge2)
    Point pvec = { direction.y() * edge2.z() - direction.z() * edge2.y(),
                   direction.z() * edge2.x() - direction.x() * edge2.z(),
                   direction.x() * edge2.y() - direction.y() * edge2.x() };

    // Calculate determinant (dot product of edge1 and pvec)
    double det = edge1.x() * pvec.x() + edge1.y() * pvec.y() + edge1.z() * pvec.z();

    // If determinant is near zero, ray lies in plane of triangle or ray is parallel to
    // plane Use absolute value to handle both left and right-handed coordinate systems
    if (fabs(det) < epsilon)
        return false;

    double inv_det = 1.0 / det;

    // Calculate vector from v0 to ray origin
    Point tvec = { origin.x() - v0.x(), origin.y() - v0.y(), origin.z() - v0.z() };

    // Calculate u parameter and test bounds
    double u =
      (tvec.x() * pvec.x() + tvec.y() * pvec.y() + tvec.z() * pvec.z()) * inv_det;
    if (u < 0.0 || u > 1.0)
        return false;

    // Calculate qvec (qvec = tvec × edge1)
    Point qvec = { tvec.y() * edge1.z() - tvec.z() * edge1.y(),
                   tvec.z() * edge1.x() - tvec.x() * edge1.z(),
                   tvec.x() * edge1.y() - tvec.y() * edge1.x() };

    // Calculate v parameter and test bounds
    double v =
      (direction.x() * qvec.x() + direction.y() * qvec.y() + direction.z() * qvec.z()) *
      inv_det;
    if (v < 0.0 || u + v > 1.0)
        return false;

    // Calculate t parameter - ray intersects triangle
    double t =
      (edge2.x() * qvec.x() + edge2.y() * qvec.y() + edge2.z() * qvec.z()) * inv_det;

    // Only consider intersections in front of the ray
    return (t > epsilon);
}  // ray_intersects_triangle

/**
 * @brief Using ray casting method to determine if a point is inside a hexahedron.
 */
static bool
is_point_in_hexahedron_using_raycasting(const Point &point_rh,
                                        const std::array<Point, 8> &vertices)
{

    // Define the 6 faces of the hexahedron (each face is 4 corners)
    // Face indices: 0=top, 1=bottom, 2=front, 3=right, 4=back, 5=left
    std::array<std::array<int, 4>, 6> faces = { {
      { 0, 1, 2, 3 },  // top face (upper)
      { 4, 5, 6, 7 },  // bottom face (lower)
      { 0, 1, 5, 4 },  // front face
      { 1, 2, 6, 5 },  // right face
      { 2, 3, 7, 6 },  // back face
      { 3, 0, 4, 7 }   // left face
    } };

    // Ray casting: count intersections along a ray in +X direction
    int intersections = 0;

    // Check each face for intersection with the ray from point to +X infinity
    for (const auto &face : faces) {
        // Get the 4 corners of this face
        Point a = vertices[face[0]];
        Point b = vertices[face[1]];
        Point c = vertices[face[2]];
        Point d = vertices[face[3]];

        // For a ray in +X direction, we need AT LEAST ONE corner has X > point.x
        // the right) We don't need points on the left for the face to possibly
        // intersect the ray
        bool has_point_right = (a.x() > point_rh.x() || b.x() > point_rh.x() ||
                                c.x() > point_rh.x() || d.x() > point_rh.x());

        // Check if face's YZ range contains point's YZ coordinates
        bool y_in_range = (std::min({ a.y(), b.y(), c.y(), d.y() }) <= point_rh.y()) &&
                          (std::max({ a.y(), b.y(), c.y(), d.y() }) >= point_rh.y());
        bool z_in_range = (std::min({ a.z(), b.z(), c.z(), d.z() }) <= point_rh.z()) &&
                          (std::max({ a.z(), b.z(), c.z(), d.z() }) >= point_rh.z());

        // To potentially intersect with ray, we need:
        // 1. At least one point to the right of the ray origin
        // 2. The point's YZ position is within the face's YZ extents
        bool may_intersect = has_point_right && y_in_range && z_in_range;

        // Special case: if face is in a plane parallel to ray (all X values equal)
        if (a.x() == b.x() && b.x() == c.x() && c.x() == d.x() &&
            a.x() > point_rh.x()) {
            // If all points are at the same X coordinate AND that X is greater than
            // the point's X, we may have an intersection
            may_intersect = y_in_range && z_in_range;
        }

        // If the face may intersect, do detailed intersection test
        if (may_intersect) {
            // For planar quadrilateral faces, we should count only one
            // intersection
            // regardless of how we split it into triangles
            bool intersected = false;

            // Test if either triangle is intersected
            if (geometry::ray_intersects_triangle(point_rh, { 1, 0, 0 }, a, b, c) ||
                geometry::ray_intersects_triangle(point_rh, { 1, 0, 0 }, a, c, d)) {
                intersections++;
                intersected = true;
            }
        }
    }
    // If the number of intersections is odd, the point is inside
    return (intersections % 2) == 1;
}

// =====================================================================================
// Public function for Raycasting method
// =====================================================================================

/**
 * @brief Raycasting method to determine if a point is inside a hexahedron.
 * This method uses ray casting to check if the point is inside the hexahedron.
 * @param point The point to test, negated Z compared to python
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 * @throws std::invalid_argument if the method is not recognized
 */
bool
is_point_in_hexahedron_raycasting(const Point &point,
                                  const HexahedronCorners &hexahedron_corners)
{
    std::array<Point, 8> vertices = {
        hexahedron_corners.upper_sw, hexahedron_corners.upper_se,
        hexahedron_corners.upper_ne, hexahedron_corners.upper_nw,
        hexahedron_corners.lower_sw, hexahedron_corners.lower_se,
        hexahedron_corners.lower_ne, hexahedron_corners.lower_nw
    };

    // tolerance scaler is currently not in use
    return is_point_in_hexahedron_using_raycasting(point, vertices);

}  // is_point_in_hexahedron_raycasting

}  // namespace geometry
