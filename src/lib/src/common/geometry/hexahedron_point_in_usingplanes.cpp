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

using geometry::point::cross_product;
using geometry::point::dot_product;
using geometry::point::magnitude_squared;
using geometry::point::subtract;

// =====================================================================================
// Using planes calculations for a point inside a hexahedron
// =====================================================================================
/**
 * @brief Determines if a point is inside or on the boundary of a hexahedron using plane
 * side tests. This method triangulates each face and checks if the point lies on the
 * inner side of all 12 resulting triangles. It uses the centroid to consistently orient
 * face normals outwards. Handles potential numerical precision issues and degenerate
 * triangles.
 *
 * @param point The point to test.
 * @param vertices The 8 vertices of the hexahedron in standard order:
 *        0-3: upper face (sw, se, ne, nw)
 *        4-7: lower face (sw, se, ne, nw)
 * @return true if the point is inside or on the boundary, false otherwise.
 */
static bool
is_point_in_hexahedron_usingplanes_internal(const Point &point,
                                            const std::array<Point, 8> &vertices,
                                            const Point &min_pt,
                                            const Point &max_pt)
{
    double diagonal =
      std::sqrt(std::pow(max_pt.x - min_pt.x, 2) + std::pow(max_pt.y - min_pt.y, 2) +
                std::pow(max_pt.z - min_pt.z, 2));
    // If diagonal is zero (degenerate hexahedron), handle appropriately
    if (diagonal < std::numeric_limits<double>::epsilon()) {
        // All vertices are coincident. Point is inside only if it matches the vertex.
        return magnitude_squared(subtract(point, vertices[0])) <
               std::numeric_limits<double>::epsilon();
    }
    const double epsilon = 1e-6 * diagonal;  // Relative tolerance

    // Define the 6 faces of the hexahedron by vertex indices
    // Ensure consistent winding order for outward normals (e.g., counter-clockwise when
    // viewed from outside)
    //      3----2
    //     /|   /|
    //    0----1 |   Upper face (z+)
    //    | 7--|-6
    //    |/   |/
    //    4----5     Lower face (z-)
    //
    const std::array<std::array<int, 4>, 6> faces = { {
      { 0, 1, 2, 3 },  // Top (viewed from +Z)
      { 4, 7, 6, 5 },  // Bottom (viewed from -Z)
      { 0, 4, 5, 1 },  // Front (viewed from +Y)
      { 1, 5, 6, 2 },  // Right (viewed from +X)
      { 2, 6, 7, 3 },  // Back (viewed from -Y)
      { 3, 7, 4, 0 }   // Left (viewed from -X)
    } };

    // Calculate the centroid of the hexahedron
    Point centroid = { 0.0, 0.0, 0.0 };
    for (const auto &v : vertices) {
        centroid.x += v.x;
        centroid.y += v.y;
        centroid.z += v.z;
    }
    centroid.x /= 8.0;
    centroid.y /= 8.0;
    centroid.z /= 8.0;

    // Check the point against the planes defined by the triangles of each face
    for (const auto &face_indices : faces) {
        // Split the quadrilateral face into two triangles (using the first vertex)
        const std::array<std::array<int, 3>, 2> triangles = {
            { { face_indices[0], face_indices[1], face_indices[2] },
              { face_indices[0], face_indices[2], face_indices[3] } }
        };

        for (const auto &triangle_indices : triangles) {
            const Point &p0 = vertices[triangle_indices[0]];
            const Point &p1 = vertices[triangle_indices[1]];
            const Point &p2 = vertices[triangle_indices[2]];

            // Calculate the triangle normal
            Point edge1 = subtract(p1, p0);
            Point edge2 = subtract(p2, p0);
            Point normal = cross_product(edge1, edge2);

            // Check for degenerate triangles (zero area). Normal magnitude squared is
            // proportional to area squared. Use a tolerance slightly larger than
            // machine epsilon for squared values.
            if (magnitude_squared(normal) < epsilon * epsilon * 1e-6) {
                // Skip degenerate triangles. If a point lies exactly on a degenerate
                // face, other non-degenerate faces should still correctly classify it.
                continue;
            }

            // Orient the normal outwards using the centroid.
            // Check if the centroid is on the same side as the normal direction
            // relative to the plane origin p0.
            Point p0_to_centroid = subtract(centroid, p0);
            if (dot_product(normal, p0_to_centroid) > 0) {
                // Normal is pointing inwards relative to the centroid, flip it
                normal.x = -normal.x;
                normal.y = -normal.y;
                normal.z = -normal.z;
            }

            // Check if the test point is on the outer side of the plane defined by the
            // triangle. Calculate signed distance: dot(normal, point - p0)
            Point p0_to_point = subtract(point, p0);
            double dist = dot_product(normal, p0_to_point);

            // If point is clearly outside (positive distance beyond tolerance, given
            // outward normal)
            if (dist > epsilon) {
                return false;  // Point is outside this face plane
            }
        }
    }

    // If the point was not outside any face plane (i.e., dist <= EPSILON for all), it's
    // inside or on the boundary.
    return true;
}

// =====================================================================================
// Entry function for using planes method
// =====================================================================================

/**
 * @brief A central function where one can select appropriate method
 * for point-in-cell test.
 * @param point The point to test, negated Z compared to python
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 * @throws std::invalid_argument if the method is not recognized
 */
bool
is_point_in_hexahedron_usingplanes(const Point &point,
                                   const HexahedronCorners &hexahedron_corners)
{
    auto [min_pt, max_pt] = get_hexahedron_bounding_box(hexahedron_corners);

    std::array<Point, 8> vertices = {
        hexahedron_corners.upper_sw, hexahedron_corners.upper_se,
        hexahedron_corners.upper_ne, hexahedron_corners.upper_nw,
        hexahedron_corners.lower_sw, hexahedron_corners.lower_se,
        hexahedron_corners.lower_ne, hexahedron_corners.lower_nw
    };

    // tolerance scaler is currently not in use; set to 1.0
    return is_point_in_hexahedron_usingplanes_internal(point, vertices, min_pt, max_pt);

}  // is_point_in_hexahedron_usingplanes

}  // namespace geometry
