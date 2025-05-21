#include <cmath>   // Required for fabs
#include <limits>  // Required for numeric_limits
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using xyz::Point;

/**
 * @brief Mimics the former x_point_in_hexahedron_v1 in the old C base
 * Note that the old base delt with the hexahedron in a different way, and hence the
 * vertex number order is changed (but should mean the same).
 */
static int
is_point_in_hexahedron_legacy_internal(const Point &point,
                                       const std::array<Point, 8> &vertices)
{

    // Decompose the hexahedron into 5 tetrahedrons
    const std::array<std::array<int, 4>, 5> tetrahedrons1 = { {
      { 0, 2, 3, 7 },  // former cornercell based { 0, 2, 3, 4 } Tetrahedron 1
      { 0, 1, 2, 5 },  // former cornercell based { 0, 1, 3, 5 } Tetrahedron 2
      { 0, 4, 5, 7 },  // former cornercell based { 0, 4, 5, 6 } Tetrahedron 3
      { 2, 7, 6, 5 },  // former cornercell based { 3, 6, 7, 5 } Tetrahedron 4
      { 0, 2, 5, 7 }   // former cornercell based { 0, 3, 5, 6 } Tetrahedron 5
    } };

    bool inside1 = false;
    for (const auto &tetra : tetrahedrons1) {
        const Point &v0 = vertices[tetra[0]];
        const Point &v1 = vertices[tetra[1]];
        const Point &v2 = vertices[tetra[2]];
        const Point &v3 = vertices[tetra[3]];

        inside1 = is_point_in_tetrahedron_legacy(point, v0, v1, v2, v3);
        if (inside1) {
            break;
        }
    }
    // Decompose the hexahedron into 5 OTHER tetrahedrons
    const std::array<std::array<int, 4>, 5> tetrahedrons2 = { {
      { 0, 1, 3, 4 },  // former cornercell base { 0, 1, 2, 4 } Tetrahedron 1
      { 1, 3, 2, 6 },  // former cornercell base { 0, 2, 3, 7 } Tetrahedron 2
      { 4, 5, 6, 1 },  // former cornercell base { 4, 5, 7, 1 } Tetrahedron 3
      { 7, 4, 6, 3 },  // former cornercell base { 6, 4, 7, 2 } Tetrahedron 4
      { 1, 3, 4, 6 }   // former cornercell base { 1, 2, 4, 7 } Tetrahedron 5
    } };

    bool inside2 = false;
    for (const auto &tetra : tetrahedrons2) {
        const Point &v0 = vertices[tetra[0]];
        const Point &v1 = vertices[tetra[1]];
        const Point &v2 = vertices[tetra[2]];
        const Point &v3 = vertices[tetra[3]];

        inside2 = is_point_in_tetrahedron_legacy(point, v0, v1, v2, v3);
        if (inside2) {
            break;
        }
    }

    int sum = inside1 + inside2;
    return sum;  // 0 = outside, 1 = uncertain, 2 = inside
}

// =====================================================================================
// Main function for legacy tetrahedral decomposition (resembling method in old C base)
// =====================================================================================

/**
 * @brief A central function where one can select appropriate method
 * for point-in-cell test.
 * @param point The point to test, negated Z compared to python
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 * @throws std::invalid_argument if the method is not recognized
 */
int
is_point_in_hexahedron_tetrahedrons_legacy(const Point &point,
                                           const HexahedronCorners &hexahedron_corners)
{

    std::array<Point, 8> vertices = {
        hexahedron_corners.upper_sw, hexahedron_corners.upper_se,
        hexahedron_corners.upper_ne, hexahedron_corners.upper_nw,
        hexahedron_corners.lower_sw, hexahedron_corners.lower_se,
        hexahedron_corners.lower_ne, hexahedron_corners.lower_nw
    };

    return is_point_in_hexahedron_legacy_internal(point, vertices);  // no tolerance

}  // is_point_in_hexahedron

}  // namespace geometry
