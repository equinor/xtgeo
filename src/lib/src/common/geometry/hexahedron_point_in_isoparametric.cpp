#include <algorithm>  // For std::all_of (in tetrahedral decomposition example)
#include <array>
#include <cmath>
#include <limits>     // Required for numeric_limits
#include <numeric>    // For std::iota (in tetrahedral decomposition example)
#include <stdexcept>  // For runtime_error
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using xyz::Point;

using geometry::matrix::calculate_jacobian;
using geometry::matrix::invert_matrix3x3;
using geometry::matrix::Matrix3x3;
using geometry::matrix::multiply_matrix_vector;
using geometry::point::add;
using geometry::point::cross_product;
using geometry::point::dot_product;
using geometry::point::magnitude;
using geometry::point::magnitude_squared;
using geometry::point::map_local_to_global;
using geometry::point::subtract;

constexpr double EPSILON = 1e-9;

static int
is_point_in_hexahedron_isoparametric_internal(const Point &p_test,
                                              const Point hex_vertices[8],
                                              int max_iterations = 20,
                                              double tolerance = 1e-7)
{
    Point local_coords = {
        0, 0, 0
    };  // Initial guess for local coordinates (center of parent element)

    for (int iter = 0; iter < max_iterations; ++iter) {
        Point current_global_coords = map_local_to_global(local_coords, hex_vertices);
        Point residual = subtract(p_test, current_global_coords);

        if (magnitude(residual) < tolerance) {  // Converged
            break;
        }

        Matrix3x3 J = calculate_jacobian(local_coords, hex_vertices);
        Matrix3x3 inv_J;
        if (!invert_matrix3x3(J, inv_J)) {
            // Jacobian is singular, cannot proceed reliably
            // This can happen for degenerate or very distorted cells
            return -1;
        }

        Point delta_local = multiply_matrix_vector(inv_J, residual);
        local_coords = add(local_coords, delta_local);

        // Optional: Clamp local_coords to [-1,1] range if it goes too far,
        // though for well-behaved problems it should converge without this.
        // local_coords.x() = std::max(-1.0, std::min(1.0, local_coords.x()));
        // local_coords.y() = std::max(-1.0, std::min(1.0, local_coords.y()));
        // local_coords.z() = std::max(-1.0, std::min(1.0, local_coords.z()));
    }

    // Check if the converged local coordinates are within the parent element [-1, 1]^3
    bool on_xi_boundary = std::abs(std::abs(local_coords.x()) - 1.0) < EPSILON;
    bool on_eta_boundary = std::abs(std::abs(local_coords.y()) - 1.0) < EPSILON;
    bool on_zeta_boundary = std::abs(std::abs(local_coords.z()) - 1.0) < EPSILON;

    bool inside_xi =
      local_coords.x() >= -1.0 - EPSILON && local_coords.x() <= 1.0 + EPSILON;
    bool inside_eta =
      local_coords.y() >= -1.0 - EPSILON && local_coords.y() <= 1.0 + EPSILON;
    bool inside_zeta =
      local_coords.z() >= -1.0 - EPSILON && local_coords.z() <= 1.0 + EPSILON;

    if (inside_xi && inside_eta && inside_zeta) {
        if (on_xi_boundary || on_eta_boundary || on_zeta_boundary) {
            // Further check: ensure the point re-mapped from boundary local coords is
            // indeed close to p_test
            Point remapped_p = map_local_to_global(local_coords, hex_vertices);
            if (magnitude(subtract(p_test, remapped_p)) <
                tolerance * 10) {  // Use a slightly larger tolerance for boundary check
                return 2;          // On boundary
            } else {
                // This can happen if the iteration stopped near the boundary but the
                // point is actually outside, or if the element is very distorted. A
                // more robust check might be needed here, or classify as outside. For
                // simplicity, if it's within local bounds but remapping is off,
                // consider it outside.
                if (local_coords.x() < -1.0 + EPSILON ||
                    local_coords.x() > 1.0 - EPSILON ||
                    local_coords.y() < -1.0 + EPSILON ||
                    local_coords.y() > 1.0 - EPSILON ||
                    local_coords.z() < -1.0 + EPSILON ||
                    local_coords.z() > 1.0 - EPSILON) {
                    // It was only within bounds due to EPSILON, but not strictly
                    // inside.
                } else {
                    return 1;  // Inside (if not on boundary and remapping was close
                               // enough initially)
                }
            }
        }
        // Check if strictly inside (not on boundary by EPSILON)
        if (local_coords.x() > -1.0 + EPSILON && local_coords.x() < 1.0 - EPSILON &&
            local_coords.y() > -1.0 + EPSILON && local_coords.y() < 1.0 - EPSILON &&
            local_coords.z() > -1.0 + EPSILON && local_coords.z() < 1.0 - EPSILON) {
            return 1;  // Inside
        }
        // If it reached here, it was within [-1,1] due to EPSILON, but not strictly
        // inside, and the boundary check above might have classified it. If not
        // classified as boundary, and not strictly inside, it's likely on boundary or a
        // precision issue. A conservative approach might be to re-check if it's truly
        // on boundary or outside. For this example, if it's within [-1,1]+EPSILON and
        // not strictly inside, and not on boundary, it's likely on boundary.
        return 2;  // On boundary (or very close)
    }

    return 0;  // Outside
}

/**
 * @brief Determines if a point is inside a hexahedron using isoparametric mapping.
 * This method uses the isoparametric mapping technique to check if the point is
 * inside the hexahedron. It is efficient and works well for convex hexahedra, but
 * may be less reliable for points near edges or faces.
 *
 * @param point The point to check.
 * @param hex_corners The corners of the hexahedron.
 * @return 0 if outside, 1 if inside, 2 if on boundary, -1 if degenerate somehow
 */

int
is_point_in_hexahedron_isoparametric(const xyz::Point &point,
                                     const HexahedronCorners &hex_corners)
{

    // Create array of Point from the HexahedronCorners structure
    // following the expected vertex order for the isoparametric mapping
    Point hex_vertices[8] = {
        hex_corners.upper_sw,  // v0: upper southwest (-1,-1,1)
        hex_corners.upper_se,  // v1: upper southeast (1,-1,1)
        hex_corners.upper_ne,  // v2: upper northeast (1,1,1)
        hex_corners.upper_nw,  // v3: upper northwest (-1,1,1)
        hex_corners.lower_sw,  // v4: lower southwest (-1,-1,-1)
        hex_corners.lower_se,  // v5: lower southeast (1,-1,-1)
        hex_corners.lower_ne,  // v6: lower northeast (1,1,-1)
        hex_corners.lower_nw   // v7: lower northwest (-1,1,-1)
    };

    double tol = 1e-7;

    int result =
      is_point_in_hexahedron_isoparametric_internal(point, hex_vertices, 20, tol);

    return result;
}  // is_point_in_hexahedron_isoparametric

}  // namespace xtgeo::geometry
