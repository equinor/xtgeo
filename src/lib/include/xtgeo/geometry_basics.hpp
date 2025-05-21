#ifndef XTGEO_GEOMETRY_BASICS_HPP_
#define XTGEO_GEOMETRY_BASICS_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

/** Basic mathematic helper functions for geometry */

// =====================================================================================
// Point operations, namespace xtgeo::geometry::point
// =====================================================================================

namespace xtgeo::geometry::generic {

// Local coordinates for the 8 nodes of a standard hex element in [-1,1]^3
// v0: (-1,-1, 1), v1: (1,-1, 1), v2: (1,1,1), v3: (-1,1,1) (Top face, Z=1)
// v4: (-1,-1,-1), v5: (1,-1,-1), v6: (1,1,-1), v7: (-1,1,-1) (Bottom face, Z=-1)
constexpr double xi_coords[8] = { -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };
constexpr double eta_coords[8] = { -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0 };
constexpr double zeta_coords[8] = { 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0 };

// Shape function N_i(xi, eta, zeta)
inline double
shape_function(int i, double xi, double eta, double zeta)
{
    return 0.125 * (1.0 + xi * xi_coords[i]) * (1.0 + eta * eta_coords[i]) *
           (1.0 + zeta * zeta_coords[i]);
}

}  // namespace xtgeo::geometry::generic

namespace xtgeo::geometry::point {

using xyz::Point;

// Point operations
inline Point
subtract(Point a, Point b)
{
    return { a.x - b.x, a.y - b.y, a.z - b.z };
}

inline Point
add(Point a, Point b)
{
    return { a.x + b.x, a.y + b.y, a.z + b.z };
}

inline Point
scale(Point p, double s)
{
    return { p.x * s, p.y * s, p.z * s };
}

inline double
dot(Point a, Point b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double
dot_product(Point a, Point b)
{
    return dot(a, b);
}

inline Point
cross(Point a, Point b)
{
    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };
}

// alias dot to dot_product
inline Point
cross_product(Point a, Point b)
{
    return cross(a, b);
}

inline double
magnitude(Point p)
{
    return std::sqrt(dot(p, p));
}

// Helper function to calculate magnitude squared (avoid sqrt)
inline double
magnitude_squared(const xyz::Point &v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

// Maps local coordinates to global coordinates using shape functions
inline Point
map_local_to_global(const Point local_pt, const Point hex_vertices[8])
{
    Point global_pt = { 0, 0, 0 };
    for (int i = 0; i < 8; ++i) {
        double N_i = generic::shape_function(i, local_pt.x, local_pt.y, local_pt.z);
        global_pt.x += N_i * hex_vertices[i].x;
        global_pt.y += N_i * hex_vertices[i].y;
        global_pt.z += N_i * hex_vertices[i].z;
    }
    return global_pt;
}

/**
 * Calculate the normal vector of a plane defined by three points.
 */
inline Point
calculate_normal(const Point &a, const Point &b, const Point &c)
{
    // Calculate vectors
    Point ab = { b.x - a.x, b.y - a.y, b.z - a.z };
    Point ac = { c.x - a.x, c.y - a.y, c.z - a.z };

    // Compute the cross product
    return cross(ab, ac);
}

}  // namespace xtgeo::geometry::point

// =====================================================================================
// Matrix operations, namespace xtgeo::geometry::matrix
// =====================================================================================

namespace xtgeo::geometry::matrix {

constexpr double MATRIX_EPSILON = 1e-9;

using xyz::Point;

using Matrix3x3 = std::array<std::array<double, 3>, 3>;

// Invert a 3x3 matrix. Returns true if successful, false otherwise.
inline bool
invert_matrix3x3(const Matrix3x3 &matrix, Matrix3x3 &inv_matrix)
{
    // Calculate the determinant using cofactor expansion
    double det =
      matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
      matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
      matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);

    if (std::abs(det) < MATRIX_EPSILON) {
        return false;  // Matrix is singular
    }

    double inv_det = 1.0 / det;

    // Calculate the cofactor matrix and transpose it to get the adjugate
    inv_matrix[0][0] =
      (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) * inv_det;
    inv_matrix[0][1] =
      (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) * inv_det;
    inv_matrix[0][2] =
      (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) * inv_det;

    inv_matrix[1][0] =
      (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) * inv_det;
    inv_matrix[1][1] =
      (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) * inv_det;
    inv_matrix[1][2] =
      (matrix[1][0] * matrix[0][2] - matrix[0][0] * matrix[1][2]) * inv_det;

    inv_matrix[2][0] =
      (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]) * inv_det;
    inv_matrix[2][1] =
      (matrix[2][0] * matrix[0][1] - matrix[0][0] * matrix[2][1]) * inv_det;
    inv_matrix[2][2] =
      (matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]) * inv_det;

    return true;
}

inline Point
multiply_matrix_vector(const Matrix3x3 &matrix, Point vec)
{
    return { matrix[0][0] * vec.x + matrix[0][1] * vec.y + matrix[0][2] * vec.z,
             matrix[1][0] * vec.x + matrix[1][1] * vec.y + matrix[1][2] * vec.z,
             matrix[2][0] * vec.x + matrix[2][1] * vec.y + matrix[2][2] * vec.z };
}

// --- Isoparametric Mapping Constants and Functions ---

// Derivatives of shape functions
inline double
dn_dxi(int i, double xi, double eta, double zeta)
{
    return 0.125 * generic::xi_coords[i] * (1.0 + eta * generic::eta_coords[i]) *
           (1.0 + zeta * generic::zeta_coords[i]);
}

inline double
dn_deta(int i, double xi, double eta, double zeta)
{
    return 0.125 * generic::eta_coords[i] * (1.0 + xi * generic::xi_coords[i]) *
           (1.0 + zeta * generic::zeta_coords[i]);
}

inline double
dn_dzeta(int i, double xi, double eta, double zeta)
{
    return 0.125 * generic::zeta_coords[i] * (1.0 + xi * generic::xi_coords[i]) *
           (1.0 + eta * generic::eta_coords[i]);
}
// Calculates the Jacobian matrix at given local coordinates
inline Matrix3x3
calculate_jacobian(const Point local_pt, const Point hex_vertices[8])
{
    Matrix3x3 J = { { { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 } } };

    double xi = local_pt.x;
    double eta = local_pt.y;
    double zeta = local_pt.z;

    for (int i = 0; i < 8; ++i) {
        double dNi_dxi = dn_dxi(i, xi, eta, zeta);
        double dNi_deta = dn_deta(i, xi, eta, zeta);
        double dNi_dzeta = dn_dzeta(i, xi, eta, zeta);

        J[0][0] += dNi_dxi * hex_vertices[i].x;
        J[0][1] += dNi_deta * hex_vertices[i].x;
        J[0][2] += dNi_dzeta * hex_vertices[i].x;

        J[1][0] += dNi_dxi * hex_vertices[i].y;
        J[1][1] += dNi_deta * hex_vertices[i].y;
        J[1][2] += dNi_dzeta * hex_vertices[i].y;

        J[2][0] += dNi_dxi * hex_vertices[i].z;
        J[2][1] += dNi_deta * hex_vertices[i].z;
        J[2][2] += dNi_dzeta * hex_vertices[i].z;
    }
    return J;
}

}  // namespace xtgeo::geometry::matrix

#endif  // XTGEO_GEOMETRY_BASICS_HPP_
