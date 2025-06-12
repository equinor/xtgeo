#ifndef XTGEO_GEOMETRY_BASICS_HPP_
#define XTGEO_GEOMETRY_BASICS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
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

inline double
lerp(double x1, double x2, double t)
{
    return x1 + (x2 - x1) * t;
}

}  // namespace xtgeo::geometry::generic

namespace xtgeo::geometry {

// Use Eigen types
using Vector3d = Eigen::Vector3d;
using Matrix3d = Eigen::Matrix3d;

namespace point {

using xyz::Point;

// Point operations using the new Point class with embedded Eigen operations
// These now simply delegate to the Point's built-in operators

inline Point
subtract(Point a, Point b)
{
    return a - b;  // Using Point's operator-
}

inline Point
add(Point a, Point b)
{
    return a + b;  // Using Point's operator+
}

inline Point
scale(Point p, double s)
{
    return p * s;  // Using Point's operator*
}

inline double
dot(Point a, Point b)
{
    return a.dot(b);  // Using Point's dot method
}

inline double
dot_product(Point a, Point b)
{
    return dot(a, b);
}

inline Point
cross(Point a, Point b)
{
    return a.cross(b);  // Using Point's cross method
}

// alias to cross
inline Point
cross_product(Point a, Point b)
{
    return cross(a, b);
}

inline double
magnitude(Point p)
{
    return p.norm();  // Using Point's norm method
}

// Helper function to calculate magnitude squared (avoid sqrt)
inline double
magnitude_squared(const xyz::Point &v)
{
    return v.squared_norm();  // Using Point's --> Eigen's squaredNorm method
}

// Maps local coordinates to global coordinates using shape functions
// This is specific to your application and doesn't need Eigen directly
inline Point
map_local_to_global(const Point local_pt, const Point hex_vertices[8])
{
    Point global_pt = { 0, 0, 0 };
    for (int i = 0; i < 8; ++i) {
        double N_i =
          generic::shape_function(i, local_pt.x(), local_pt.y(), local_pt.z());
        global_pt = global_pt + (hex_vertices[i] * N_i);  // Use built-in operators
    }
    return global_pt;
}

/**
 * Calculate the normal vector of a plane defined by three points.
 */
inline Point
calculate_normal(const Point &a, const Point &b, const Point &c)
{
    // No need for conversions anymore, just use Point's operators
    return (b - a).cross(c - a);
}

}  // namespace xtgeo::geometry::point

// =====================================================================================
// Matrix operations, namespace xtgeo::geometry::matrix
// =====================================================================================

namespace matrix {

constexpr double MATRIX_EPSILON = 1e-9;

using xyz::Point;

using Matrix3x3 = std::array<std::array<double, 3>, 3>;

// Convert between Matrix3x3 and Eigen::Matrix3d
inline Matrix3d
to_eigen(const Matrix3x3 &mat)
{
    Matrix3d result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result(i, j) = mat[i][j];
        }
    }
    return result;
}

inline Matrix3x3
from_eigen(const Matrix3d &mat)
{
    Matrix3x3 result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = mat(i, j);
        }
    }
    return result;
}

// Invert a 3x3 matrix using Eigen. Returns true if successful.
inline bool
invert_matrix3x3(const Matrix3x3 &matrix, Matrix3x3 &inv_matrix)
{
    // Convert to Eigen
    Matrix3d eigenMatrix = to_eigen(matrix);

    // Check if matrix is invertible
    double det = eigenMatrix.determinant();
    if (std::abs(det) < MATRIX_EPSILON) {
        return false;  // Matrix is singular
    }

    // Compute inverse
    Matrix3d eigenInverse = eigenMatrix.inverse();

    // Convert back to Matrix3x3
    inv_matrix = from_eigen(eigenInverse);

    return true;
}

inline Point
multiply_matrix_vector(const Matrix3x3 &matrix, Point vec)
{
    // Convert to Eigen
    Matrix3d eigenMatrix = to_eigen(matrix);

    // Get the internal Eigen vector directly
    const Eigen::Vector3d &eigenVec = vec.data();

    // Multiply
    Eigen::Vector3d result = eigenMatrix * eigenVec;

    // Return as Point - use the constructor that takes Eigen::Vector3d
    return Point(result);
}

// --- Isoparametric Mapping Constants and Functions ---

// Derivatives of shape functions (not changing these)
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
    // Initialize Eigen matrix
    Matrix3d J = Matrix3d::Zero();

    double xi = local_pt.x();
    double eta = local_pt.y();
    double zeta = local_pt.z();

    for (int i = 0; i < 8; ++i) {
        double dNi_dxi = dn_dxi(i, xi, eta, zeta);
        double dNi_deta = dn_deta(i, xi, eta, zeta);
        double dNi_dzeta = dn_dzeta(i, xi, eta, zeta);

        // Access vertex data directly
        const Eigen::Vector3d &vertex = hex_vertices[i].data();

        J(0, 0) += dNi_dxi * vertex.x();
        J(0, 1) += dNi_deta * vertex.x();
        J(0, 2) += dNi_dzeta * vertex.x();

        J(1, 0) += dNi_dxi * vertex.y();
        J(1, 1) += dNi_deta * vertex.y();
        J(1, 2) += dNi_dzeta * vertex.y();

        J(2, 0) += dNi_dxi * vertex.z();
        J(2, 1) += dNi_deta * vertex.z();
        J(2, 2) += dNi_dzeta * vertex.z();
    }

    // Convert back to Matrix3x3
    return from_eigen(J);
}

}  // namespace matrix
}  // namespace xtgeo::geometry

#endif  // XTGEO_GEOMETRY_BASICS_HPP_
