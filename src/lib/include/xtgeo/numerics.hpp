#ifndef XTGEO_NUMERICS_HPP_
#define XTGEO_NUMERICS_HPP_
#include <pybind11/pybind11.h>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::numerics {

inline xyz::Point
lerp3d(double x1, double y1, double z1, double x2, double y2, double z2, double t)
{
    return xyz::Point{ x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1) };
}

inline void
init(py::module &m)
{
    auto m_numerics = m.def_submodule("numerics", "Internal functions for numerics.");

    m_numerics.attr("UNDEF_DOUBLE") = numerics::UNDEF_DOUBLE;
    m_numerics.attr("EPSILON") = numerics::EPSILON;
    m_numerics.attr("TOLERANCE") = numerics::TOLERANCE;
}

/**
 * @brief Performs cubic Catmull-Rom interpolation.
 *
 * This function interpolates a value between four control points (p0, p1, p2, p3)
 * at a given fractional distance 't' from p1 towards p2.
 *
 * @param p0 The control point before the interpolation segment.
 * @param p1 The start point of the interpolation segment.
 * @param p2 The end point of the interpolation segment.
 * @param p3 The control point after the interpolation segment.
 * @param t The fractional distance [0, 1] between p1 and p2.
 * @return The interpolated value.
 */
inline double
catmull_rom_interpolate(double p0, double p1, double p2, double p3, double t)
{
    double t2 = t * t;
    double t3 = t2 * t;
    return 0.5 *
           ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
            (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

/**
 * @brief Performs linear interpolation between points in a vector.
 *
 * For each segment between two consecutive points, this function generates
 * 'ndiv' evenly spaced points (including the start point, excluding the end point).
 * The final point of the entire set is added at the end.
 *
 * @tparam T A container type that supports `[]` and `.size()`, like `std::vector`
 *           or `pybind11::detail::unchecked_reference`.
 * @param points The input data points.
 * @param ndiv The number of divisions to create within each segment.
 * @return A vector of interpolated values.
 */
template<typename T>
inline std::vector<double>
linear_interpolate(const T &points, int ndiv)
{
    size_t n_points = points.size();
    if (ndiv <= 1 || n_points < 2) {
        std::vector<double> result;
        result.reserve(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            result.push_back(points[i]);
        }
        return result;
    }

    size_t num_new_points = (n_points - 1) * ndiv + 1;
    std::vector<double> result;
    result.reserve(num_new_points);

    for (size_t i = 0; i < n_points - 1; ++i) {
        for (int l = 0; l < ndiv; ++l) {
            double fraction = static_cast<double>(l) / ndiv;
            result.push_back(points[i] + fraction * (points[i + 1] - points[i]));
        }
    }
    result.push_back(points[n_points - 1]);

    return result;
}

/**
 * @brief Performs Catmull-Rom spline interpolation on a vector of points.
 *
 * @tparam T A container type that supports `[]` and `.size()`, like `std::vector`.
 * @param points The input data points.
 * @param ndiv The number of divisions to create within each segment.
 * @return A vector of interpolated values.
 */
template<typename T>
inline std::vector<double>
catmull_rom_spline(const T &points, int ndiv)
{
    size_t n_points = points.size();
    if (ndiv <= 1 || n_points < 2) {
        std::vector<double> result;
        result.reserve(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            result.push_back(points[i]);
        }
        return result;
    }

    size_t num_new_points = (n_points - 1) * ndiv + 1;
    std::vector<double> result;
    result.reserve(num_new_points);

    for (size_t i = 0; i < n_points - 1; ++i) {
        // We need 4 points for Catmull-Rom: p0, p1, p2, p3
        // The segment is between p1 and p2.
        double p1 = points[i];
        double p2 = points[i + 1];

        // Handle boundaries by clamping
        double p0 = (i > 0) ? points[i - 1] : p1;
        double p3 = (i < n_points - 2) ? points[i + 2] : p2;

        for (int l = 0; l < ndiv; ++l) {
            double fraction = static_cast<double>(l) / ndiv;
            result.push_back(catmull_rom_interpolate(p0, p1, p2, p3, fraction));
        }
    }
    result.push_back(points[n_points - 1]);

    return result;
}

/**
 * @brief Gets a value from a vector with boundary handling.
 *
 * Handles out-of-bounds indices by clamping to the valid range [0, size-1].
 * This is useful for fetching control points at the edges of the data.
 *
 * @param data The vector of data points.
 * @param index The index to access.
 * @return The value at the clamped index.
 */
inline double
get_value_clamped(const std::vector<double> &data, int index)
{
    if (index < 0) {
        return data.front();
    }
    if (index >= static_cast<int>(data.size())) {
        return data.back();
    }
    return data[index];
}

/**
 * @brief A pre-computing solver for cubic spline interpolation.
 *
 * This class computes and stores the LU decomposition of the spline matrix,
 * allowing for rapid solving for multiple right-hand side vectors (i.e.,
 * multiple data traces).
 */
class CubicSplineSolver
{
public:
    /**
     * @brief Constructs and computes the solver for a given set of x-points.
     * @param x_pts The x-coordinates of the data points.
     */
    explicit CubicSplineSolver(const std::vector<double> &x_pts) : n_(x_pts.size())
    {
        if (n_ < 3) {
            return;
        }

        std::vector<double> h(n_ - 1);
        for (size_t i = 0; i < n_ - 1; ++i) {
            h[i] = x_pts[i + 1] - x_pts[i];
        }

        Eigen::SparseMatrix<double> A(n_, n_);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(3 * n_);

        for (size_t i = 1; i < n_ - 1; ++i) {
            triplets.emplace_back(i, i - 1, h[i - 1]);
            triplets.emplace_back(i, i, 2 * (h[i - 1] + h[i]));
            triplets.emplace_back(i, i + 1, h[i]);
        }

        triplets.emplace_back(0, 0, h[1]);
        triplets.emplace_back(0, 1, -(h[0] + h[1]));
        triplets.emplace_back(0, 2, h[0]);
        triplets.emplace_back(n_ - 1, n_ - 3, h[n_ - 2]);
        triplets.emplace_back(n_ - 1, n_ - 2, -(h[n_ - 3] + h[n_ - 2]));
        triplets.emplace_back(n_ - 1, n_ - 1, h[n_ - 3]);

        A.setFromTriplets(triplets.begin(), triplets.end());
        solver_.compute(A);
    }

    /**
     * @brief Interpolates y-values using the pre-computed solver.
     */
    std::vector<double> interpolate(const std::vector<double> &x_pts,
                                    const std::vector<double> &y_pts,
                                    const std::vector<double> &new_x_pts) const
    {
        if (n_ < 2) {
            return std::vector<double>(new_x_pts.size(), n_ > 0 ? y_pts[0] : 0.0);
        }
        if (n_ < 3) {
            std::vector<double> result;
            result.reserve(new_x_pts.size());
            for (const auto &x : new_x_pts) {
                double t = (x - x_pts[0]) / (x_pts[1] - x_pts[0]);
                result.push_back(y_pts[0] + t * (y_pts[1] - y_pts[0]));
            }
            return result;
        }

        std::vector<double> h(n_ - 1);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_);
        for (size_t i = 0; i < n_ - 1; ++i) {
            h[i] = x_pts[i + 1] - x_pts[i];
            if (i > 0) {
                b(i) = 6 * ((y_pts[i + 1] - y_pts[i]) / h[i] -
                            (y_pts[i] - y_pts[i - 1]) / h[i - 1]);
            }
        }

        if (solver_.info() != Eigen::Success) {
            return {};
        }
        Eigen::VectorXd M = solver_.solve(b);

        std::vector<double> y_new;
        y_new.reserve(new_x_pts.size());

        for (const auto &x : new_x_pts) {
            auto it = std::upper_bound(x_pts.begin(), x_pts.end(), x);
            size_t seg = std::distance(x_pts.begin(), it) - 1;
            seg = std::max((size_t)0, std::min(seg, n_ - 2));

            double dx1 = x - x_pts[seg];
            double dx2 = x_pts[seg + 1] - x;
            double h_seg = h[seg];

            double val =
              (dx2 / h_seg) * y_pts[seg] + (dx1 / h_seg) * y_pts[seg + 1] +
              ((dx2 * dx2 * dx2 / (6 * h_seg)) - (dx2 * h_seg / 6)) * M(seg) +
              ((dx1 * dx1 * dx1 / (6 * h_seg)) - (dx1 * h_seg / 6)) * M(seg + 1);
            y_new.push_back(val);
        }
        return y_new;
    }

private:
    size_t n_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
};

}  // namespace xtgeo::numerics

#endif  // XTGEO_NUMERICS_HPP_
