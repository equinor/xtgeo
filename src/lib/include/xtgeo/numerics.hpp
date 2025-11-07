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

inline xyz::Point
lerp3d(xyz::Point pt1, xyz::Point pt2, double t)
{
    return xyz::Point{ pt1.x() + t * (pt2.x() - pt1.x()),
                       pt1.y() + t * (pt2.y() - pt1.y()),
                       pt1.z() + t * (pt2.z() - pt1.z()) };
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
        std::vector<double> result(n_points);
        for (size_t i = 0; i < n_points; ++i) {
            result[i] = points[i];
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
 * @brief A pre-computing solver for cubic spline interpolation.
 *
 * This class computes and stores the LU decomposition of the spline matrix,
 * allowing for rapid solving for multiple right-hand side vectors (i.e.,
 * multiple data traces). Resembles scipy's CubicSpline with 'not-a-knot' boundary
 * conditions.
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
        if (n_ < 2) {  // Allow for linear interpolation case
            return;
        }
        h_.resize(n_ - 1);
        for (size_t i = 0; i < n_ - 1; ++i) {
            h_[i] = x_pts[i + 1] - x_pts[i];
            if (!(h_[i] > 0.0)) {  // Epsilon comparison might be safer
                throw std::invalid_argument(
                  "CubicSplineSolver: x_pts must be strictly increasing");
            }
        }

        if (n_ < 3) {  // Not enough points for cubic spline
            return;
        }

        Eigen::SparseMatrix<double> A(n_, n_);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(3 * n_ - 4);  // More precise reservation

        // Standard interior points
        for (size_t i = 1; i < n_ - 1; ++i) {
            triplets.emplace_back(i, i - 1, h_[i - 1]);
            triplets.emplace_back(i, i, 2 * (h_[i - 1] + h_[i]));
            triplets.emplace_back(i, i + 1, h_[i]);
        }

        // Not-a-knot boundary conditions
        triplets.emplace_back(0, 0, h_[1]);
        triplets.emplace_back(0, 1, -(h_[0] + h_[1]));
        triplets.emplace_back(0, 2, h_[0]);
        triplets.emplace_back(n_ - 1, n_ - 3, h_[n_ - 2]);
        triplets.emplace_back(n_ - 1, n_ - 2, -(h_[n_ - 3] + h_[n_ - 2]));
        triplets.emplace_back(n_ - 1, n_ - 1, h_[n_ - 3]);

        A.setFromTriplets(triplets.begin(), triplets.end());
        solver_.compute(A);
        if (solver_.info() != Eigen::Success) {
            // It's good practice to check for decomposition failure here.
            // Depending on desired behavior, you could throw or set a failure state.
        }
    }

    /**
     * @brief Interpolates y-values using the pre-computed solver.
     */
    std::vector<double> interpolate(const std::vector<double> &x_pts,
                                    const std::vector<double> &y_pts,
                                    const std::vector<double> &new_x_pts) const
    {
        if (n_ != x_pts.size() || n_ != y_pts.size()) {
            throw std::invalid_argument("Input vector sizes do not match solver size.");
        }

        if (n_ < 2) {
            return std::vector<double>(new_x_pts.size(), n_ == 1 ? y_pts[0] : 0.0);
        }
        if (n_ == 2) {  // Handle linear case explicitly
            std::vector<double> result;
            result.reserve(new_x_pts.size());
            if (h_[0] <= 0)
                return std::vector<double>(new_x_pts.size(),
                                           y_pts[0]);  // Avoid division by zero
            for (const auto &x : new_x_pts) {
                double t = (x - x_pts[0]) / h_[0];
                result.push_back(y_pts[0] + t * (y_pts[1] - y_pts[0]));
            }
            return result;
        }

        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_);
        for (size_t i = 1; i < n_ - 1; ++i) {
            b(i) = 6.0 * ((y_pts[i + 1] - y_pts[i]) / h_[i] -
                          (y_pts[i] - y_pts[i - 1]) / h_[i - 1]);
        }

        if (solver_.info() != Eigen::Success) {
            // Consider throwing an exception or returning an empty vector
            // to indicate that the solver was not successfully initialized.
            return {};
        }
        Eigen::VectorXd M = solver_.solve(b);
        if (solver_.info() != Eigen::Success) {
            // Solving failed for this specific `b` vector.
            return {};
        }

        std::vector<double> y_new;
        y_new.reserve(new_x_pts.size());

        size_t seg = 0;
        for (const auto &x : new_x_pts) {
            // Find segment for x, can be optimized if new_x_pts is sorted
            while (seg < n_ - 2 && x_pts[seg + 1] < x) {
                seg++;
            }

            double dx1 = x - x_pts[seg];
            double dx2 = x_pts[seg + 1] - x;
            double h_seg = h_[seg];

            double val =
              (dx2 / h_seg) * y_pts[seg] + (dx1 / h_seg) * y_pts[seg + 1] +
              (dx1 * dx1 * dx1 - h_seg * h_seg * dx1) * M(seg + 1) / (6 * h_seg) +
              (dx2 * dx2 * dx2 - h_seg * h_seg * dx2) * M(seg) / (6 * h_seg);
            y_new.push_back(val);
        }
        return y_new;
    }

private:
    size_t n_;
    std::vector<double> h_;
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver_;
};

}  // namespace xtgeo::numerics

#endif  // XTGEO_NUMERICS_HPP_
