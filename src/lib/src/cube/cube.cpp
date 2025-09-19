#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>
#include <xtgeo/cube.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>

#ifdef XTGEO_USE_OPENMP
#include <omp.h>
#endif
namespace py = pybind11;

namespace xtgeo::cube {

/**
 * Compute cube statistics per column, returning 2D arrays of min, max, mean,
 * var, rms, maxabs, maxpos, maxneg, meanabs, meanpos, meanneg, sumpos, sumneg, sumabs.
 *
 * @param cube Cube instance
 * @param upper_surface 2D array of upper surface Z values
 * @param lower_surface 2D array of lower surface Z values
 * @param depth_array 1D array of depth values corresponding to cube layers
 * @param ndiv Number of subdivisions between layers for interpolation (default: 1, no
 * subdivision)
 * @param interpolation Interpolation method ("cubic" or "linear")
 * @param min_thickness Minimum thickness for valid computation. Ie.e if lower - upper
 * maps <= min_thickness per map node, stats are NaN for those nodes (default: 0.0)
 * @param min_index Optional minimum Z index for cube slicing (default: 0)
 * @param max_index Optional maximum Z index for cube slicing (default: nlay)
 * @return A map of arrays: min, max, mean, var, rms, maxpos, maxneg, maxabs,
 * meanpos, meanneg, meanabs, sumpos, sumneg, sumabs
 */
std::unordered_map<std::string, py::array_t<double>>
cube_stats_along_z(const Cube &cube,
                   const py::array_t<double> &upper_surface,
                   const py::array_t<double> &lower_surface,
                   const py::array_t<double> &depth_array,
                   int ndiv,
                   const std::string &interpolation,
                   double min_thickness,
                   int min_index,
                   int max_index)
{
    auto &logger = xtgeo::logging::LoggerManager::get("xtgeo.cube.cube_stats_along_z");
    logger.debug("Computing cube statistics along Z axis");
    size_t ncol = cube.ncol;
    size_t nrow = cube.nrow;
    size_t nlay = cube.nlay;

    // Handle default slice indices
    if (max_index < 0) {
        max_index = static_cast<int>(nlay);
    }
    if (min_index < 0) {
        min_index = 0;
    }
    if (max_index > static_cast<int>(nlay)) {
        max_index = static_cast<int>(nlay);
    }

    auto upper_ = upper_surface.unchecked<2>();
    auto lower_ = lower_surface.unchecked<2>();
    auto depth_ = depth_array.unchecked<1>();

    py::array_t<double> minv({ ncol, nrow });
    py::array_t<double> maxv({ ncol, nrow });
    py::array_t<double> meanv({ ncol, nrow });
    py::array_t<double> varv({ ncol, nrow });
    py::array_t<double> rmsv({ ncol, nrow });
    py::array_t<double> maxposv({ ncol, nrow });
    py::array_t<double> maxnegv({ ncol, nrow });
    py::array_t<double> maxabsv({ ncol, nrow });
    py::array_t<double> meanposv({ ncol, nrow });
    py::array_t<double> meannegv({ ncol, nrow });
    py::array_t<double> meanabsv({ ncol, nrow });
    py::array_t<double> sumposv({ ncol, nrow });
    py::array_t<double> sumnegv({ ncol, nrow });
    py::array_t<double> sumabsv({ ncol, nrow });

    auto cubev_ = cube.values.unchecked<3>();  // Use unchecked for efficiency
    auto traceidcodes_ = cube.traceidcodes.unchecked<2>();  // Access traceidcodes
    auto minv_ = minv.mutable_unchecked<2>();
    auto maxv_ = maxv.mutable_unchecked<2>();
    auto meanv_ = meanv.mutable_unchecked<2>();
    auto varv_ = varv.mutable_unchecked<2>();
    auto rmsv_ = rmsv.mutable_unchecked<2>();
    auto maxposv_ = maxposv.mutable_unchecked<2>();
    auto maxnegv_ = maxnegv.mutable_unchecked<2>();
    auto maxabsv_ = maxabsv.mutable_unchecked<2>();
    auto meanposv_ = meanposv.mutable_unchecked<2>();
    auto meannegv_ = meannegv.mutable_unchecked<2>();
    auto meanabsv_ = meanabsv.mutable_unchecked<2>();
    auto sumposv_ = sumposv.mutable_unchecked<2>();
    auto sumnegv_ = sumnegv.mutable_unchecked<2>();
    auto sumabsv_ = sumabsv.mutable_unchecked<2>();

    // Pre-compute effective layer count and spline solver
    if (min_index >= max_index) {
        throw std::invalid_argument("min_index must be less than max_index");
    }

    size_t effective_nlay = max_index - min_index;
    std::unique_ptr<numerics::CubicSplineSolver> spline_solver;
    if (interpolation == "cubic" && ndiv > 1 && effective_nlay > 2) {
        std::vector<double> x_pts(effective_nlay);
        std::iota(x_pts.begin(), x_pts.end(), 0);
        spline_solver = std::make_unique<numerics::CubicSplineSolver>(x_pts);
    }

    // clang-format off
    #ifdef XTGEO_USE_OPENMP
        #pragma omp parallel for collapse(2) schedule(static)
    #endif
    // clang-format on
    for (int i = 0; i < static_cast<int>(ncol); i++) {  // int due to MSVC OpenMP...
        for (int j = 0; j < static_cast<int>(nrow); j++) {

            double upper_z = upper_(i, j);
            double lower_z = lower_(i, j);

            // Check if traceidcode indicates dead trace or thickness is too small
            if (traceidcodes_(i, j) == 2 || lower_z - upper_z <= min_thickness) {
                minv_(i, j) = numerics::QUIET_NAN;
                maxv_(i, j) = numerics::QUIET_NAN;
                meanv_(i, j) = numerics::QUIET_NAN;
                varv_(i, j) = numerics::QUIET_NAN;
                rmsv_(i, j) = numerics::QUIET_NAN;
                maxabsv_(i, j) = numerics::QUIET_NAN;
                meanabsv_(i, j) = numerics::QUIET_NAN;
                sumabsv_(i, j) = numerics::QUIET_NAN;
                maxposv_(i, j) = numerics::QUIET_NAN;
                meanposv_(i, j) = numerics::QUIET_NAN;
                sumposv_(i, j) = numerics::QUIET_NAN;
                maxnegv_(i, j) = numerics::QUIET_NAN;
                meannegv_(i, j) = numerics::QUIET_NAN;
                sumnegv_(i, j) = numerics::QUIET_NAN;

                continue;
            }

            // Extract trace using slice indices and perform interpolation
            std::vector<double> trace(effective_nlay);
            for (size_t k = 0; k < effective_nlay; ++k) {
                trace[k] = cubev_(i, j, min_index + k);
            }

            // Slice the depth array once - this is common to all branches
            std::vector<double> sliced_depth(effective_nlay);
            for (size_t k = 0; k < effective_nlay; ++k) {
                sliced_depth[k] = depth_(min_index + k);
            }

            std::vector<double> refined_trace;
            std::vector<double> refined_depth;

            if (spline_solver && ndiv > 1) {
                // Use pre-computed spline solver
                std::vector<double> x_pts(effective_nlay);
                std::iota(x_pts.begin(), x_pts.end(), 0);

                size_t num_new_points = (effective_nlay - 1) * ndiv + 1;
                std::vector<double> new_x_pts(num_new_points);
                double step =
                  static_cast<double>(effective_nlay - 1) / (num_new_points - 1);
                for (size_t k = 0; k < num_new_points; ++k) {
                    new_x_pts[k] = k * step;
                }

                refined_trace = spline_solver->interpolate(x_pts, trace, new_x_pts);
                refined_depth = numerics::linear_interpolate(sliced_depth, ndiv);

            } else if (ndiv > 1 && effective_nlay > 1) {  // Linear interpolation
                refined_trace = numerics::linear_interpolate(trace, ndiv);
                refined_depth = numerics::linear_interpolate(sliced_depth, ndiv);
            } else {  // No interpolation
                refined_trace = trace;
                refined_depth = numerics::linear_interpolate(sliced_depth, 1);
            }
            double min_val = std::numeric_limits<double>::max();
            double max_val = std::numeric_limits<double>::lowest();
            double sum = 0.0;
            double sum_sq = 0.0;
            double sum_abs = 0.0;
            double sum_pos = 0.0;
            double sum_neg = 0.0;
            double max_abs = 0.0;
            double max_pos = 0.0;
            double max_neg = 0.0;
            size_t n_items = 0;
            size_t n_items_pos = 0;
            size_t n_items_neg = 0;
            double mean = 0.0;
            double variance_sum = 0.0;

            for (size_t k = 0; k < refined_trace.size(); k++) {
                double depth = refined_depth[k];
                if (depth < upper_z || depth > lower_z) {
                    continue;
                }
                float value = refined_trace[k];
                if (std::isnan(value)) {
                    continue;
                }
                n_items++;

                min_val = std::min(min_val, static_cast<double>(value));
                max_val = std::max(max_val, static_cast<double>(value));
                sum += value;
                sum_sq += value * value;
                double abs_value = std::abs(value);
                sum_abs += abs_value;
                if (value >= 0) {
                    sum_pos += value;
                    n_items_pos++;
                    max_pos = std::max(max_pos, static_cast<double>(value));
                } else {
                    sum_neg += value;
                    n_items_neg++;
                    max_neg = std::min(max_neg, static_cast<double>(value));
                }
                max_abs = std::max(max_abs, abs_value);

                // Incremental variance computation using Welford's algorithm
                double delta = value - mean;
                mean += delta / n_items;
                double delta2 = value - mean;
                variance_sum += delta * delta2;
            }

            if (n_items > 0) {
                minv_(i, j) = min_val;
                maxv_(i, j) = max_val;
                meanv_(i, j) = mean;

                // Compute final variance
                double variance = variance_sum / n_items;
                varv_(i, j) = variance;

                // Compute RMS
                double rms = std::sqrt(sum_sq / n_items);
                rmsv_(i, j) = rms;

                // Compute other statistics
                maxabsv_(i, j) = max_abs;
                meanabsv_(i, j) = sum_abs / n_items;
                sumabsv_(i, j) = sum_abs;
            } else {
                minv_(i, j) = numerics::QUIET_NAN;
                maxv_(i, j) = numerics::QUIET_NAN;
                meanv_(i, j) = numerics::QUIET_NAN;
                varv_(i, j) = numerics::QUIET_NAN;
                rmsv_(i, j) = numerics::QUIET_NAN;
                maxabsv_(i, j) = numerics::QUIET_NAN;
                meanabsv_(i, j) = numerics::QUIET_NAN;
                sumabsv_(i, j) = numerics::QUIET_NAN;
            }

            // Compute statistics for positive values
            if (n_items_pos > 0) {
                maxposv_(i, j) = max_pos;
                meanposv_(i, j) = sum_pos / n_items_pos;
                sumposv_(i, j) = sum_pos;
            } else {
                maxposv_(i, j) = numerics::QUIET_NAN;
                meanposv_(i, j) = numerics::QUIET_NAN;
                sumposv_(i, j) = numerics::QUIET_NAN;
            }

            // Compute statistics for negative values
            if (n_items_neg > 0) {
                maxnegv_(i, j) = max_neg;
                meannegv_(i, j) = sum_neg / n_items_neg;
                sumnegv_(i, j) = sum_neg;
            } else {
                maxnegv_(i, j) = numerics::QUIET_NAN;
                meannegv_(i, j) = numerics::QUIET_NAN;
                sumnegv_(i, j) = numerics::QUIET_NAN;
            }
        }
    }

    // Populate the map with the computed statistics
    std::unordered_map<std::string, py::array_t<double>> stats;
    stats["min"] = minv;
    stats["max"] = maxv;
    stats["mean"] = meanv;
    stats["var"] = varv;
    stats["rms"] = rmsv;
    stats["maxpos"] = maxposv;
    stats["maxneg"] = maxnegv;
    stats["maxabs"] = maxabsv;
    stats["meanpos"] = meanposv;
    stats["meanneg"] = meannegv;
    stats["meanabs"] = meanabsv;
    stats["sumpos"] = sumposv;
    stats["sumneg"] = sumnegv;
    stats["sumabs"] = sumabsv;

    logger.debug("Cube statistics computed successfully");
    return stats;
}  // cube_stats_along_z()

}  // namespace xtgeo::cube
