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

namespace py = pybind11;

namespace xtgeo::cube {

/*
 * Compute cube statistics per column, returning 2D arrays of min, max, mean, var, rms,
 * maxabs, maxpos, maxneg, meanabs, meanpos, meanneg, sumpos, sumneg, sumabs. This could
 * technically be done in numpy, but this approach is more efficient as it can do all
 * operations per column in one operation.
 *
 * @param ncol Cube dimensions ncol
 * @param nrow Cube dimensions nrow
 * @param nlay Cube dimensions nlay
 * @param cubev Cube values vector
 * @return A map of arrays: min, max, mean, var, rms, maxpos, maxneg, maxabs,
 * meanpos, meanneg, meanabs, sumpos, sumneg, sumabs
 */

std::unordered_map<std::string, py::array_t<double>>
cube_stats_along_z(const size_t ncol,
                   const size_t nrow,
                   const size_t nlay,
                   const py::array_t<float> &cubev)
{
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

    auto cubev_ = cubev.unchecked<3>();  // Use unchecked for efficiency
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

    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
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

            for (size_t k = 0; k < nlay; k++) {
                float value = cubev_(i, j, k);
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
                minv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                maxv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                meanv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                varv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                rmsv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                maxabsv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                meanabsv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                sumabsv_(i, j) = std::numeric_limits<double>::quiet_NaN();
            }

            // Compute statistics for positive values
            if (n_items_pos > 0) {
                maxposv_(i, j) = max_pos;
                meanposv_(i, j) = sum_pos / n_items_pos;
                sumposv_(i, j) = sum_pos;
            } else {
                maxposv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                meanposv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                sumposv_(i, j) = std::numeric_limits<double>::quiet_NaN();
            }

            // Compute statistics for negative values
            if (n_items_neg > 0) {
                maxnegv_(i, j) = max_neg;
                meannegv_(i, j) = sum_neg / n_items_neg;
                sumnegv_(i, j) = sum_neg;
            } else {
                maxnegv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                meannegv_(i, j) = std::numeric_limits<double>::quiet_NaN();
                sumnegv_(i, j) = std::numeric_limits<double>::quiet_NaN();
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

    return stats;
}  // cube_stats_along_z()

}  // namespace xtgeo::cube
