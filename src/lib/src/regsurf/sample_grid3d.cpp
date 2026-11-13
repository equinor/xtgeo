#include <pybind11/pybind11.h>  // always in top, refer to pybind11 docs
#include <pybind11/numpy.h>
#include <cmath>
#include <cstddef>

#ifdef XTGEO_USE_OPENMP
#include <omp.h>
#endif

#include <iostream>
#include <vector>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/regsurf.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

/*
 * @brief Sample I J and depths from 3D grid to regularsurface
 *
 * @param regsurf RegularSurface object representing the surface input
 * @param grd Grid object representing the 3D grid
 * @param layers The layer to sample per column, base 0; -1 skips a column
 * @param index_position 0: top, 1: base|bot, 2: center
 * @param boundary -1: stored layer, 0: grid top, 1: grid base
 * @return Tuple of 6 numpy arrays: I index, J index, K index, Depth_top,
 * Depth_base, Inactive
 */
std::tuple<py::array_t<int>,
           py::array_t<int>,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
sample_grid3d(const RegularSurface &regsurf,
              const grid3d::Grid &grd,
              const py::array_t<int> &layers,
              const int index_position,  // 0: top, 1: base|bot, 2: center
              const int boundary,
              const int num_threads)
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("xtgeo.regsurf.sample_grid3d_layer");
    logger.debug("Sampling grid3d to regular surface");
    logger.debug("Initial number of threads: {}", num_threads);

    // clang-format off
    #ifdef XTGEO_USE_OPENMP
      if (num_threads > 0) omp_set_num_threads(num_threads);
      logger.debug("Actual number of threads: {}", omp_get_max_threads());
    #endif
    // clang-format on

    // Check if yinc is negative which may occur if the RegularSurface is flipped
    if (regsurf.yinc < 0) {
        throw py::value_error("yinc must be positive, but got " +
                              std::to_string(regsurf.yinc));
    }

    if (layers.ndim() != 2 || !shape_equals(layers.shape(0), grd.get_ncol()) ||
        !shape_equals(layers.shape(1), grd.get_nrow())) {
        throw py::value_error("layers must have the grid's column and row dimensions");
    }
    if (boundary < -1 || boundary > 1) {
        throw py::value_error("boundary must be -1, 0 or 1");
    }
    auto klayers = layers.unchecked<2>();
    for (size_t i = 0; i < grd.get_ncol(); ++i) {
        for (size_t j = 0; j < grd.get_nrow(); ++j) {
            if (klayers(i, j) >= static_cast<int>(grd.get_nlay())) {
                throw py::value_error("layer index is outside the grid");
            }
        }
    }

    auto actnumsv_ = grd.get_actnumsv().unchecked<3>();

    // Initialize 2D numpy arrays to store the sampled values
    py::array_t<int> iindex({ regsurf.ncol, regsurf.nrow });
    py::array_t<int> jindex({ regsurf.ncol, regsurf.nrow });
    py::array_t<int> kindex({ regsurf.ncol, regsurf.nrow });
    py::array_t<double> depth_top({ regsurf.ncol, regsurf.nrow });
    py::array_t<double> depth_bot({ regsurf.ncol, regsurf.nrow });
    py::array_t<bool> inactive({ regsurf.ncol, regsurf.nrow });

    // Get unchecked access to the arrays
    auto iindex_ = iindex.mutable_unchecked<2>();
    auto jindex_ = jindex.mutable_unchecked<2>();
    auto kindex_ = kindex.mutable_unchecked<2>();
    auto depth_top_ = depth_top.mutable_unchecked<2>();
    auto depth_bot_ = depth_bot.mutable_unchecked<2>();
    auto inactive_ = inactive.mutable_unchecked<2>();

    // Set all elements to -1 or undef initially
    std::fill(iindex_.mutable_data(0, 0),
              iindex_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), -1);
    std::fill(jindex_.mutable_data(0, 0),
              jindex_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), -1);
    std::fill(kindex.mutable_data(), kindex.mutable_data() + kindex.size(), -1);
    std::fill(inactive_.mutable_data(0, 0),
              inactive_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), false);

    std::fill(depth_top_.mutable_data(0, 0),
              depth_top_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow),
              numerics::QUIET_NAN);
    std::fill(depth_bot_.mutable_data(0, 0),
              depth_bot_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow),
              numerics::QUIET_NAN);

    grd.get_cell_corners_cache();  // Ensure cell corners are computed

    // Windows (MSVC) does not support size_t in OpenMP (!), so we use int
    int ncol = static_cast<int>(grd.get_ncol());
    int nrow = static_cast<int>(grd.get_nrow());
    std::vector<double> selected_depth(regsurf.ncol * regsurf.nrow,
                                       numerics::QUIET_NAN);
    std::vector<unsigned char> selected_boundary(regsurf.ncol * regsurf.nrow, false);
    // clang-format off
    #ifdef XTGEO_USE_OPENMP
      // Lock only the sampled node, so unrelated output updates can run in parallel.
      std::vector<omp_lock_t> locks(regsurf.ncol * regsurf.nrow);
      for (auto &lock : locks)
          omp_init_lock(&lock);
    #endif
    // clang-format on

    // clang-format off
    #ifdef XTGEO_USE_OPENMP
      #pragma omp parallel for collapse(2) schedule(static)
    #endif
    // clang-format on
    for (int icell = 0; icell < ncol; icell++) {
        for (int jcell = 0; jcell < nrow; jcell++) {

            size_t i_cell = static_cast<size_t>(icell);
            size_t j_cell = static_cast<size_t>(jcell);
            int klayer = klayers(i_cell, j_cell);
            if (klayer < 0)
                continue;
            // Get cell corners
            auto corners =
              grid3d::get_cell_corners_from_ijk(grd, i_cell, j_cell, klayer);

            // Find the min/max of the cell corners. This is the bounding box of the
            // cell and will narrow the search for the points that are within the
            auto minmax = grid3d::get_corners_minmax(corners);
            auto [xmin, xmax, ymin, ymax, zmin, zmax] = std::tie(
              minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]);

            // Find the range of the cells (expanded <expand> cell) in the local map
            int expand = 0;
            auto [mxmin, mxmax, mymin, mymax] =
              regsurf::find_cell_range(regsurf, xmin, xmax, ymin, ymax, expand);

            if (mxmin == -1) {
                // Cell is outside the local map
                continue;
            }

            // Loop over the local map
            for (size_t j = mymin; j <= mymax; j++) {
                for (size_t i = mxmin; i <= mxmax; i++) {

                    auto p = regsurf::get_xy_from_ij(regsurf, i, j);
                    // Check if the point is within the cell
                    bool is_inside_top =
                      grid3d::is_xy_point_in_cell(p.x(), p.y(), corners, 0);

                    bool is_inside_bot =
                      grid3d::is_xy_point_in_cell(p.x(), p.y(), corners, 1);

                    bool is_inside_mid =
                      grid3d::is_xy_point_in_cell(p.x(), p.y(), corners, 2);

                    if (is_inside_top && is_inside_bot && !is_inside_mid) {
                        // not sure if this is a bug... but set is_inside_mid too
                        is_inside_mid = is_inside_top;
                    }

                    double new_depth_top =
                      is_inside_top
                        ? grid3d::get_depth_in_cell(p.x(), p.y(), corners, 0)
                        : numerics::QUIET_NAN;
                    double new_depth_bot =
                      is_inside_bot
                        ? grid3d::get_depth_in_cell(p.x(), p.y(), corners, 1)
                        : numerics::QUIET_NAN;
                    bool contains_index = index_position == 0   ? is_inside_top
                                          : index_position == 1 ? is_inside_bot
                                                                : is_inside_mid;
                    double depth = boundary == 1 ? new_depth_bot : new_depth_top;
                    bool has_boundary_depth = !std::isnan(depth);
                    if (contains_index && std::isnan(depth)) {
                        depth = grid3d::cell_center(corners).z();
                    }
                    // Check if the cell is active or not
                    bool cell_inactive = actnumsv_(i_cell, j_cell, klayer) == 0;
                    size_t node = i * regsurf.nrow + j;
                    // clang-format off
                    #ifdef XTGEO_USE_OPENMP
                      omp_set_lock(&locks[node]);
                    #endif
                    // clang-format on
                    if (is_inside_top && (std::isnan(depth_top_(i, j)) ||
                                          new_depth_top < depth_top_(i, j))) {
                        // Update the depth if the new depth is smaller
                        depth_top_(i, j) = new_depth_top;
                    }
                    if (is_inside_bot &&
                        (std::isnan(depth_bot_(i, j)) ||
                         (boundary == 1 ? new_depth_bot > depth_bot_(i, j)
                                        : new_depth_bot < depth_bot_(i, j)))) {
                        depth_bot_(i, j) = new_depth_bot;
                    }
                    // the selected cell determines activity and i, j, k indices
                    if (contains_index) {
                        double previous = selected_depth[node];
                        int cell = icell * nrow + jcell;
                        int previous_cell = iindex_(i, j) * nrow + jindex_(i, j);
                        // Exact keys: the result does not depend on thread order.
                        bool choose;
                        if (iindex_(i, j) < 0) {
                            choose = true;
                        } else if (cell_inactive != inactive_(i, j)) {
                            choose = !cell_inactive;
                        } else if (boundary == -1) {
                            choose = cell > previous_cell;
                        } else if (has_boundary_depth != selected_boundary[node]) {
                            // Prefer a real boundary depth over a center fallback.
                            choose = has_boundary_depth;
                        } else if (depth != previous) {
                            choose =
                              boundary == 1 ? depth > previous : depth < previous;
                        } else {
                            choose = cell < previous_cell;
                        }
                        if (choose) {
                            iindex_(i, j) = i_cell;
                            jindex_(i, j) = j_cell;
                            kindex_(i, j) = klayer;
                            inactive_(i, j) = cell_inactive;
                            selected_depth[node] = depth;
                            selected_boundary[node] = has_boundary_depth;
                        }
                    }
                    // clang-format off
                    #ifdef XTGEO_USE_OPENMP
                      omp_unset_lock(&locks[node]);
                    #endif
                    // clang-format on
                }
            }
        }
    }

    // clang-format off
    #ifdef XTGEO_USE_OPENMP
      for (auto &lock : locks)
          omp_destroy_lock(&lock);
    #endif
    // clang-format on
    return std::make_tuple(iindex, jindex, kindex, depth_top, depth_bot, inactive);
}

std::tuple<py::array_t<int>,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
sample_grid3d_layer(const RegularSurface &regsurf,
                    const grid3d::Grid &grd,
                    const size_t klayer,
                    const int index_position,
                    const int num_threads)
{
    py::array_t<int> layers({ grd.get_ncol(), grd.get_nrow() });
    std::fill(layers.mutable_data(), layers.mutable_data() + layers.size(), klayer);
    auto [i, j, k, top, base, inactive] =
      sample_grid3d(regsurf, grd, layers, index_position, -1, num_threads);
    return { i, j, top, base, inactive };
}

}  // namespace xtgeo::regsurf
