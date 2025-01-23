#include <pybind11/pybind11.h>  // always in top, refer to pybind11 docs
#include <pybind11/numpy.h>
#include <cmath>
#include <cstddef>

#ifdef __linux__
#include <omp.h>
#endif

#include <iostream>
#include <vector>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/regsurf.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

/*
 * @brief Sample I J and depths from 3D grid to regularsurface
 *
 * @param regsurf RegularSurface object representing the surface input
 * @param grd Grid object representing the 3D grid
 * @param klayer The layer to sample, base 0
 * @param index_position 0: top, 1: base|bot, 2: center
 * @return Tuple of 5 numpy arrays: I index, J index, Depth_top, Depth_base, Inactive
 */
std::tuple<py::array_t<int>,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
sample_grid3d_layer(const RegularSurface &regsurf,
                    const grid3d::Grid &grd,
                    const size_t klayer,
                    const int index_position,  // 0: top, 1: base|bot, 2: center
                    const int num_threads = -1)
{
    // clang-format off
    #ifdef __linux__
      if (num_threads > 0) omp_set_num_threads(num_threads);
    #endif
    // clang-format on

    // Check if yinc is negative which may occur if the RegularSurface is flipped
    if (regsurf.yinc < 0) {
        throw py::value_error("yinc must be positive, but got " +
                              std::to_string(regsurf.yinc));
    }

    auto actnumsv_ = grd.actnumsv.unchecked<3>();

    // Initialize 2D numpy arrays to store the sampled values
    py::array_t<int> iindex({ regsurf.ncol, regsurf.nrow });
    py::array_t<int> jindex({ regsurf.ncol, regsurf.nrow });
    py::array_t<double> depth_top({ regsurf.ncol, regsurf.nrow });
    py::array_t<double> depth_bot({ regsurf.ncol, regsurf.nrow });
    py::array_t<bool> inactive({ regsurf.ncol, regsurf.nrow });

    // Get unchecked access to the arrays
    auto iindex_ = iindex.mutable_unchecked<2>();
    auto jindex_ = jindex.mutable_unchecked<2>();
    auto depth_top_ = depth_top.mutable_unchecked<2>();
    auto depth_bot_ = depth_bot.mutable_unchecked<2>();
    auto inactive_ = inactive.mutable_unchecked<2>();

    // Set all elements to -1 or undef initially
    std::fill(iindex_.mutable_data(0, 0),
              iindex_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), -1);
    std::fill(jindex_.mutable_data(0, 0),
              jindex_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), -1);
    std::fill(inactive_.mutable_data(0, 0),
              inactive_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow), false);

    std::fill(depth_top_.mutable_data(0, 0),
              depth_top_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow),
              numerics::QUIET_NAN);
    std::fill(depth_bot_.mutable_data(0, 0),
              depth_bot_.mutable_data(0, 0) + (regsurf.ncol * regsurf.nrow),
              numerics::QUIET_NAN);

    // clang-format off
    #pragma omp parallel for collapse(2) schedule(static)
    // clang-format on

    for (size_t icell = 0; icell < grd.ncol; icell++) {
        for (size_t jcell = 0; jcell < grd.nrow; jcell++) {

            // Get cell corners
            auto corners = grid3d::get_cell_corners_from_ijk(grd, icell, jcell, klayer);

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
                      grid3d::is_xy_point_in_cell(p.x, p.y, corners, 0);

                    bool is_inside_bot =
                      grid3d::is_xy_point_in_cell(p.x, p.y, corners, 1);

                    bool is_inside_mid =
                      grid3d::is_xy_point_in_cell(p.x, p.y, corners, 2);

                    if (is_inside_top && is_inside_bot && !is_inside_mid) {
                        // not sure if this is a bug... but set is_inside_mid too
                        is_inside_mid = is_inside_top;
                    }

                    if (is_inside_top) {
                        auto previous_depth_top = depth_top_(i, j);
                        auto new_depth_top =
                          grid3d::get_depth_in_cell(p.x, p.y, corners, 0);

                        if (std::isnan(previous_depth_top) ||
                            new_depth_top < previous_depth_top) {
                            // Update the depth if the new depth is smaller
                            depth_top_(i, j) = new_depth_top;
                        }
                        if (index_position == 0) {
                            iindex_(i, j) = icell;
                            jindex_(i, j) = jcell;
                        }
                    }
                    if (is_inside_bot) {
                        auto previous_depth_bot = depth_bot_(i, j);
                        auto new_depth_bot =
                          grid3d::get_depth_in_cell(p.x, p.y, corners, 1);
                        if (std::isnan(previous_depth_bot) ||
                            new_depth_bot < previous_depth_bot) {
                            depth_bot_(i, j) = new_depth_bot;
                        }
                        if (index_position == 1) {
                            iindex_(i, j) = icell;
                            jindex_(i, j) = jcell;
                        }
                    }
                    // the mid cell determines the active status and i, j index
                    if (is_inside_mid) {
                        // Check if the cell is active or not
                        if (actnumsv_(icell, jcell, klayer) == 0) {
                            inactive_(i, j) = true;
                        }
                        if (index_position < 0 || index_position > 1) {
                            iindex_(i, j) = icell;
                            jindex_(i, j) = jcell;
                        }
                    }
                }
            }
        }
    }

    return std::make_tuple(iindex, jindex, depth_top, depth_bot, inactive);
}  // regsurf_sample_grid3d_layer

}  // namespace xtgeo::regsurf
