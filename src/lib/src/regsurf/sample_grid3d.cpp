#include <pybind11/pybind11.h>  // always in top, refer to pybind11 docs
#include <pybind11/numpy.h>
#include <cmath>  // Include for std::isnan
#include <cstddef>
#include <vector>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/regsurf.hpp>
namespace py = pybind11;

namespace xtgeo::regsurf {

/*
 * Sample I J and depths from 3D grid to regularsurface
 * @param ncol Number of columns in the regular surface
 * @param nrow Number of rows in the regular surface
 * @param xori X origin of the regular surface
 * @param yori Y origin of the regular surface
 * @param xinc X increment of the regular surface
 * @param yinc Y increment of the regular surface
 * @param rotation Rotation of the regular surface
 * @param ncolgrid3d Number of columns in the 3D grid
 * @param nrowgrid3d Number of rows in the 3D grid
 * @param nlaygrid3d Number of layers in the 3D grid
 * @param coordsv Coordinates of the 3D grid
 * @param zcornsv Z corners of the 3D grid
 * @param actnumsv Active cells of the 3D grid
 * @param klayer The layer to sample, base 0
 * @param option Option to sample the top (0) or bottom (1) of the cell
 * @param activeonly If 1, only sample active cells
 * @return Tuple of 3 numpy arrays: I index, J index, Depth
 */

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>>
sample_grid3d_layer(const size_t ncol,
                    const size_t nrow,
                    const double xori,
                    const double yori,
                    const double xinc,
                    const double yinc,
                    const double rotation,
                    const size_t ncolgrid3d,
                    const size_t nrowgrid3d,
                    const size_t nlaygrid3d,
                    const py::array_t<double> &coordsv,
                    const py::array_t<float> &zcornsv,
                    const py::array_t<int> &actnumsv,
                    const size_t klayer,
                    const int option,
                    const int activeonly)
{
    Logger logger(__func__);
    logger.debug("Sampling 3D grid layer to a regular surface...");

    // Check if yinc is negative which may occur if the RegilarSurface is flipped
    if (yinc < 0) {
        throw py::value_error("yinc must be positive, but got " + std::to_string(yinc));
    }

    // Initialize 2D numpy arrays to store the sampled values
    py::array_t<int> iindex({ ncol, nrow });
    py::array_t<int> jindex({ ncol, nrow });
    py::array_t<double> depth({ ncol, nrow });

    // Get unchecked access to the arrays
    auto iindex_ = iindex.mutable_unchecked<2>();
    auto jindex_ = jindex.mutable_unchecked<2>();
    auto depth_ = depth.mutable_unchecked<2>();

    // Set all elements to -1 or nan initially
    std::fill(iindex_.mutable_data(0, 0), iindex_.mutable_data(0, 0) + (ncol * nrow),
              -1);
    std::fill(jindex_.mutable_data(0, 0), jindex_.mutable_data(0, 0) + (ncol * nrow),
              -1);
    std::fill(depth_.mutable_data(0, 0), depth_.mutable_data(0, 0) + (ncol * nrow),
              std::numeric_limits<double>::max());

    // Loop over the grid
    logger.debug("Looping 3D GRID cell NCOLROW and NROW is", ncolgrid3d, nrowgrid3d);
    for (size_t icell = 0; icell < ncolgrid3d; icell++) {
        for (size_t jcell = 0; jcell < nrowgrid3d; jcell++) {
<<<<<<< HEAD

            // Check if the cell is active
            if (activeonly == 1 & actnumsv.at(icell, jcell, klayer) == 0) {
                continue;
            }
=======
>>>>>>> 05a56c32 (ENH: rewrite/improve surface_from_grid3d function)
            // Get cell corners
            auto corners =
              grid3d::cell_corners(icell, jcell, klayer, ncolgrid3d, nrowgrid3d,
                                   nlaygrid3d, coordsv, zcornsv);

            // Find the min/max of the cell corners. This is the bounding box of the
            // cell and will narrow the search for the points that are within the cell
            auto minmax = grid3d::get_corners_minmax(corners);
            auto [xmin, xmax, ymin, ymax, zmin, zmax] = std::tie(
              minmax[0], minmax[1], minmax[2], minmax[3], minmax[4], minmax[5]);

            // Find the range of the cells (expanded 1 cell) in the local map
            auto [mxmin, mxmax, mymin, mymax] = regsurf::find_cell_range(
              xmin, xmax, ymin, ymax, xori, yori, xinc, yinc, rotation, ncol, nrow, 1);

            if (mxmin == -1) {
                // Cell is outside the local map
                continue;
            }

            // Loop over the local map
            for (size_t j = mymin; j <= mymax; j++) {
                for (size_t i = mxmin; i <= mxmax; i++) {

                    auto p = regsurf::get_xy_from_ij(i, j, xori, yori, xinc, yinc, ncol,
                                                     nrow, rotation);
                    // Check if the point is within the cell
                    bool is_inside = grid3d::is_xy_point_in_cell(
                      p.x, p.y, icell, jcell, klayer, ncolgrid3d, nrowgrid3d,
                      nlaygrid3d, coordsv, zcornsv, option);

                    if (is_inside) {
                        auto previous_depth = depth_(i, j);
                        auto new_depth = grid3d::get_depth_in_cell(
                          p.x, p.y, icell, jcell, klayer, ncolgrid3d, nrowgrid3d,
                          nlaygrid3d, coordsv, zcornsv, option);
                        // keep the cell which has the smallest depth (in case of e.g.
                        // reverse faulting which gives overlapping cells)
                        if (std::isnan(previous_depth) || new_depth < previous_depth) {
                            depth_(i, j) = new_depth;
                            iindex_(i, j) = icell;
                            jindex_(i, j) = jcell;
                        }
                    }
                }
            }
        }
    }

    logger.debug("Sampling 3D grid layer to a regular surface... done");
    return std::make_tuple(iindex, jindex, depth);
}  // regsurf_sample_grid3d_layer

}  // namespace xtgeo::regsurf
