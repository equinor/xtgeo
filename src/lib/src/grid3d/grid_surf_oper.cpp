// File: grid_surf_oper.cpp focus on grid or gridproperty that are associated with
// and/or modified by surface(s) input. This file is part of the xtgeo library.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/regsurf.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Create a parameter that is 1 if the cell center is between the top and bottom
 * surfaces
 *
 * @param grd Grid instance
 * @param top The top surface (RegularSurface object)
 * @param bot The bottom surface (RegularSurface object)
 * @return An int array containing 1 if the cell is between the surfaces, 0 otherwise
 */

py::array_t<int8_t>
get_gridprop_value_between_surfaces(const Grid &grd,
                                    const regsurf::RegularSurface &top,
                                    const regsurf::RegularSurface &bot)
{
    pybind11::array_t<int8_t> result({ grd.ncol, grd.nrow, grd.nlay });
    auto result_ = result.mutable_unchecked<3>();

    py::array_t<double> xmid, ymid, zmid;
    std::tie(xmid, ymid, zmid) = get_cell_centers(grd, true);

    // Access the np array without bounds checking for optimizing speed
    auto xmid_ = xmid.unchecked<3>();
    auto ymid_ = ymid.unchecked<3>();
    auto zmid_ = zmid.unchecked<3>();

    for (size_t i = 0; i < grd.ncol; i++) {
        for (size_t j = 0; j < grd.nrow; j++) {
            for (size_t k = 0; k < grd.nlay; k++) {
                // for every cell, project the center to the top and bottom surfaces
                double xm = xmid_(i, j, k);
                double ym = ymid_(i, j, k);
                double zm = zmid_(i, j, k);

                // check if zm is NaN which can occur for inactive cells
                if (std ::isnan(zm)) {
                    result_(i, j, k) = 0;
                    continue;
                }

                double top_z = regsurf::get_z_from_xy(top, xm, ym);
                double bot_z = regsurf::get_z_from_xy(bot, xm, ym);

                // check if top_z or bot_z is NaN
                if (std::isnan(top_z) || std::isnan(bot_z)) {
                    result_(i, j, k) = 0;
                    continue;
                }

                if (zm >= top_z && zm <= bot_z) {  // inside
                    result_(i, j, k) = 1;
                } else {
                    result_(i, j, k) = 0;
                }
            }
        }
    }
    return result;
}  // get_gridprop_value_between_surfaces

/*
 * Given a 3D grid and surfaces, construct layers in 3D grid (zcorns) from surface. Note
 * that this grid is based on a "shoebox grid", which has vertical pillars in the input!
 *
 * @param grd Grid instance, which already has the correct number of layers
 * @param rsurfs The RegularSurface objects (array/list input), sorted from top
 * to bottom
 * @param tolerance The tolerance for the interpolation
 * @return updated zcorn and actnum values in the grid
 */

std::tuple<py::array_t<float>, py::array_t<int>>
adjust_boxgrid_layers_from_regsurfs(Grid &grd,
                                    const std::vector<regsurf::RegularSurface> &rsurfs,
                                    const double tolerance)
{
    std::vector<size_t> shape = { grd.ncol + 1, grd.nrow + 1, grd.nlay + 1, 4 };
    // Create the array with the specified shape
    py::array_t<float> zcorn_result(shape);
    py::array_t<int> actnum_result({ grd.ncol, grd.nrow, grd.nlay });

    auto zcorn_result_ = zcorn_result.mutable_unchecked<4>();
    auto actnum_result_ = actnum_result.mutable_unchecked<3>();

    auto coordsv_ = grd.coordsv.unchecked<3>();

    for (size_t i = 0; i < actnum_result_.shape(0); ++i) {
        for (size_t j = 0; j < actnum_result_.shape(1); ++j) {
            for (size_t k = 0; k < actnum_result_.shape(2); ++k) {
                actnum_result_(i, j, k) = 1;
            }
        }
    }

    if (rsurfs.size() != grd.nlay + 1) {
        throw std::invalid_argument("Wrong number of input surfaces vs grid layers");
    }

    for (size_t icol = 0; icol < grd.ncol + 1; icol++) {
        for (size_t jrow = 0; jrow < grd.nrow + 1; jrow++) {
            // Get pillar coordinate, just using the top pillar point as the pillars
            // shall be vertical
            double x = coordsv_(icol, jrow, 0);
            double y = coordsv_(icol, jrow, 1);
            size_t ksurf = 0;

            int actnum = 1;

            for (const auto &rsurf : rsurfs) {
                double z = regsurf::get_z_from_xy(rsurf, x, y, tolerance);

                if (std::isnan(z)) {
                    z = 1000;  // Default value; TODO make this better
                    actnum = 0;
                }

                // Update the z values of the cell corners
                for (size_t l = 0; l < 4; l++) {
                    zcorn_result_(icol, jrow, ksurf, l) = z;
                }

                ksurf++;
            }

            // If actnum is 0, then all 4 cells that share this pillar should be
            // inactive for the entire column
            if (actnum == 0) {
                int ia = static_cast<int>(icol);
                int ja = static_cast<int>(jrow);

                for (size_t k = 0; k < grd.nlay; k++) {
                    for (int ii = ia - 1; ii <= ia; ii++) {
                        for (int jj = ja - 1; jj <= ja; jj++) {
                            // Ensure indices are within bounds
                            size_t apply_i =
                              std::clamp(ii, 0, static_cast<int>(grd.ncol - 1));
                            size_t apply_j =
                              std::clamp(jj, 0, static_cast<int>(grd.nrow - 1));

                            // Set actnum_result_ to 0 for the entire column
                            actnum_result_(apply_i, apply_j, k) = 0;
                        }
                    }
                }
            }
        }
    }
    return std::make_tuple(zcorn_result, actnum_result);
}

}  // namespace xtgeo::grid3d
