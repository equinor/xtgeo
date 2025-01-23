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

}  // namespace xtgeo::grid3d
