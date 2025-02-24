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
 * that this grid shall be a "shoebox grid", which has vertical pillars in the input!
 *
 * @param grd Grid instance, which already has the correct number of layers
 * @param rsurfs The RegularSurface objects (array/list input), sorted from top
 * @return updated zcorn values in the grid
 */

py::array_t<float>
adjust_boxgrid_layers_from_regsurfs(Grid &grd,
                                    const std::vector<regsurf::RegularSurface> &rsurfs)
{
    std::vector<size_t> shape = { grd.ncol + 1, grd.nrow + 1, grd.nlay + 1, 4 };
    // Create the array with the specified shape
    py::array_t<float> zcorn_result(shape);
    auto zcorn_result_ = zcorn_result.mutable_unchecked<4>();

    auto coordsv_ = grd.coordsv.unchecked<3>();

    if (rsurfs.size() != grd.nlay + 1) {
        throw std::invalid_argument("Wrong number of input surfaces vs grid layers");
    }

    for (size_t i = 0; i < grd.ncol + 1; i++) {
        for (size_t j = 0; j < grd.nrow + 1; j++) {
            // get pillar coordinate, just using the the top pillar point as the
            // pillars shall be vertical
            double x = coordsv_(i, j, 0);
            double y = coordsv_(i, j, 1);
            size_t k = 0;
            for (const auto &rsurf : rsurfs) {
                double z = regsurf::get_z_from_xy(rsurf, x, y);
                printf("k and z: %d  %f\n", k, z);
                if (std::isnan(z)) {
                    z = 1000;  // default value; TODO make this better
                }

                // update the z values of the cell corners
                for (size_t l = 0; l < 4; l++) {
                    zcorn_result_(i, j, k, l) = z;
                }
                k++;
            }
        }
    }
    return zcorn_result;
}  // adjust_boxgrid_layer_to_regsurf

}  // namespace xtgeo::grid3d
