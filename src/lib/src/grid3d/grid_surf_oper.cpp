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
 * @param ncol Grid dimensions ncol/nx
 * @param nrow Grid dimensions nrow/ny
 * @param nlay Grid dimensions nlay/nz
 * @param xmid Grid X midpoints
 * @param ymid Grid Y midpoints
 * @param zmid Grid Z midpoints
 * @param top_ncol Top surface ncol
 * @param top_nrow Top surface nrow
 * @param top_xori Top surface x origin
 * @param top_yori Top surface y origin
 * @param top_xinc Top surface x increment
 * @param top_yinc Top surface y increment
 * @param top_rotation Top surface rotation
 * @param top_values Top surface values
 * @param bot_ncol Bottom surface ncol
 * @param bot_nrow Bottom surface nrow
 * @param bot_xori Bottom surface x origin
 * @param bot_yori Bottom surface y origin
 * @param bot_xinc Bottom surface x increment
 * @param bot_yinc Bottom surface y increment
 * @param bot_rotation Bottom surface rotation
 * @param bot_values Bottom surface values
 * @return An int array containing 1 if the cell is between the surfaces, 0 otherwise
 */
py::array_t<int8_t>
grid_assign_value_between_surfaces(const size_t ncol,
                                   const size_t nrow,
                                   const size_t nlay,
                                   const py::array_t<double> &xmid,
                                   const py::array_t<double> &ymid,
                                   const py::array_t<double> &zmid,
                                   const size_t top_ncol,
                                   const size_t top_nrow,
                                   const double top_xori,
                                   const double top_yori,
                                   const double top_xinc,
                                   const double top_yinc,
                                   const double top_rotation,
                                   const py::array_t<double> &top_values,
                                   const size_t bot_ncol,
                                   const size_t bot_nrow,
                                   const double bot_xori,
                                   const double bot_yori,
                                   const double bot_xinc,
                                   const double bot_yinc,
                                   const double bot_rotation,
                                   const py::array_t<double> &bot_values)
{
    pybind11::array_t<int8_t> result({ ncol, nrow, nlay });
    auto result_ = result.mutable_unchecked<3>();

    // Access the np array without bounds checking for optimizing speed
    auto xmid_ = xmid.unchecked<3>();
    auto ymid_ = ymid.unchecked<3>();
    auto zmid_ = zmid.unchecked<3>();

    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
            for (size_t k = 0; k < nlay; k++) {
                // for every cell, project the center to the top and bottom surfaces
                double xm = xmid_(i, j, k);
                double ym = ymid_(i, j, k);
                double zm = zmid_(i, j, k);

                // check if zm is NaN which can occur for inactive cells
                if (std ::isnan(zm)) {
                    result_(i, j, k) = 0;
                    continue;
                }

                double top_z =
                  regsurf::get_z_from_xy(xm, ym, top_xori, top_yori, top_xinc, top_yinc,
                                         top_ncol, top_nrow, top_rotation, top_values);
                double bot_z =
                  regsurf::get_z_from_xy(xm, ym, bot_xori, bot_yori, bot_xinc, bot_yinc,
                                         bot_ncol, bot_nrow, bot_rotation, bot_values);

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
}  // grid_assign_value_between_surfaces

}  // namespace xtgeo::grid3d
