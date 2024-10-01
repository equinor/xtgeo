#ifndef XTGEO_GRID3D_HPP_
#define XTGEO_GRID3D_HPP_
#include <pybind11/pybind11.h>  // should be included first according to pybind11 docs
#include <pybind11/numpy.h>
#include <cstddef>
#include <optional>
#include <tuple>

namespace py = pybind11;

namespace xtgeo::grid3d {

py::array_t<double>
grid_cell_volumes(const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const py::array_t<int> &actnumsv,
                  const int precision,
                  const bool asmasked);
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
grid_height_above_ffl(const size_t ncol,
                      const size_t nrow,
                      const size_t nlay,
                      const py::array_t<double> &coordsv,
                      const py::array_t<float> &zcornsv,
                      const py::array_t<int> &actnumsv,
                      const py::array_t<float> &ffl,
                      const size_t option);
std::vector<double>
cell_corners(const size_t i,
             const size_t j,
             const size_t k,
             const size_t ncol,
             const size_t nrow,
             const size_t nlay,
             const py::array_t<double> &coordsv,
             const py::array_t<float> &zcornsv);

std::vector<double>
get_corners_minmax(std::vector<double> &corners);

bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const size_t i,
                    const size_t j,
                    const size_t k,
                    const size_t ncol,
                    const size_t nrow,
                    const size_t nlay,
                    const py::array_t<double> &coordsv,
                    const py::array_t<float> &zcornsv,
                    const int option);

double
get_depth_in_cell(const double x,
                  const double y,
                  const size_t i,
                  const size_t j,
                  const size_t k,
                  const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const int option);

inline void
init(py::module &m)
{
    auto m_grid3d =
      m.def_submodule("grid3d", "Internal functions for operations on 3d grids.");

    m_grid3d.def("grid_cell_volumes", &grid_cell_volumes,
                 "Compute the bulk volume of cell.");
    m_grid3d.def("grid_height_above_ffl", &grid_height_above_ffl,
                 "Compute the height above a FFL (free fluid level).");
    m_grid3d.def("cell_corners", &cell_corners,
                 "Get a vector containing the corners of a cell.");
    m_grid3d.def("get_corners_minmax", &get_corners_minmax,
                 "Get a vector containing the minmax of a single corner set");
    m_grid3d.def("is_xy_point_in_cell", &is_xy_point_in_cell,
                 "Determine if a XY point is inside a cell, top or base.");
    m_grid3d.def("get_depth_in_cell", &get_depth_in_cell,
                 "Determine the interpolated cell face Z from XY, top or base.");
}

}  // namespace xtgeo::grid3d

#endif  // XTGEO_GRID3D_HPP_
