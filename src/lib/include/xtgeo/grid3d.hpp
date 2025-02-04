#ifndef XTGEO_GRID3D_HPP_
#define XTGEO_GRID3D_HPP_
#include <pybind11/pybind11.h>  // should be included first according to pybind11 docs
#include <pybind11/numpy.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

py::array_t<double>
get_cell_volumes(const Grid &grid_cpp, const int precision, const bool asmasked);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_cell_centers(const Grid &grid_cpp, const bool asmasked);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_height_above_ffl(const Grid &grid_cpp,
                     const py::array_t<float> &ffl,
                     const size_t option);

CellCorners
get_cell_corners_from_ijk(const Grid &grid_cpp,
                          const size_t i,
                          const size_t j,
                          const size_t k);

std::vector<double>
get_corners_minmax(CellCorners &get_cell_corners_from_ijk);

bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const CellCorners &corners,
                    int option);

double
get_depth_in_cell(const double x,
                  const double y,
                  const CellCorners &corners,
                  int option);

py::array_t<int8_t>
get_gridprop_value_between_surfaces(const Grid &grd,
                                    const regsurf::RegularSurface &top,
                                    const regsurf::RegularSurface &bot);

inline void
init(py::module &m)
{
    auto m_grid3d =
      m.def_submodule("grid3d", "Internal functions for operations on 3d grids.");

    py::class_<Grid>(m_grid3d, "Grid")
      .def(py::init<const py::object &>(), py::arg("grid"))
      .def_readonly("ncol", &Grid::ncol)
      .def_readonly("nrow", &Grid::nrow)
      .def_readonly("nlay", &Grid::nlay)
      .def_readonly("coordsv", &Grid::coordsv)
      .def_readonly("zcornsv", &Grid::zcornsv)
      .def_readonly("actnumsv", &Grid::actnumsv)

      .def("get_cell_volumes", &get_cell_volumes, "Compute the bulk volume of cell.")

      .def("get_cell_centers", &get_cell_centers,
           "Compute the cells centers coordinates as 3 arrays")
      .def("get_gridprop_value_between_surfaces", &get_gridprop_value_between_surfaces,
           "Make a property that is one if cell center is between two surfaces.")
      .def("get_height_above_ffl", &get_height_above_ffl,
           "Compute the height above a FFL (free fluid level).")
      .def("get_cell_corners_from_ijk", &get_cell_corners_from_ijk,
           "Get a vector containing the corners of a specified IJK cell.")

      ;

    py::class_<CellCorners>(m_grid3d, "CellCorners")
      // a constructor that takes 8 xyz::Point objects
      .def(py::init<xyz::Point, xyz::Point, xyz::Point, xyz::Point, xyz::Point,
                    xyz::Point, xyz::Point, xyz::Point>())
      // a constructor that takes a one-dimensional array of 24 elements
      .def(py::init<const py::array_t<double> &>())

      .def_readonly("upper_sw", &CellCorners::upper_sw)
      .def_readonly("upper_se", &CellCorners::upper_se)
      .def_readonly("upper_nw", &CellCorners::upper_nw)
      .def_readonly("upper_ne", &CellCorners::upper_ne)
      .def_readonly("lower_sw", &CellCorners::lower_sw)
      .def_readonly("lower_se", &CellCorners::lower_se)
      .def_readonly("lower_nw", &CellCorners::lower_nw)
      .def_readonly("lower_ne", &CellCorners::lower_ne)
      .def("to_numpy", &CellCorners::to_numpy);

    m_grid3d.def("arrange_corners", &CellCorners::arrange_corners,
                 "Arrange the corners in a single array for easier access.");

    m_grid3d.def("get_corners_minmax", &get_corners_minmax,
                 "Get a vector containing the minmax of a single corner set");
    m_grid3d.def("is_xy_point_in_cell", &is_xy_point_in_cell,
                 "Determine if a XY point is inside a cell, top or base.");
    m_grid3d.def("get_depth_in_cell", &get_depth_in_cell,
                 "Determine the interpolated cell face Z from XY, top or base.");
}

}  // namespace xtgeo::grid3d

#endif  // XTGEO_GRID3D_HPP_
