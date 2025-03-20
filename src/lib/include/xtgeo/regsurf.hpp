#ifndef XTGEO_REGSURF_HPP_
#define XTGEO_REGSURF_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <stdexcept>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

std::tuple<xyz::Point, xyz::Point, xyz::Point, xyz::Point>
get_outer_corners(const RegularSurface &regsurf);

xyz::Point
get_xy_from_ij(const RegularSurface &rs,
               const size_t i,
               const size_t j,
               const int yflip = 1);

double
get_z_from_xy(const RegularSurface &rs,
              const double x,
              const double y,
              const double tolerance = numerics::TOLERANCE);

std::tuple<int, int, int, int>
find_cell_range(const RegularSurface &rs,
                const double xmin,
                const double xmax,
                const double ymin,
                const double ymax,
                const int expand);

std::tuple<py::array_t<int>,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
sample_grid3d_layer(const RegularSurface &rs_cpp,
                    const grid3d::Grid &grid_cpp,
                    const size_t klayer,
                    const int index_position,
                    const int num_threads = -1);

inline void
init(py::module &m)
{
    auto m_regsurf = m.def_submodule(
      "regsurf", "Internal functions for operations on regular surface.");

    py::class_<RegularSurface>(m_regsurf, "RegularSurface")
      .def(py::init<const py::object &>(), py::arg("rs"))  // Constructor
      .def_readonly("ncol", &RegularSurface::ncol)
      .def_readonly("nrow", &RegularSurface::nrow)
      .def_readonly("xori", &RegularSurface::xori)
      .def_readonly("yori", &RegularSurface::yori)
      .def_readonly("xinc", &RegularSurface::xinc)
      .def_readonly("yinc", &RegularSurface::yinc)
      .def_readonly("rotation", &RegularSurface::rotation)
      .def_readonly("values", &RegularSurface::values)
      .def_readonly("mask", &RegularSurface::mask)

      // need py::arg(...) when keys with default values are used in python:
      .def("get_outer_corners", &get_outer_corners,
           "Get the outer corners of a regular surface.")
      .def("find_cell_range", &find_cell_range,
           "Find the range of regular surface 2D nodes within a box.")
      .def("get_xy_from_ij", &get_xy_from_ij,
           "Get the XY coordinates from the 2D grid nodes indices.", py::arg("i"),
           py::arg("j"), py::arg("yflip") = 1)
      .def("get_z_from_xy", &get_z_from_xy,
           "Get the Z value in a regsurf from a x y point.", py::arg("x"), py::arg("y"),
           py::arg("tolerance") = numerics::TOLERANCE)
      .def("sample_grid3d_layer", &sample_grid3d_layer,
           "Sample values for regular surface from a 3D grid.", py::arg("grid_cpp"),
           py::arg("klayer"), py::arg("index_position"), py::arg("num_threads") = -1);
}

}  // namespace xtgeo::regsurf

#endif  // XTGEO_REGSURF_HPP_
