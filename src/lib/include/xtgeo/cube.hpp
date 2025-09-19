#ifndef XTGEO_CUBE_HPP_
#define XTGEO_CUBE_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <stdexcept>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::cube {

std::unordered_map<std::string, py::array_t<double>>
cube_stats_along_z(const Cube &cube_cpp,
                   const py::array_t<double> &upper_surface,
                   const py::array_t<double> &lower_surface,
                   const py::array_t<double> &depth_array,
                   int ndiv = 1,
                   const std::string &interpolation = "linear",
                   double min_thickness = 0.0,
                   int min_index = 0,
                   int max_index = -1);

inline void
init(py::module &m)
{
    auto m_cube =
      m.def_submodule("cube", "Internal functions for operations on 3d cubes.");

    py::class_<Cube>(m_cube, "Cube")
      .def(py::init<const py::object &>(), py::arg("cube"))
      .def_readonly("ncol", &Cube::ncol)
      .def_readonly("nrow", &Cube::nrow)
      .def_readonly("nlay", &Cube::nlay)
      .def_readonly("xori", &Cube::xori)
      .def_readonly("yori", &Cube::yori)
      .def_readonly("zori", &Cube::zori)
      .def_readonly("xinc", &Cube::xinc)
      .def_readonly("yinc", &Cube::yinc)
      .def_readonly("zinc", &Cube::zinc)
      .def_readonly("rotation", &Cube::rotation)
      .def_readonly("values", &Cube::values)
      .def_readonly("traceidcodes", &Cube::traceidcodes)

      .def("cube_stats_along_z", &cube_stats_along_z,
           "Compute various statistics for cube along the Z axis, returning "
           "maps.")

      ;
}

}  // namespace xtgeo::cube

#endif  // XTGEO_CUBE_HPP_
