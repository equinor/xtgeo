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
cube_stats_along_z(const Cube &cube_cpp);

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

      .def("cube_stats_along_z", &cube_stats_along_z,
           "Compute various statistics for cube along the Z axis, returning maps.")

      ;
}

}  // namespace xtgeo::cube

#endif  // XTGEO_CUBE_HPP_
