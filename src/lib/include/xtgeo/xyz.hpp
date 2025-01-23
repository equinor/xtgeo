#ifndef XTGEO_XYZ_HPP_
#define XTGEO_XYZ_HPP_

#include <pybind11/pybind11.h>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo {
namespace xyz {

inline void
init(py::module &m)
{
    auto m_xyz =
      m.def_submodule("xyz", "Internal functions for operations on point(s).");

    py::class_<Point>(m_xyz, "Point")
      .def(py::init<double, double>())          // Constructor with 2 arguments
      .def(py::init<double, double, double>())  // Constructor with 3 arguments

      .def_readonly("x", &Point::x)
      .def_readonly("y", &Point::y)
      .def_readonly("z", &Point::z);

    py::class_<Polygon>(m_xyz, "Polygon")
      .def(py::init<const std::vector<xtgeo::xyz::Point> &>())
      .def(py::init<const py::array_t<double> &>())
      .def("add_point", &xtgeo::xyz::Polygon::add_point)
      .def("size", &xtgeo::xyz::Polygon::size)
      .def_readonly("xyz", &xtgeo::xyz::Polygon::points);
}

}  // namespace xyz
}  // namespace xtgeo

#endif  // XTGEO_XYZ_HPP_
