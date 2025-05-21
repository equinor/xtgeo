#ifndef XTGEO_XYZ_HPP_
#define XTGEO_XYZ_HPP_

#include <pybind11/pybind11.h>
#include <utility>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo {
namespace xyz {

inline std::pair<Point, Point>
get_bounding_box(const Polygon &polygon)
{
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::min();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::min();
    double min_z = std::numeric_limits<double>::max();
    double max_z = std::numeric_limits<double>::min();

    for (const auto &point : polygon.points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
        min_z = std::min(min_z, point.z);
        max_z = std::max(max_z, point.z);
    }

    return { { min_x, min_y, min_z }, { max_x, max_y, max_z } };
}

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

    // Bind PointSet as an alias for Polygon
    m_xyz.attr("PointSet") = m_xyz.attr("Polygon");

    m_xyz.def("get_bounding_box", &get_bounding_box, py::arg("polygon"),
              "Get the bounding box of a polygon");
}

}  // namespace xyz
}  // namespace xtgeo

#endif  // XTGEO_XYZ_HPP_
