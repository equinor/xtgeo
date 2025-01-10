#ifndef XTGEO_REGSURF_HPP_
#define XTGEO_REGSURF_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <xtgeo/point.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

std::tuple<Point, Point, Point, Point>
get_outer_corners(const double xori,
                  const double yori,
                  const double xinc,
                  const double yinc,
                  const size_t ncol,
                  const size_t nrow,
                  const double A_deg);

Point
get_xy_from_ij(const size_t i,
               const size_t j,
               const double xori,
               const double yori,
               const double xinc,
               const double yinc,
               const size_t ncol,
               const size_t nrow,
               const double angle_deg);

double
get_z_from_xy(const double x,
              const double y,
              const double xori,
              const double yori,
              const double xinc,
              const double yinc,
              const size_t ncol,
              const size_t nrow,
              const double angle_deg,
              const py::array_t<double> &values);

std::tuple<int, int, int, int>
find_cell_range(const double xmin,
                const double xmax,
                const double ymin,
                const double ymax,
                const double xori,
                const double yori,
                const double xinc,
                const double yinc,
                const double rotation_degrees,
                const size_t ncol,
                const size_t nrow,
                const int expand);

std::tuple<py::array_t<int>,
           py::array_t<int>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
sample_grid3d_layer(const size_t ncol,
                    const size_t nrow,
                    const double xori,
                    const double yori,
                    const double xinc,
                    const double yinc,
                    const double rotation,
                    const size_t ncolgrid3d,
                    const size_t nrowgrid3d,
                    const size_t nlaygrid3d,
                    const py::array_t<double> &coordsv,
                    const py::array_t<float> &zcornsv,
                    const py::array_t<int> &actnumsv,
                    const size_t klayer,
                    const int index_position,
                    const int num_threads);
inline void
init(py::module &m)
{
    auto m_regsurf = m.def_submodule(
      "regsurf", "Internal functions for operations on regular surface.");

    py::class_<Point>(m_regsurf, "Point")
      .def(py::init<>())
      .def_readwrite("x", &Point::x)
      .def_readwrite("y", &Point::y);

    m_regsurf.def("get_outer_corners", &get_outer_corners,
                  "Get the outer corners of a regular surface.");
    m_regsurf.def("get_xy_from_ij", &get_xy_from_ij,
                  "Get the XY coordinates from the grid indices.");
    m_regsurf.def("get_z_from_xy", &get_z_from_xy,
                  "Get the Z value in a regsurf from a x y point.");
    m_regsurf.def("find_cell_range", &find_cell_range,
                  "Find the range of regular surface 2D nodes within a box.");
    m_regsurf.def("sample_grid3d_layer", &sample_grid3d_layer,
                  "Sample values for regular surface from a 3D grid.");
}

}  // namespace xtgeo::regsurf

#endif  // XTGEO_REGSURF_HPP_
