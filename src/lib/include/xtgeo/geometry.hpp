#ifndef XTGEO_GEOMETRY_HPP_
#define XTGEO_GEOMETRY_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <vector>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::geometry {

constexpr int TETRAHEDRON_VERTICES[4][6][4] = {
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 5
      { 3, 7, 4, 5 },
      { 0, 4, 7, 5 },
      { 0, 3, 1, 5 },
      // upper left common vertex 6
      { 0, 4, 7, 6 },
      { 3, 7, 4, 6 },
      { 0, 3, 2, 6 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    {
      // upper right common vertex 7
      { 1, 5, 6, 7 },
      { 2, 6, 5, 7 },
      { 1, 2, 3, 7 },
      // lower left common vertex 4
      { 1, 5, 6, 4 },
      { 2, 6, 5, 4 },
      { 1, 2, 0, 4 },
    },

    // Another combination...
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 1
      { 3, 7, 0, 1 },
      { 0, 4, 3, 1 },
      { 4, 7, 5, 1 },
      // upper left common vertex 2
      { 0, 4, 3, 2 },
      { 3, 7, 0, 2 },
      { 4, 7, 6, 2 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    { // upper right common vertex 3
      { 1, 5, 2, 3 },
      { 2, 6, 1, 3 },
      { 5, 6, 7, 3 },
      // lower left common vertex 0
      { 1, 5, 2, 0 },
      { 2, 6, 1, 0 },
      { 5, 6, 4, 0 } }
};

inline double
hexahedron_dz(const grid3d::CellCorners &corners)
{
    // TODO: This does not account for overall zflip ala Petrel or cells that
    // are malformed
    double dzsum = 0.0;
    dzsum += std::abs(corners.upper_sw.z - corners.lower_sw.z);
    dzsum += std::abs(corners.upper_se.z - corners.lower_se.z);
    dzsum += std::abs(corners.upper_nw.z - corners.lower_nw.z);
    dzsum += std::abs(corners.upper_ne.z - corners.lower_ne.z);
    return dzsum / 4.0;
}

inline double
triangle_area(const xyz::Point &p1, const xyz::Point &p2, const xyz::Point &p3)
{
    return 0.5 *
           std::abs(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));
}

double
hexahedron_volume(const grid3d::CellCorners &corners, const int precision);

bool
is_xy_point_in_polygon(const double x, const double y, const xyz::Polygon &polygon);

bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const xyz::Point &p1,
                             const xyz::Point &p2,
                             const xyz::Point &p3,
                             const xyz::Point &p4,
                             const double tolerance = numerics::TOLERANCE);
double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const xyz::Point &p1,
                         const xyz::Point &p2,
                         const xyz::Point &p3,
                         const xyz::Point &p4,
                         const double tolerance = numerics::TOLERANCE);

double
interpolate_z_4p(const double x,
                 const double y,
                 const xyz::Point &p1,
                 const xyz::Point &p2,
                 const xyz::Point &p3,
                 const xyz::Point &p4,
                 const double tolerance = numerics::TOLERANCE);

std::array<double, 8>
find_rect_corners_from_center(const double x,
                              const double y,
                              const double xinc,
                              const double yinc,
                              const double rot);

// functions exposed to Python:
inline void
init(py::module &m)
{
    auto m_geometry = m.def_submodule("geometry", "Internal geometric functions");
    m_geometry.def("hexahedron_volume", &hexahedron_volume,
                   "Estimate the volume of a hexahedron i.e. a cornerpoint cell.");
    m_geometry.def("is_xy_point_in_polygon", &is_xy_point_in_polygon,
                   "Return True if a XY point is inside a polygon seen from above, "
                   "False otherwise.");
    m_geometry.def("is_xy_point_in_quadrilateral", &is_xy_point_in_quadrilateral,
                   "Return True if a XY point is inside a quadrilateral seen from , "
                   "above. False otherwise.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
    m_geometry.def("interpolate_z_4p_regular", &interpolate_z_4p_regular,
                   "Interpolate Z when having 4 corners in a regular XY space, "
                   "typically a regular surface.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
    m_geometry.def("interpolate_z_4p", &interpolate_z_4p,
                   "Interpolate Z when having 4 corners in a non regular XY space, "
                   "like the top of a 3D grid cell.",
                   py::arg("x"), py::arg("y"), py::arg("p1"), py::arg("p2"),
                   py::arg("p3"), py::arg("p4"),
                   py::arg("tolerance") = numerics::TOLERANCE);
}
}  // namespace xtgeo::geometry

#endif  // XTGEO_GEOMETRY_HPP_
