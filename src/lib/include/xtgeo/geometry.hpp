#ifndef XTGEO_GEOMETRY_HPP_
#define XTGEO_GEOMETRY_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <array>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846  // seems like Windows does not define M_PI i cmath
#endif

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
hexahedron_dz(const std::vector<double> &corners)
{
    // TODO: This does not account for overall zflip ala Petrel or cells that
    // are malformed
    double dzsum = 0.0;
    for (auto i = 0; i < 4; i++) {
        dzsum += std::abs(corners[3 * i + 2] - corners[3 * i + 2 + 12]);
    }
    return dzsum / 4.0;
}

inline double
triangle_area(const std::array<double, 2> &p1,
              const std::array<double, 2> &p2,
              const std::array<double, 2> &p3)
{
    return 0.5 * std::abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) +
                          p3[0] * (p1[1] - p2[1]));
}

double
hexahedron_volume(const std::vector<double> &corners, const int precision);

bool
is_xy_point_in_polygon(const double x,
                       const double y,
                       const std::vector<std::array<double, 2>> &polygon);

bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const std::array<double, 3> &p1,
                             const std::array<double, 3> &p2,
                             const std::array<double, 3> &p3,
                             const std::array<double, 3> &p4);

double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const std::array<double, 3> &p1,
                         const std::array<double, 3> &p2,
                         const std::array<double, 3> &p3,
                         const std::array<double, 3> &p4);

double
interpolate_z_4p(const double x,
                 const double y,
                 const std::array<double, 3> &p1,
                 const std::array<double, 3> &p2,
                 const std::array<double, 3> &p3,
                 const std::array<double, 3> &p4);

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
                   "above. False otherwise.");
    m_geometry.def("interpolate_z_4p_regular", &interpolate_z_4p_regular,
                   "Interpolate Z when having 4 corners in a regular XY space, "
                   "typically a regular surface.");
    m_geometry.def("interpolate_z_4p", &interpolate_z_4p,
                   "Interpolate Z when having 4 corners in a non regular XY space, "
                   "like the top of a 3D grid cell.");
}
}  // namespace xtgeo::geometry

#endif  // XTGEO_GEOMETRY_HPP_
