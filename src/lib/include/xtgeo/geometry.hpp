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

/**
 * @brief Enumeration of methods for determining if a point is inside a hexahedron.
 */
enum class PointInHexahedronMethod
{
    RayCasting,
    Tetrahedrons,
    UsingPlanes,
    Legacy,
    Isoparametric,
    Optimized  // combining appropriate methods
};

// volume precision enum
enum class HexVolumePrecision
{
    P1 = 1,
    P2 = 2,
    P4 = 4
};

// Cell-Polygon relationship enum
enum class CellPolygonRelation
{
    Inside,       // All cell corners are inside the polygon
    Outside,      // All cell corners are outside the polygon
    Intersecting  // Some corners inside, some outside
};

// =====================================================================================
// TETRAHEDRONS
// =====================================================================================

/**
 * @brief Compute the signed volume of a tetrahedron defined by four points.
 *
 * This function calculates the signed volume of a tetrahedron defined by four points in
 * 3D space. The signed volume is positive if the points are ordered counter-clockwise
 * and negative if they are ordered clockwise.
 *
 * @param a The first vertex of the tetrahedron.
 * @param b The second vertex of the tetrahedron.
 * @param c The third vertex of the tetrahedron.
 * @param d The fourth vertex of the tetrahedron.
 * @return The signed volume of the tetrahedron.
 */
double
signed_tetrahedron_volume(const xyz::Point &a,
                          const xyz::Point &b,
                          const xyz::Point &c,
                          const xyz::Point &d);

/**
 * @brief Determine if a point is inside a tetrahedron defined by four vertices.
 *
 * @param point The point to check.
 * @param v0 The first vertex of the tetrahedron.
 * @param v1 The second vertex of the tetrahedron.
 * @param v2 The third vertex of the tetrahedron.
 * @param v3 The fourth vertex of the tetrahedron.
 * @return int 1 if the point is inside the tetrahedron, 0 otherwise.
 */
int
is_point_in_tetrahedron(const xyz::Point &point,
                        const xyz::Point &v0,
                        const xyz::Point &v1,
                        const xyz::Point &v2,
                        const xyz::Point &v3);

/**
 * @brief Determine if a point is inside a tetrahedron using the legacy method.
 *
 * This function uses the legacy method (from SWIG/C base) to check if a point is inside
 * a tetrahedron defined by four vertices. The legacy method is kept for reference for a
 * while
 *
 * @param point The point to check.
 * @param v0 The first vertex of the tetrahedron.
 * @param v1 The second vertex of the tetrahedron.
 * @param v2 The third vertex of the tetrahedron.
 * @param v3 The fourth vertex of the tetrahedron.
 * @return bool True if the point is inside the tetrahedron, false otherwise.
 */
bool
is_point_in_tetrahedron_legacy(const xyz::Point &point,
                               const xyz::Point &v0,
                               const xyz::Point &v1,
                               const xyz::Point &v2,
                               const xyz::Point &v3);
// =====================================================================================
// POLYGONS (TRIANGLES, QUADRILATERALS, ...)
// =====================================================================================

/**
 * @brief Compute the area of a triangle defined by three points.
 *
 * @param p1 The first vertex of the triangle.
 * @param p2 The second vertex of the triangle.
 * @param p3 The third vertex of the triangle.
 * @return The area of the triangle.
 */
inline double
triangle_area(const xyz::Point &p1, const xyz::Point &p2, const xyz::Point &p3)
{
    return 0.5 * std::abs(p1.x() * (p2.y() - p3.y()) + p2.x() * (p3.y() - p1.y()) +
                          p3.x() * (p1.y() - p2.y()));
}

/**
 * @brief Compute the area of a quadrilateral defined by four points.
 *
 * The quadrilateral is divided into two triangles to calculate the area.
 *
 * @param p1 The first vertex of the quadrilateral.
 * @param p2 The second vertex of the quadrilateral.
 * @param p3 The third vertex of the quadrilateral.
 * @param p4 The fourth vertex of the quadrilateral.
 * @return The area of the quadrilateral.
 */
inline double
quadrilateral_area(const xyz::Point &p1,
                   const xyz::Point &p2,
                   const xyz::Point &p3,
                   const xyz::Point &p4)
{
    // Note points are in clockwise order or counter-clockwise order
    return triangle_area(p1, p2, p3) + triangle_area(p1, p3, p4);
}

/**
 * @brief Determine if a point is inside, or on the boundary of a polygon defined by its
 * vertices.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param polygon The polygon to check.
 * @return bool True if the point is inside the polygon, false otherwise.
 */
bool
is_xy_point_in_polygon(const double x, const double y, const xyz::Polygon &polygon);

/**
 * @brief Determine the relationship between a cell (defined by CellCorners) and a
 * polygon.
 *
 * This function checks all 8 corners of the cell against the polygon boundary in the
 * XY plane (bird's eye view). It returns:
 * - Inside: All corners are inside the polygon
 * - Outside: All corners are outside the polygon
 * - Intersecting: Some corners are inside, some are outside (cell crosses boundary)
 *
 * @param cell The cell corners to check.
 * @param polygon The polygon boundary to check against.
 * @return CellPolygonRelation The relationship between the cell and polygon.
 */
CellPolygonRelation
cell_polygon_relation(const grid3d::CellCorners &cell, const xyz::Polygon &polygon);

/**
 * @brief Overload with precomputed polygon bounding box.
 *
 * @param cell The cell corners to check.
 * @param polygon The polygon boundary to check against.
 * @param poly_bbox Precomputed polygon bounding box [min_x, max_x, min_y, max_y].
 * @return CellPolygonRelation The relationship between the cell and polygon.
 */
CellPolygonRelation
cell_polygon_relation(const grid3d::CellCorners &cell,
                      const xyz::Polygon &polygon,
                      const std::array<double, 4> &poly_bbox);

/**
 * @brief Compute the axis-aligned bounding box of a polygon projected onto the X–Y
 * plane.
 *
 * @param polygon The polygon whose vertices are used to compute the bounding box.
 * @return std::array<double, 4> with the bounds in the form { min_x, max_x, min_y,
 * max_y }.
 */
std::array<double, 4>
get_polygon_xy_bbox(const xyz::Polygon &polygon);

/**
 * @brief Determine if a XY point is inside a quadrilateral defined by its vertices.
 *
 * This is optimized for quadrilaterals.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param p1 The first vertex of the quadrilateral.
 * @param p2 The second vertex of the quadrilateral.
 * @param p3 The third vertex of the quadrilateral.
 * @param p4 The fourth vertex of the quadrilateral.
 * @param tolerance The tolerance for checking if the point is on the boundary.
 * @return bool True if the point is inside or on the boundary of the quadrilateral,
 * false otherwise.
 */
bool
is_xy_point_in_quadrilateral(const double x,
                             const double y,
                             const xyz::Point &p1,
                             const xyz::Point &p2,
                             const xyz::Point &p3,
                             const xyz::Point &p4,
                             const double tolerance = numerics::TOLERANCE);
/**
 * @brief Interpolate the Z value at a point within a regular quadrilateral defined by
 * its vertices, typically a regular map cell.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param p1 The first vertex of the quadrilateral.
 * @param p2 The second vertex of the quadrilateral.
 * @param p3 The third vertex of the quadrilateral.
 * @param p4 The fourth vertex of the quadrilateral.
 * @param tolerance The tolerance for checking if the point is on the boundary.
 * @return double The interpolated Z value at the point.
 */
double
interpolate_z_4p_regular(const double x,
                         const double y,
                         const xyz::Point &p1,
                         const xyz::Point &p2,
                         const xyz::Point &p3,
                         const xyz::Point &p4,
                         const double tolerance = numerics::TOLERANCE);

/**
 * @brief Interpolate the Z value at a point within a quadrilateral defined by its
 * vertices, typically a non-regular map or grid top/base face cell.
 *
 * @param x The x-coordinate of the point.
 * @param y The y-coordinate of the point.
 * @param p1 The first vertex of the quadrilateral.
 * @param p2 The second vertex of the quadrilateral.
 * @param p3 The third vertex of the quadrilateral.
 * @param p4 The fourth vertex of the quadrilateral.
 * @param tolerance The tolerance for checking if the point is on the boundary.
 * @return double The interpolated Z value at the point.
 */
double
interpolate_z_4p(const double x,
                 const double y,
                 const xyz::Point &p1,
                 const xyz::Point &p2,
                 const xyz::Point &p3,
                 const xyz::Point &p4,
                 const double tolerance = numerics::TOLERANCE);

/**
 * @brief Find the corners of a rectangle given its center, increments, and rotation.
 *
 * @param x The x-coordinate of the center.
 * @param y The y-coordinate of the center.
 * @param xinc The increment in the x-direction.
 * @param yinc The increment in the y-direction.
 * @param rot The rotation angle in radians.
 * @return std::array<double, 8> An array containing the coordinates of the rectangle
 * corners (x1, y1, x2, y2, x3, y3, x4, y4).
 */
std::array<double, 8>
find_rect_corners_from_center(const double x,
                              const double y,
                              const double xinc,
                              const double yinc,
                              const double rot);

// =====================================================================================
// HEXAHEDRON
// In general a hexahderson is a cell with 8 corners, here defined by the
// HexahedronCorners structure. The corners are defined in a specific order, which is
// important for the calculations. Not that Z increases upwards (right handed system)
// The order is:
// 1. upper_sw
// 2. upper_se
// 3. upper_ne
// 4. upper_nw
// 5. lower_sw
// 6. lower_se
// 7. lower_ne
// 8. lower_nw
// =====================================================================================

/**
 * @brief Find the DZ (average vertical distance) of a hexahedron defined by its
 * corners.
 *
 * The corners are defined by the HexahedronCorners structure, which contains the
 * coordinates of the eight corners of the hexahedron.
 */
inline double
hexahedron_dz(const HexahedronCorners &corners)
{
    double dzsum = 0.0;
    dzsum += std::abs(corners.upper_sw.z() - corners.lower_sw.z());
    dzsum += std::abs(corners.upper_se.z() - corners.lower_se.z());
    dzsum += std::abs(corners.upper_nw.z() - corners.lower_nw.z());
    dzsum += std::abs(corners.upper_ne.z() - corners.lower_ne.z());
    return dzsum / 4.0;
}

// /**
//  * @brief Find the volume of a hexahedron defined by its corners.
//  *
//  * The is a legacy function (from C/SWIG base) that uses the absolute tetrahedron
//  volume
//  * method to calculate the volume of a hexahedron defined by its corners. Hence it is
//  * not precise for concave cells.
//  * The corners are defined by the HexahedronCorners structure, which contains the
//  * coordinates of the eight corners of the hexahedron.
//  */
// double
// hexahedron_volume_legacy(const HexahedronCorners &corners, const int precision);

// // overload for CellCorners
// double
// hexahedron_volume_legacy(const grid3d::CellCorners &corners, const int precision);

/**
 * @brief Find the volume of a hexahedron defined by its corners.
 *
 * This function uses the signed tetrahedron volume method to calculate the volume of a
 * hexahedron defined by its corners. The volume is calculated by
 * dividing the hexahedron into N tetrahedrons (and M combinations) and summing their
 * signed volumes. The function is designed to handle both convex and concave hexahedra.
 *
 * @param corners The corners of the hexahedron defined by the HexahedronCorners
 */
double
hexahedron_volume(const HexahedronCorners &corners, HexVolumePrecision precision);

// overload for CellCorners
double
hexahedron_volume(const grid3d::CellCorners &corners, HexVolumePrecision precision);

bool
is_point_in_hexahedron_raycasting(const xyz::Point &point,
                                  const HexahedronCorners &corners);
bool
is_point_in_hexahedron_usingplanes(const xyz::Point &point,
                                   const HexahedronCorners &corners);
int
is_point_in_hexahedron_tetrahedrons_legacy(const xyz::Point &point,
                                           const HexahedronCorners &corners);

bool
is_point_in_hexahedron_tetrahedrons_by_scheme(const xyz::Point &point,
                                              const HexahedronCorners &corners);

int
is_point_in_hexahedron_isoparametric(const xyz::Point &point,
                                     const HexahedronCorners &corners);

bool
is_hexahedron_non_convex(const HexahedronCorners &corners);

bool
is_hexahedron_severely_distorted(const xtgeo::geometry::HexahedronCorners &corners);

bool
is_hexahedron_thin(const HexahedronCorners &corners, const double threshold = 0.05);

bool
is_hexahedron_concave_projected(const HexahedronCorners &corners);

std::vector<double>
get_hexahedron_minmax(const HexahedronCorners &corners);

std::tuple<xyz::Point, xyz::Point>
get_hexahedron_bounding_box(const HexahedronCorners &corners);

bool
is_point_in_hexahedron_bounding_box(const xyz::Point &point,
                                    const HexahedronCorners &hexahedron_corners);
bool
is_point_in_hexahedron_bounding_box_minmax_pt(const xyz::Point &point,
                                              const xyz::Point &min_pt,
                                              const xyz::Point &max_pt);

// =====================================================================================
// PYTHON BINDINGS
// =====================================================================================
inline void
init(py::module &m)
{
    auto m_geometry = m.def_submodule("geometry", "Internal geometric functions");

    py::enum_<PointInHexahedronMethod>(m_geometry, "PointInHexahedronMethod")
      .value("RayCasting", PointInHexahedronMethod::RayCasting)
      .value("Tetrahedrons", PointInHexahedronMethod::Tetrahedrons)
      .value("UsingPlanes", PointInHexahedronMethod::UsingPlanes)
      .value("Legacy", PointInHexahedronMethod::Legacy)
      .value("Isoparametric", PointInHexahedronMethod::Isoparametric)
      .value("Optimized", PointInHexahedronMethod::Optimized)
      .export_values();  // Makes the enum values accessible as attributes of the enum

    py::enum_<HexVolumePrecision>(m_geometry, "HexVolumePrecision")
      .value("P1", HexVolumePrecision::P1)
      .value("P2", HexVolumePrecision::P2)
      .value("P4", HexVolumePrecision::P4)
      .export_values();  // Makes the enum values accessible as attributes of the enum

    py::class_<HexahedronCorners>(m_geometry, "HexahedronCorners")
      // a constructor that takes 8 xyz::Point objects
      .def(py::init<xyz::Point, xyz::Point, xyz::Point, xyz::Point, xyz::Point,
                    xyz::Point, xyz::Point, xyz::Point>())
      // a constructor that takes a one-dimensional array of 24 elements
      // Note that HexahedronCorners differs from CellCorners (slightly)
      .def(py::init<const py::array_t<double> &>())

      .def_readonly("upper_sw", &HexahedronCorners::upper_sw)
      .def_readonly("upper_se", &HexahedronCorners::upper_se)
      .def_readonly("upper_ne", &HexahedronCorners::upper_ne)
      .def_readonly("upper_nw", &HexahedronCorners::upper_nw)
      .def_readonly("lower_sw", &HexahedronCorners::lower_sw)
      .def_readonly("lower_se", &HexahedronCorners::lower_se)
      .def_readonly("lower_ne", &HexahedronCorners::lower_ne)
      .def_readonly("lower_nw", &HexahedronCorners::lower_nw)

      ;

    m_geometry.def(
      "hexahedron_volume",
      [](const grid3d::CellCorners &corners,
         HexVolumePrecision precision) {  // overload for CellCorners
          return hexahedron_volume(corners, precision);
      },
      "Estimate the volume of a hexahedron i.e. a cornerpoint cell using "
      "CornerPoints.");

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
    m_geometry.def("is_hexahedron_non_convex", &is_hexahedron_non_convex,
                   "Determine if a hexahedron is non-convex");
    m_geometry.def("is_hexahedron_severely_distorted",
                   &is_hexahedron_severely_distorted,
                   "Determine if a hexahedron is severely distorted");
    m_geometry.def("is_hexahedron_thin", &is_hexahedron_thin,
                   "Determine if a hexahedron is thin", py::arg("corners"),
                   py::arg("threshold") = 0.05);
    m_geometry.def("is_hexahedron_concave_projected", &is_hexahedron_concave_projected,
                   "Determine if a hexahedron is concave projected");
}
}  // namespace xtgeo::geometry

#endif  // XTGEO_GEOMETRY_HPP_
