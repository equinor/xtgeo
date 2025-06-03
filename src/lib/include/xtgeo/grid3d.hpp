#ifndef XTGEO_GRID3D_HPP_
#define XTGEO_GRID3D_HPP_
#include <pybind11/pybind11.h>  // should be included first according to pybind11 docs
#include <pybind11/numpy.h>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/regsurf.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

// =====================================================================================
// OPERATIONS ON INDIVIDUAL CELLS
// =====================================================================================

CellCorners
get_cell_corners_from_ijk(const Grid &grid_cpp,
                          const size_t i,
                          const size_t j,
                          const size_t k);

std::vector<double>
get_corners_minmax(const CellCorners &corners);

std::tuple<xyz::Point, xyz::Point>
get_cell_bounding_box(const CellCorners &corners);

bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const CellCorners &corners,
                    int option);

bool
is_point_in_cell(const xyz::Point &point,
                 const CellCorners &corners,
                 geometry::PointInHexahedronMethod method =
                   geometry::PointInHexahedronMethod::Optimized);

double
get_depth_in_cell(const double x,
                  const double y,
                  const CellCorners &corners,
                  int option);

bool
is_cell_non_convex(const CellCorners &corners);

bool
is_cell_distorted(const CellCorners &corners);

// =====================================================================================
// OPERATIONS ON MULTIPLE CELLS/GRID
// =====================================================================================

py::array_t<double>
get_cell_volumes(const Grid &grid_cpp,
                 geometry::HexVolumePrecision precision,
                 const bool asmasked);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_cell_centers(const Grid &grid_cpp, const bool asmasked = false);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_height_above_ffl(const Grid &grid_cpp,
                     const py::array_t<float> &ffl,
                     const size_t option);

py::array_t<int8_t>
get_gridprop_value_between_surfaces(const Grid &grd,
                                    const regsurf::RegularSurface &top,
                                    const regsurf::RegularSurface &bot);

std::tuple<py::array_t<double>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
convert_xtgeo_to_rmsapi(const Grid &grd);

void
process_edges_rmsapi(py::array_t<float> zcornsv);

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
create_grid_from_cube(const cube::Cube &cube,
                      const bool use_cell_center = false,
                      const int flip = 1);

std::tuple<pybind11::array_t<float>, pybind11::array_t<int>>
adjust_boxgrid_layers_from_regsurfs(Grid &grd,
                                    const std::vector<regsurf::RegularSurface> &rsurfs,
                                    const double tolerance = numerics::TOLERANCE);

std::tuple<py::array_t<float>, py::array_t<int8_t>>
refine_vertically(const Grid &grid_cpp, const py::array_t<uint8_t> refine_layer);
Grid
extract_onelayer_grid(const Grid &original_grid);

std::tuple<xyz::Point, xyz::Point>
get_bounding_box(const Grid &grid);

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
get_indices_from_pointset(const Grid &grid,
                          const xyz::PointSet &points,
                          const Grid &one_grid,
                          const regsurf::RegularSurface &top_i,
                          const regsurf::RegularSurface &top_j,
                          const regsurf::RegularSurface &base_i,
                          const regsurf::RegularSurface &base_j,
                          const bool active_only);

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_columns(const Grid &grid_cpp, const py::array_t<uint8_t> refinement);

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_rows(const Grid &grid_cpp, const py::array_t<uint8_t> refinement);
// =====================================================================================
// PYTHON BINDINGS, IF NEEDED
// =====================================================================================

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
      .def("convert_xtgeo_to_rmsapi", &convert_xtgeo_to_rmsapi,
           "Convert XTGeo grid to RMSAPI grid layout (for storing grid in RMS)")
      .def("adjust_boxgrid_layers_from_regsurfs", &adjust_boxgrid_layers_from_regsurfs,
           "Adjust layers in a boxgrid given a list of regular surfaces.",
           py::arg("rsurfs"), py::arg("tolerance") = numerics::TOLERANCE)
      .def("extract_onelayer_grid", &extract_onelayer_grid, "Get a a onelayer grid")
      .def("get_bounding_box", &get_bounding_box, "Get bounding box of full grid")
      .def("get_indices_from_pointset", &get_indices_from_pointset,
           "Get the indices of a point set in the grid")
      .def("refine_vertically", &refine_vertically, "Refine vertically, proportionally")
      .def("refine_columns", &refine_columns, "Refine per column proportionally")
      .def("refine_rows", &refine_rows, "Refine per row proportionally");

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
    m_grid3d.def("get_cell_bounding_box", &get_cell_bounding_box,
                 "Get the bounding box for a cell");
    m_grid3d.def("is_cell_non_convex", &is_cell_non_convex,
                 "Check if a cell is non-convex");
    m_grid3d.def("is_cell_distorted", &is_cell_distorted,
                 "Check if a cell is (highly) distorted");
    m_grid3d.def("is_xy_point_in_cell", &is_xy_point_in_cell,
                 "Determine if a XY point is inside a cell, top or base.");
    m_grid3d.def("is_point_in_cell", &is_point_in_cell,
                 "Determine if a point XYZ is inside a cell, in 3D", py::arg("point"),
                 py::arg("corners"),
                 py::arg("method") = geometry::PointInHexahedronMethod::Optimized);
    m_grid3d.def("get_depth_in_cell", &get_depth_in_cell,
                 "Determine the interpolated cell face Z from XY, top or base.");
    m_grid3d.def("process_edges_rmsapi", &process_edges_rmsapi, "Edge processing...");
    m_grid3d.def("create_grid_from_cube", &create_grid_from_cube,
                 "Create a 3D grid from a cube specification.", py::arg("cube"),
                 py::arg("use_cell_center") = false, py::arg("flip") = 1);
}

}  // namespace xtgeo::grid3d

#endif  // XTGEO_GRID3D_HPP_
