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

/**
 * @brief Enumeration of option for determining cell height above FFL
 */
enum class HeightAboveFFLOption
{
    CellCenter = 1,
    CellCorners = 2,
    TruncatedCellCorners = 3
};

/**
 * @brief Direction of adjacency between two grid cells.
 *
 * I — cells share an east/west face (cell2 is one step in the I direction from cell1).
 * J — cells share a north/south face (cell2 is one step in the J direction from cell1).
 * K — cells share a top/bottom face  (cell2 is one step in the K direction from cell1).
 */
enum class FaceDirection
{
    I,
    J,
    K
};

/**
 * @brief Label identifying one of the six faces of a grid cell.
 *
 * This is used by the face-label overload of adjacent_cells_overlap_area() to
 * handle cases where physically-touching cells are NOT IJK-neighbours — for
 * example in nested hybrid grids where cell (1,3,5) may share a face with
 * (33,99,4).
 */
enum class CellFaceLabel
{
    Top,     // upper face  (shallower / higher Z in right-hand convention)
    Bottom,  // lower face  (deeper   / lower Z in right-hand convention)
    East,    // I+ face
    West,    // I- face
    North,   // J+ face
    South    // J- face
};

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

/**
 * @brief Compute the geometric center of a cell.
 *
 * Returns the center point of a cell as the simple average of its 8 corner points.
 *
 * @param corners The cell corners
 * @return xyz::Point The center point of the cell
 */
xyz::Point
cell_center(const CellCorners &corners);

/**
 * @brief Extract one of the six faces of a cell as an ordered array of 4 corners.
 *
 * Corners are ordered CCW when viewed from outside the cell (outward-facing normal).
 * This is used in conjunction with adjacent_cells_overlap_area to handle the
 * nested-hybrid-grid case where touching cells are not IJK-neighbours.
 *
 * @param cell  The cell whose face is to be extracted.
 * @param face  Which of the six faces to extract.
 * @return std::array<xyz::Point, 4> CCW corners of the face.
 */
std::array<xyz::Point, 4>
get_cell_face(const CellCorners &cell, CellFaceLabel face);

/**
 * @brief Compute the overlap area between any two cell faces.
 *
 * This overload does **not** require the cells to be IJK-neighbours. It is the
 * correct choice for nested hybrid grids where a large cell (e.g. (1,3,5)) is
 * physically adjacent to a smaller cell (e.g. (33,99,4)) that does not share
 * pillars with it. The caller must specify which face of each cell is the
 * touching one.
 *
 * Both faces are projected onto the plane perpendicular to the average of
 * their outward normals, the Sutherland-Hodgman algorithm clips the two
 * projected quadrilaterals, and the area of the intersection polygon is returned.
 *
 * **Non-adjacency guard** — if the centroid-to-centroid distance along the
 * average face normal exceeds `max_normal_gap`, the faces are considered
 * geometrically separated (not truly adjacent) and the function returns 0.
 * The default (no limit) preserves backward-compatible behaviour; pass e.g.
 * the maximum expected fault throw or pillar mismatch to enable the guard.
 *
 * @param cell1           The first cell.
 * @param face1           Which face of cell1 is touching cell2.
 * @param cell2           The second cell.
 * @param face2           Which face of cell2 is touching cell1.
 * @param max_normal_gap  Reject face pairs whose centroids differ by more than
 *                        this distance along the average normal. Default: no limit.
 * @return Overlap area in the same length units squared as the input coordinates,
 *         or 0 if the faces do not intersect or are further apart than
 *         max_normal_gap.
 */
double
adjacent_cells_overlap_area(const CellCorners &cell1,
                            CellFaceLabel face1,
                            const CellCorners &cell2,
                            CellFaceLabel face2,
                            double max_normal_gap = -1.0);

/**
 * @brief Compute the overlap area between two IJK-adjacent cells.
 *
 * Convenience wrapper for the common case where cell2 is exactly one grid step
 * away from cell1.  Internally determines the touching face pair from @p direction
 * and delegates to adjacent_cells_overlap_area(cell1, face1, cell2, face2).
 *
 * For the nested-hybrid-grid case (non-IJK neighbours), use the face-label
 * overload instead.
 *
 * @param cell1           The first cell.
 * @param cell2           The second cell, adjacent to cell1 in @p direction.
 * @param direction       I: cell2 is the I+1 neighbour (east);
 *                        J: cell2 is the J+1 neighbour (north);
 *                        K: cell2 is the K+1 neighbour (deeper).
 * @param max_normal_gap  See the face-label overload for a full description.
 *                        Default: -1 (auto-compute from face diagonal).
 * @return Overlap area in the same length units squared as the input coordinates.
 */
double
adjacent_cells_overlap_area(const CellCorners &cell1,
                            const CellCorners &cell2,
                            FaceDirection direction,
                            double max_normal_gap = -1.0);

/**
 * @brief Result of a face overlap computation with TPFA half-distances.
 *
 * Contains everything needed to assemble a TPFA transmissibility:
 *
 *   HT_i = k_i * area / d_i        (half-transmissibility, cell i)
 *   T    = HT_1 * HT_2 / (HT_1 + HT_2)
 *
 * `d_i` is the distance from cell i's geometric centre to the shared face centroid,
 * measured along the face normal.  For axis-aligned box cells this equals half the
 * cell width in the normal direction.  For distorted cells it approximates the
 * projection distance used in TPFA reservoir simulators.
 *
 * When area == 0 (no overlap or gap-guard triggered), d1 and d2 are also 0.
 */
struct FaceOverlapResult
{
    double area;        ///< Overlap area in coordinate units² (0 when no overlap).
    xyz::Point normal;  ///< Unit normal aligned with the average face plane.
    double d1;  ///< |fc1-cc1|² / |n·(fc1-cc1)| — OPM-compatible TPFA half-distance.
    double d2;  ///< |fc2-cc2|² / |n·(fc2-cc2)| — OPM-compatible TPFA half-distance.
};

/**
 * @brief Compute overlap area, face normal, and TPFA half-distances for two faces.
 *
 * Returns the same area as adjacent_cells_overlap_area() plus the unit normal and
 * per-cell half-distances, so the caller can directly compute TPFA
 * half-transmissibilities without repeating the geometry computation.
 *
 * Cell centres are computed as the average of the 8 CellCorners points.
 * The face centroid is the average of the 4 face-corner points.
 *
 * @param cell1          First cell.
 * @param face1          Which face of cell1 touches cell2.
 * @param cell2          Second cell.
 * @param face2          Which face of cell2 touches cell1.
 * @param max_normal_gap See adjacent_cells_overlap_area() for a full description.
 * @return FaceOverlapResult{area, normal, d1, d2}.
 */
FaceOverlapResult
face_overlap_result(const CellCorners &cell1,
                    CellFaceLabel face1,
                    const CellCorners &cell2,
                    CellFaceLabel face2,
                    double max_normal_gap = -1.0,
                    int coord_axis = -1);

// =====================================================================================
// TRANSMISSIBILITIES
// =====================================================================================

/**
 * @brief NNC connection types.
 */
enum class NNCType : int32_t
{
    Fault = 0,  ///< Cross-fault connection: same face direction, different K (or I/J).
    Pinchout = 1  ///< Connection across one or more inactive/collapsed K layers.
};

/**
 * @brief Result of compute_transmissibilities().
 *
 * TRAN arrays (tranx, trany, tranz) hold the TPFA transmissibility for each
 * directly-adjacent IJK pair in the I, J, and K directions respectively.
 * They are NaN where either cell in the pair is inactive.
 *
 * NNC arrays hold cross-fault and pinch-out connections that cannot be expressed
 * as a direct IJK pair; each connection appears exactly once.
 *
 * Units: transmissibility has units of [permeability * length] (e.g. mD·m if
 * permeability is in mD and coordinates are in metres).
 */
struct TransmissibilityResult
{
    py::array_t<double> tranx;  ///< shape (ncol-1, nrow,   nlay  )
    py::array_t<double> trany;  ///< shape (ncol,   nrow-1, nlay  )
    py::array_t<double> tranz;  ///< shape (ncol,   nrow,   nlay-1)

    // NNC parallel arrays — same length, one entry per non-standard connection.
    py::array_t<int32_t> nnc_i1, nnc_j1, nnc_k1;  ///< 0-based IJK of the first cell
    py::array_t<int32_t> nnc_i2, nnc_j2, nnc_k2;  ///< 0-based IJK of the second cell
    py::array_t<double> nnc_T;                    ///< Transmissibility
    py::array_t<int32_t> nnc_type;                ///< NNCType value
};

/**
 * @brief Compute TPFA transmissibilities for all active cell pairs in a grid.
 *
 * For each pair of active cells that share a physical face the function computes
 * the two-point flux approximation transmissibility:
 *
 *   HT_i = k_eff_i * A / d_i     (half-transmissibility)
 *   T    = HT_1 * HT_2 / (HT_1 + HT_2)
 *
 * where A and d_i come from face_overlap_result().
 *
 * **Effective permeability**
 *   - I-direction: permx * ntg
 *   - J-direction: permy * ntg
 *   - K-direction: permz  (NTG does not scale vertical permeability)
 *
 * **Fault NNCs**
 * For I/J faces, the algorithm scans the adjacent column for all cells whose
 * Z range overlaps the reference face, not just the direct IJK neighbour.  Any
 * pair with non-zero overlap area that is *not* the direct same-K pair is
 * recorded as a Fault NNC.
 *
 * **Pinch-out NNCs**
 * In each (i,j) column, runs of inactive cells are found; the active cell
 * immediately above and below the run are connected as a Pinchout NNC.
 *
 * @param grid       The C++ grid object.
 * @param permx      Cell permeability in the I direction, shape (ncol, nrow, nlay).
 * @param permy      Cell permeability in the J direction.
 * @param permz      Cell permeability in the K direction.
 * @param ntg        Net-to-gross ratio, shape (ncol, nrow, nlay).
 * @param min_dz_pinchout  Z-thickness below which a K-layer is considered a
 *                         pinch-out.  Currently used to skip zero-thickness
 *                         cells in the column scan. Default: 1e-4.
 * @return TransmissibilityResult.
 */
TransmissibilityResult
compute_transmissibilities(const Grid &grid,
                           const py::array_t<double> &permx,
                           const py::array_t<double> &permy,
                           const py::array_t<double> &permz,
                           const py::array_t<double> &ntg,
                           double min_dz_pinchout = 1e-4);

py::array_t<double>
get_cell_volumes(const Grid &grid_cpp,
                 geometry::HexVolumePrecision precision,
                 const bool asmasked);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_phase_cell_volumes(const Grid &grid_cpp,
                       const py::array_t<double> &water_contact,
                       const py::array_t<double> &gas_contact,
                       const std::optional<xyz::Polygon> &boundary,
                       geometry::HexVolumePrecision precision,
                       const bool asmasked);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_cell_centers(const Grid &grid_cpp, const bool asmasked = false);

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_height_above_ffl(const Grid &grid_cpp,
                     const py::array_t<float> &ffl,
                     const HeightAboveFFLOption option);

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
refine_vertically(const Grid &grid_cpp, const py::array_t<uint16_t> refine_layer);

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
get_indices_from_pointset(const Grid &grid,
                          const xyz::PointSet &points,
                          const Grid &one_grid,
                          const regsurf::RegularSurface &top_i,
                          const regsurf::RegularSurface &top_j,
                          const regsurf::RegularSurface &base_i,
                          const regsurf::RegularSurface &base_j,
                          const regsurf::RegularSurface &top_d,
                          const regsurf::RegularSurface &base_d,
                          const double threshold_magic,
                          const bool active_only,
                          const geometry::PointInHexahedronMethod point_in_hex_method);

py::array_t<double>
get_grid_fence(const Grid &grd,
               const Grid &one_grid,
               const py::array_t<double> &fspec,
               const py::array_t<double> &property,
               const py::array_t<double> &z_vector,
               const regsurf::RegularSurface &top_i,
               const regsurf::RegularSurface &top_j,
               const regsurf::RegularSurface &base_i,
               const regsurf::RegularSurface &base_j,
               const regsurf::RegularSurface &top_d,
               const regsurf::RegularSurface &base_d,
               const double threshold_magic);

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_columns(const Grid &grid_cpp, const py::array_t<uint16_t> refinement);

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_rows(const Grid &grid_cpp, const py::array_t<uint16_t> refinement);

py::array_t<float>
collapse_inactive_cells(const Grid &grid_cpp, bool collapse_internal = true);

// =====================================================================================
// WELL TRAJECTORY OPERATIONS
// =====================================================================================

/**
 * @brief Compute the entry/exit coordinates and MD for every cell that a well
 *        trajectory intersects.
 *
 * See well_cell_intersections.cpp for full documentation.
 */
py::dict
compute_well_cell_intersections(const Grid &grid,
                                const py::array_t<double> &xv,
                                const py::array_t<double> &yv,
                                const py::array_t<double> &zv,
                                const py::array_t<double> &mdv,
                                const double sampling_step,
                                const int refine_iters,
                                const bool active_only,
                                const geometry::PointInHexahedronMethod method);

std::tuple<py::array_t<float>, py::array_t<int8_t>>
convert_to_hybrid_grid(const Grid &grid_cpp,
                       float top_level,
                       float bottom_level,
                       size_t ndiv,
                       py::array_t<int> &region_prop,
                       int use_region);

// =====================================================================================
// ZCORNSV UTILITIES
// =====================================================================================

py::array_t<float>
zcornsv_pillar_to_cell(const py::array_t<float> &zcornsv_pillar);

py::array_t<float>
zcornsv_cell_to_pillar(const py::array_t<float> &zcornsv_cell,
                       bool fill_boundary = true);

// =====================================================================================
// PYTHON BINDINGS, IF NEEDED
// =====================================================================================

inline void
init(py::module &m)
{
    auto m_grid3d =
      m.def_submodule("grid3d", "Internal functions for operations on 3d grids.");

    py::enum_<HeightAboveFFLOption>(m_grid3d, "HeightAboveFFLOption")
      .value("CellCenter", HeightAboveFFLOption::CellCenter)
      .value("CellCorners", HeightAboveFFLOption::CellCorners)
      .value("TruncatedCellCorners", HeightAboveFFLOption::TruncatedCellCorners);
    //  .export_values();

    py::class_<Grid>(m_grid3d, "Grid")
      // constructors
      .def(py::init<const py::object &>(), py::arg("grid"))
      .def(py::init<size_t, size_t, size_t, const py::array_t<double> &,
                    const py::array_t<float> &, const py::array_t<int> &>(),
           py::arg("ncol"), py::arg("nrow"), py::arg("nlay"), py::arg("coordsv"),
           py::arg("zcornsv"), py::arg("actnumsv"))

      // members and properties
      .def_property_readonly("ncol", &Grid::get_ncol)
      .def_property_readonly("nrow", &Grid::get_nrow)
      .def_property_readonly("nlay", &Grid::get_nlay)
      .def_property_readonly("coordsv", &Grid::get_coordsv)
      .def_property_readonly("zcornsv", &Grid::get_zcornsv)
      .def_property_readonly("actnumsv", &Grid::get_actnumsv)

      .def("extract_onelayer_grid", &Grid::extract_onelayer_grid,
           "Get a one-layer grid - returns coordv, zcornsv, and actnumsv arrays ")
      .def("get_bounding_box", &Grid::get_bounding_box, "Get bounding box of full grid")
      .def("fix_zero_pillars", &Grid::fix_zero_pillars, "Fix zero pillars in the grid.")

      // free form functions in C++ to members in Python
      .def("get_cell_volumes", &get_cell_volumes, "Compute the bulk volume of cell.")
      .def("get_phase_cell_volumes", &get_phase_cell_volumes,
           "Compute the phase bulk volume of cell.")
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
      .def("get_indices_from_pointset", &get_indices_from_pointset,
           "Get the indices of a point set in the grid")
      .def("refine_vertically", &refine_vertically, "Refine vertically, proportionally")
      .def("refine_columns", &refine_columns, "Refine per column proportionally")
      .def("refine_rows", &refine_rows, "Refine per row proportionally")
      .def("get_grid_fence", &get_grid_fence,
           "Get a grid fence from a grid, fspec, property and z_vector")
      .def("collapse_inactive_cells", &collapse_inactive_cells,
           "Collapse inactive cells in the grid.", py::arg("collapse_internal") = true)
      .def("compute_well_cell_intersections", &compute_well_cell_intersections,
           "Compute X, Y, Z and MD entry/exit coordinates for every cell that a well "
           "trajectory passes through.",
           py::arg("xv"), py::arg("yv"), py::arg("zv"), py::arg("mdv"),
           py::arg("sampling_step") = 1.0, py::arg("refine_iters") = 20,
           py::arg("active_only") = false,
           py::arg("method") = geometry::PointInHexahedronMethod::Optimized)
      .def("convert_to_hybrid_grid", &convert_to_hybrid_grid,
           "Convert the grid to a hybrid grid.");

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

    py::enum_<FaceDirection>(m_grid3d, "FaceDirection")
      .value("I", FaceDirection::I)
      .value("J", FaceDirection::J)
      .value("K", FaceDirection::K)
      .export_values();

    py::enum_<CellFaceLabel>(m_grid3d, "CellFaceLabel")
      .value("Top", CellFaceLabel::Top)
      .value("Bottom", CellFaceLabel::Bottom)
      .value("East", CellFaceLabel::East)
      .value("West", CellFaceLabel::West)
      .value("North", CellFaceLabel::North)
      .value("South", CellFaceLabel::South)
      .export_values();

    m_grid3d.def(
      "get_cell_face", &get_cell_face,
      "Extract one of the six faces of a cell as an ordered list of 4 corners.",
      py::arg("cell"), py::arg("face"));

    // Overload 1: explicit face labels — works for any two cells, including
    // non-IJK-neighbours in nested hybrid grids.
    m_grid3d.def(
      "adjacent_cells_overlap_area",
      py::overload_cast<const CellCorners &, CellFaceLabel, const CellCorners &,
                        CellFaceLabel, double>(&adjacent_cells_overlap_area),
      "Compute the overlap area between two touching cell faces identified by their "
      "face labels. Use this overload for nested hybrid grids where the cells are "
      "not IJK-neighbours. Pass max_normal_gap (in coordinate units) to guard "
      "against false positives when the caller is uncertain about true adjacency: "
      "if the face centroids differ by more than max_normal_gap along the average "
      "face normal the function returns 0.",
      py::arg("cell1"), py::arg("face1"), py::arg("cell2"), py::arg("face2"),
      py::arg("max_normal_gap") = -1.0);

    // Overload 2: IJK-direction shorthand for regular neighbour pairs.
    m_grid3d.def(
      "adjacent_cells_overlap_area",
      py::overload_cast<const CellCorners &, const CellCorners &, FaceDirection,
                        double>(&adjacent_cells_overlap_area),
      "Compute the overlap area between two IJK-adjacent cells. For non-IJK "
      "neighbours (nested hybrid grids) use the face-label overload instead. "
      "Pass max_normal_gap to guard against accidentally passing non-adjacent cells.",
      py::arg("cell1"), py::arg("cell2"), py::arg("direction"),
      py::arg("max_normal_gap") = -1.0);

    py::class_<FaceOverlapResult>(m_grid3d, "FaceOverlapResult")
      .def_readonly("area", &FaceOverlapResult::area,
                    "Overlap area in coordinate units squared; 0 when no overlap.")
      .def_readonly("normal", &FaceOverlapResult::normal,
                    "Unit normal of the shared face plane.")
      .def_readonly("d1", &FaceOverlapResult::d1,
                    "|fc1-cc1|² / |n·(fc1-cc1)| (OPM-compatible TPFA half-distance)")
      .def_readonly("d2", &FaceOverlapResult::d2,
                    "|fc2-cc2|² / |n·(fc2-cc2)| (OPM-compatible TPFA half-distance)");

    m_grid3d.def(
      "face_overlap_result", &face_overlap_result,
      "Compute overlap area, face unit normal, and TPFA half-distances (d1, d2) for "
      "two touching cell faces. Use the result to assemble half-transmissibilities: "
      "HT_i = k_i * area / d_i, T = HT1*HT2/(HT1+HT2). "
      "Returns a FaceOverlapResult with all fields zero when area == 0.",
      py::arg("cell1"), py::arg("face1"), py::arg("cell2"), py::arg("face2"),
      py::arg("max_normal_gap") = -1.0, py::arg("coord_axis") = -1);

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
    m_grid3d.def("zcornsv_pillar_to_cell", &zcornsv_pillar_to_cell,
                 "Convert zcornsv from pillar format to cell format.");
    m_grid3d.def("zcornsv_cell_to_pillar", &zcornsv_cell_to_pillar,
                 "Convert zcornsv from cell format to pillar format.",
                 py::arg("zcornsv_cell"), py::arg("fill_boundary") = true);

    py::enum_<NNCType>(m_grid3d, "NNCType")
      .value("Fault", NNCType::Fault)
      .value("Pinchout", NNCType::Pinchout)
      .export_values();

    py::class_<TransmissibilityResult>(m_grid3d, "TransmissibilityResult")
      .def_readonly("tranx", &TransmissibilityResult::tranx)
      .def_readonly("trany", &TransmissibilityResult::trany)
      .def_readonly("tranz", &TransmissibilityResult::tranz)
      .def_readonly("nnc_i1", &TransmissibilityResult::nnc_i1)
      .def_readonly("nnc_j1", &TransmissibilityResult::nnc_j1)
      .def_readonly("nnc_k1", &TransmissibilityResult::nnc_k1)
      .def_readonly("nnc_i2", &TransmissibilityResult::nnc_i2)
      .def_readonly("nnc_j2", &TransmissibilityResult::nnc_j2)
      .def_readonly("nnc_k2", &TransmissibilityResult::nnc_k2)
      .def_readonly("nnc_T", &TransmissibilityResult::nnc_T)
      .def_readonly("nnc_type", &TransmissibilityResult::nnc_type);

    m_grid3d.def(
      "compute_transmissibilities", &compute_transmissibilities,
      "Compute TPFA transmissibilities for all active cell pairs. Returns a "
      "TransmissibilityResult with TRAN arrays (shape ncol-1/nrow-1/nlay-1) and "
      "NNC parallel arrays for fault and pinch-out connections.",
      py::arg("grid"), py::arg("permx"), py::arg("permy"), py::arg("permz"),
      py::arg("ntg"), py::arg("min_dz_pinchout") = 1e-4);
}

}  // namespace xtgeo::grid3d

#endif  // XTGEO_GRID3D_HPP_
