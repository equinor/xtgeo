#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>
#include <xtgeo/xyz.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Given a cell coordinate (i, j, k), find all corner coordinates as an
 * array with 24 values.
 *
 *      Top  --> i-dir     Base c
 *
 * (6,7,8) (9,10,11) (18,19,20) (21,22,23)
 *    |-------|          |-------|
 *    |       |          |       |
 *    |       |          |       |
 *    |-------|          |-------|
 * (0,1,2) (3,4,5)   (12,13,14) (15,16,17)
 * (i,j,k)
 *
 * @param Grid struct
 * @param i The (i) coordinate
 * @param j The (j) coordinate
 * @param k The (k) coordinate
 * @return CellCorners
 */
CellCorners
get_cell_corners_from_ijk(const Grid &grd,
                          const size_t i,
                          const size_t j,
                          const size_t k)
{
    auto coordsv_ = grd.get_coordsv().data();
    auto zcornsv_ = grd.get_zcornsv().data();

    double coords[4][6]{};
    auto num_rows = grd.get_nrow() + 1;
    auto num_layers = grd.get_nlay() + 1;
    auto n = 0;
    // Each cell is defined by 4 pillars
    for (auto x = 0; x < 2; x++) {
        for (auto y = 0; y < 2; y++) {
            for (auto z = 0; z < 6; z++) {
                auto idx = (i + y) * num_rows * 6 + (j + x) * 6 + z;
                coords[n][z] = coordsv_[idx];
            }
            n++;
        }
    }

    double z_coords[8]{};
    auto area = num_rows * num_layers;
    // Get the z value of each corner
    z_coords[0] = zcornsv_[((i + 0) * area + (j + 0) * num_layers + (k + 0)) * 4 + 3];
    z_coords[1] = zcornsv_[((i + 1) * area + (j + 0) * num_layers + (k + 0)) * 4 + 2];
    z_coords[2] = zcornsv_[((i + 0) * area + (j + 1) * num_layers + (k + 0)) * 4 + 1];
    z_coords[3] = zcornsv_[((i + 1) * area + (j + 1) * num_layers + (k + 0)) * 4 + 0];

    z_coords[4] = zcornsv_[((i + 0) * area + (j + 0) * num_layers + (k + 1)) * 4 + 3];
    z_coords[5] = zcornsv_[((i + 1) * area + (j + 0) * num_layers + (k + 1)) * 4 + 2];
    z_coords[6] = zcornsv_[((i + 0) * area + (j + 1) * num_layers + (k + 1)) * 4 + 1];
    z_coords[7] = zcornsv_[((i + 1) * area + (j + 1) * num_layers + (k + 1)) * 4 + 0];

    std::array<double, 24> corners{};
    auto crn_idx = 0;
    auto cz_idx = 0;
    for (auto layer = 0; layer < 2; layer++) {
        for (auto n = 0; n < 4; n++) {
            auto x1 = coords[n][0], y1 = coords[n][1], z1 = coords[n][2];
            auto x2 = coords[n][3], y2 = coords[n][4], z2 = coords[n][5];
            auto t = (z_coords[cz_idx] - z1) / (z2 - z1);
            auto point = numerics::lerp3d(x1, y1, z1, x2, y2, z2, t);
            // If coord lines are collapsed (preserves old behavior)
            if (std::abs(z2 - z1) < numerics::EPSILON) {
                point.set_x(x1);
                point.set_y(y1);
            }
            corners[crn_idx++] = point.x();
            corners[crn_idx++] = point.y();
            corners[crn_idx++] = z_coords[cz_idx];
            cz_idx++;
        }
    }
    return CellCorners(corners);
}

/*
 * Get the minimum and maximum values of the corners of a cell.
 * @param CellCorners struct
 * @return std::vector<double>
 */
std::vector<double>
get_corners_minmax(const CellCorners &cell_corners)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();

    auto corners = cell_corners.arrange_corners();

    for (auto i = 0; i < 24; i += 3) {
        if (corners[i] < xmin) {
            xmin = corners[i];
        }
        if (corners[i] > xmax) {
            xmax = corners[i];
        }
        if (corners[i + 1] < ymin) {
            ymin = corners[i + 1];
        }
        if (corners[i + 1] > ymax) {
            ymax = corners[i + 1];
        }
        if (corners[i + 2] < zmin) {
            zmin = corners[i + 2];
        }
        if (corners[i + 2] > zmax) {
            zmax = corners[i + 2];
        }
    }
    std::vector<double> minmax{ xmin, xmax, ymin, ymax, zmin, zmax };
    return minmax;
}  // get_corners_minmax

/**
 * @brief Get the bounding box for a cell, a wrapper for get_corners_minmax.
 * @param CellCorners struct
 * @return std::tuple<xyz::Point, xyz::Point> {min_point, max_point}
 */
std::tuple<xyz::Point, xyz::Point>
get_cell_bounding_box(const CellCorners &corners)
{
    auto minmax = get_corners_minmax(corners);
    auto min_point = xyz::Point(minmax[0], minmax[2], minmax[4]);
    auto max_point = xyz::Point(minmax[1], minmax[3], minmax[5]);
    return std::make_tuple(min_point, max_point);
}  // get_cell_bounding_box

/*
 * Get the depth of a point inside a cell.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param CellCorners struct
 * @param option 0: Use cell top, 1: Use cell bottom
 * @return double
 */

double
get_depth_in_cell(const double x,
                  const double y,
                  const CellCorners &corners,
                  int option = 0)
{
    if (option < 0 || option > 1) {
        throw std::invalid_argument("BUG! Invalid option");
    }

    double depth = numerics::QUIET_NAN;

    if (option == 1) {
        depth = geometry::interpolate_z_4p(x, y, corners.lower_sw, corners.lower_se,
                                           corners.lower_nw, corners.lower_ne);
    } else {
        depth = geometry::interpolate_z_4p(x, y, corners.upper_sw, corners.upper_se,
                                           corners.upper_nw, corners.upper_ne);
    }
    return depth;
}  // get_depth_in_cell

bool
is_cell_non_convex(const CellCorners &corners)
{
    // Check if the cell is non-convex
    return geometry::is_hexahedron_non_convex(corners.to_hexahedron_corners());
}  // is_cell_non_convex

bool
is_cell_distorted(const CellCorners &corners)
{
    // Check if the cell is distorted heavily
    return geometry::is_hexahedron_severely_distorted(corners.to_hexahedron_corners());
}  // is_cell_distorted

/*
 * Extract one of the six faces of a cell as an ordered array of 4 corners.
 *
 * Corners are ordered CCW when viewed from outside the cell (outward-facing normal),
 * which is what quadrilateral_face_overlap_area expects.
 */
std::array<xyz::Point, 4>
get_cell_face(const CellCorners &cell, CellFaceLabel face)
{
    switch (face) {
        case CellFaceLabel::Top:
            return { cell.upper_sw, cell.upper_se, cell.upper_ne, cell.upper_nw };
        case CellFaceLabel::Bottom:
            return { cell.lower_sw, cell.lower_se, cell.lower_ne, cell.lower_nw };
        case CellFaceLabel::East:
            return { cell.upper_se, cell.upper_ne, cell.lower_ne, cell.lower_se };
        case CellFaceLabel::West:
            return { cell.upper_sw, cell.upper_nw, cell.lower_nw, cell.lower_sw };
        case CellFaceLabel::North:
            return { cell.upper_nw, cell.upper_ne, cell.lower_ne, cell.lower_nw };
        case CellFaceLabel::South:
            return { cell.upper_sw, cell.upper_se, cell.lower_se, cell.lower_sw };
    }
    throw std::invalid_argument("Invalid CellFaceLabel");
}  // get_cell_face

/*
 * Compute the overlap area between any two cell faces identified by explicit face
 * labels.  This is the canonical implementation; it works for IJK-neighbours and for
 * non-IJK-neighbours alike (e.g. nested hybrid grids).
 */
double
adjacent_cells_overlap_area(const CellCorners &cell1,
                            CellFaceLabel face1,
                            const CellCorners &cell2,
                            CellFaceLabel face2,
                            double max_normal_gap)
{
    return geometry::quadrilateral_face_overlap_area(
      get_cell_face(cell1, face1), get_cell_face(cell2, face2), max_normal_gap);
}  // adjacent_cells_overlap_area (face-label overload)

/*
 * Convenience overload for the common case where cell2 is one IJK step away from
 * cell1.  Translates the direction into the appropriate face-label pair and delegates
 * to the face-label overload.
 *
 * For nested hybrid grids where touching cells are NOT IJK-neighbours, the caller
 * must use the face-label overload directly.
 */
double
adjacent_cells_overlap_area(const CellCorners &cell1,
                            const CellCorners &cell2,
                            FaceDirection direction,
                            double max_normal_gap)
{
    switch (direction) {
        case FaceDirection::I:
            return adjacent_cells_overlap_area(cell1, CellFaceLabel::East, cell2,
                                               CellFaceLabel::West, max_normal_gap);
        case FaceDirection::J:
            return adjacent_cells_overlap_area(cell1, CellFaceLabel::North, cell2,
                                               CellFaceLabel::South, max_normal_gap);
        case FaceDirection::K:
            return adjacent_cells_overlap_area(cell1, CellFaceLabel::Bottom, cell2,
                                               CellFaceLabel::Top, max_normal_gap);
    }
    throw std::invalid_argument("Invalid FaceDirection");
}  // adjacent_cells_overlap_area (direction overload)

/*
 * Compute the geometric center of a cell: simple average of its 8 corners.
 * This is the canonical implementation used throughout the codebase.
 */
xyz::Point
cell_center(const CellCorners &c)
{
    return (c.upper_sw + c.upper_se + c.upper_nw + c.upper_ne + c.lower_sw +
            c.lower_se + c.lower_nw + c.lower_ne) *
           0.125;
}

/*
 * Compute overlap area, face normal, and TPFA half-distances for two cell faces.
 */
FaceOverlapResult
face_overlap_result(const CellCorners &cell1,
                    CellFaceLabel face1,
                    const CellCorners &cell2,
                    CellFaceLabel face2,
                    double max_normal_gap,
                    int coord_axis)
{
    auto f1 = get_cell_face(cell1, face1);
    auto f2 = get_cell_face(cell2, face2);

    auto qr = geometry::quadrilateral_face_overlap_result(f1, f2, max_normal_gap);

    if (qr.area == 0.0)
        return { 0.0, qr.normal, 0.0, 0.0 };

    auto face_centroid = [](const std::array<xyz::Point, 4> &f) -> Eigen::Vector3d {
        return (f[0].data() + f[1].data() + f[2].data() + f[3].data()) * 0.25;
    };

    Eigen::Vector3d fc1 = face_centroid(f1);
    Eigen::Vector3d fc2 = face_centroid(f2);
    Eigen::Vector3d cc1 = cell_center(cell1).data();
    Eigen::Vector3d cc2 = cell_center(cell2).data();

    double d1, d2;
    if (coord_axis >= 0 && coord_axis <= 2) {
        d1 = std::abs((fc1 - cc1)[coord_axis]);
        d2 = std::abs((fc2 - cc2)[coord_axis]);
    } else {
        // OPM-compatible TPFA half-distances: d_i = |fc_i - cc_i|² / |n · (fc_i -
        // cc_i)|. Equivalent to the OPM formula  T_½ = K * A * (n · d_vec) / (d_vec ·
        // d_vec) which correctly down-weights faces whose normal is misaligned with the
        // cell-centre-to-face-centroid direction (e.g. heavily tilted/faulted cells).
        Eigen::Vector3d n = qr.normal.data();
        auto v1 = fc1 - cc1;
        auto v2 = fc2 - cc2;
        double nd1 = std::abs(n.dot(v1));
        double nd2 = std::abs(n.dot(v2));
        d1 = (nd1 > 0.0) ? v1.squaredNorm() / nd1 : v1.norm();
        d2 = (nd2 > 0.0) ? v2.squaredNorm() / nd2 : v2.norm();
    }

    return { qr.area, qr.normal, d1, d2 };
}  // face_overlap_result

}  // namespace xtgeo::grid3d
