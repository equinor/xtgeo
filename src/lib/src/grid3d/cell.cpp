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
    auto coordsv_ = grd.coordsv.data();
    auto zcornsv_ = grd.zcornsv.data();

    double coords[4][6]{};
    auto num_rows = grd.nrow + 1;
    auto num_layers = grd.nlay + 1;
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
                point.x = x1;
                point.y = y1;
            }
            corners[crn_idx++] = point.x;
            corners[crn_idx++] = point.y;
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
get_corners_minmax(CellCorners &get_cell_corners_from_ijk)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();

    auto corners = get_cell_corners_from_ijk.arrange_corners();

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

/*
 * Estimate if a point is inside a cell face top (option != 1) or cell face bottom
 * (option = 1), seen from above, and return True if it is inside, False otherwise.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param CellCorners struct
 * @param option 0: Use cell top, 1: Use cell bottom, 2 for center
 * @return Boolean
 */
bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const CellCorners &corners,
                    int option)
{
    if (option < 0 || option > 2) {
        throw std::invalid_argument("BUG! Invalid option");
    }

    // determine if point is inside the polygon
    if (option == 0) {
        return geometry::is_xy_point_in_quadrilateral(
          x, y, corners.upper_sw, corners.upper_se, corners.upper_ne, corners.upper_nw);
    } else if (option == 1) {
        return geometry::is_xy_point_in_quadrilateral(
          x, y, corners.lower_sw, corners.lower_se, corners.lower_ne, corners.lower_nw);
    } else if (option == 2) {
        // find the center Z point of the cell
        auto mid_sw = numerics::lerp3d(corners.upper_sw.x, corners.upper_sw.y,
                                       corners.upper_sw.z, corners.lower_sw.x,
                                       corners.lower_sw.y, corners.lower_sw.z, 0.5);
        auto mid_se = numerics::lerp3d(corners.upper_se.x, corners.upper_se.y,
                                       corners.upper_se.z, corners.lower_se.x,
                                       corners.lower_se.y, corners.lower_se.z, 0.5);
        auto mid_nw = numerics::lerp3d(corners.upper_nw.x, corners.upper_nw.y,
                                       corners.upper_nw.z, corners.lower_nw.x,
                                       corners.lower_nw.y, corners.lower_nw.z, 0.5);
        auto mid_ne = numerics::lerp3d(corners.upper_ne.x, corners.upper_ne.y,
                                       corners.upper_ne.z, corners.lower_ne.x,
                                       corners.lower_ne.y, corners.lower_ne.z, 0.5);

        return geometry::is_xy_point_in_quadrilateral(
          x, y, { mid_sw.x, mid_sw.y, mid_sw.z }, { mid_se.x, mid_se.y, mid_se.z },
          { mid_ne.x, mid_ne.y, mid_ne.z }, { mid_nw.x, mid_nw.y, mid_nw.z });
    }
    return false;  // unreachable
}  // is_xy_point_in_cell

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

}  // namespace xtgeo::grid3d
