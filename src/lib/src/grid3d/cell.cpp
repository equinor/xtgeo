#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>
#include <limits>
#include <optional>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/numerics.hpp>
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
 * @param i The (i) coordinate
 * @param j The (j) coordinate
 * @param k The (k) coordinate
 * @param ncol The grid column/nx dimension
 * @param nrow The grid row/ny dimension
 * @param nlay The grid layer/nz dimension
 * @param coordsv Grid coordnates vector
 * @param zcornsv Grid Z corners vector
 * @return vector of 24 doubles with the corner coordinates.
 */
std::vector<double>
cell_corners(const size_t i,
             const size_t j,
             const size_t k,
             const size_t ncol,
             const size_t nrow,
             const size_t nlay,
             const py::array_t<double> &coordsv,
             const py::array_t<float> &zcornsv)
{
    auto coordsv_ = coordsv.data();
    auto zcornsv_ = zcornsv.data();

    double coords[4][6]{};
    auto num_rows = nrow + 1;
    auto num_layers = nlay + 1;
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

    std::vector<double> corners(24, 0);
    auto crn_idx = 0;
    auto cz_idx = 0;
    for (auto layer = 0; layer < 2; layer++) {
        for (auto n = 0; n < 4; n++) {
            auto x1 = coords[n][0], y1 = coords[n][1], z1 = coords[n][2];
            auto x2 = coords[n][3], y2 = coords[n][4], z2 = coords[n][5];
            auto t = (z_coords[cz_idx] - z1) / (z2 - z1);
            auto point = numerics::lerp3d(x1, y1, z1, x2, y2, z2, t);
            // If coord lines are collapsed (preserves old behavior)
            if (std::abs(z2 - z1) < std::numeric_limits<double>::epsilon()) {
                point.x = x1;
                point.y = y1;
            }
            corners[crn_idx++] = point.x;
            corners[crn_idx++] = point.y;
            corners[crn_idx++] = z_coords[cz_idx];
            cz_idx++;
        }
    }
    return corners;
}

std::vector<double>
get_corners_minmax(std::vector<double> &corners)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();
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
 * @param i The (i) coordinate
 * @param j The (j) coordinate
 * @param k The (k) coordinate
 * @param corners A vector of doubles, length 24
 * @param point A vector of doubles, length 2
 * @return Boolean
 */
bool
is_xy_point_in_cell(const double x,
                    const double y,
                    const size_t i,
                    const size_t j,
                    const size_t k,
                    const size_t ncol,
                    const size_t nrow,
                    const size_t nlay,
                    const py::array_t<double> &coordsv,
                    const py::array_t<float> &zcornsv,
                    const int option = 0)
{

    std::array<std::array<double, 3>, 8> crn{};
    size_t idx = 0;

    auto corners = cell_corners(i, j, k, ncol, nrow, nlay, coordsv, zcornsv);
    for (auto i = 0; i < 8; i++) {
        for (auto j = 0; j < 3; j++) {
            crn[i][j] = corners[idx++];
        }
    }

    // make a closed polygon with 4 corners
    std::vector<std::array<double, 2>> quadrilateral(5);
    for (auto i = 0; i <= 4; i++) {
        auto j = i;
        if (option == 1) {  // bottom
            j = i + 4;
        }
        if (i == 2)
            j = j + 1;
        if (i == 3)
            j = j - 1;
        if (i == 4)
            j = j - 4;
        quadrilateral[i][0] = crn[j][0];
        quadrilateral[i][1] = crn[j][1];
    }

    // determine if point is inside the polygon
    return geometry::is_xy_point_in_polygon(x, y, quadrilateral);

}  // is_xy_point_in_cell

// Find the depth of a point in a cell
double
get_depth_in_cell(const double x,
                  const double y,
                  const size_t i,
                  const size_t j,
                  const size_t k,
                  const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const int option = 0)
{
    double depth = std::numeric_limits<double>::quiet_NaN();
    auto corners = cell_corners(i, j, k, ncol, nrow, nlay, coordsv, zcornsv);
    if (option == 1) {
        depth =
          geometry::interpolate_z_4p(x, y, { corners[12], corners[13], corners[14] },
                                     { corners[15], corners[16], corners[17] },
                                     { corners[18], corners[19], corners[20] },
                                     { corners[21], corners[22], corners[23] });
    } else {
        depth = geometry::interpolate_z_4p(x, y, { corners[0], corners[1], corners[2] },
                                           { corners[3], corners[4], corners[5] },
                                           { corners[6], corners[7], corners[8] },
                                           { corners[9], corners[10], corners[11] });
    }
    return depth;
}  // depth_in_cell

}  // namespace xtgeo::grid3d
