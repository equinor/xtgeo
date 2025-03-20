#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/regsurf.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xyz.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

using xyz::Point;

// Function to rotate a point (x, y) around the origin (xori, yori) by a given angle (in
// radians)
static Point
rotate_point(const Point p,
             const double xori,
             const double yori,
             const double angle_rad)
{
    // Translate point back to origin
    double x_trans = p.x - xori;
    double y_trans = p.y - yori;

    // Apply rotation matrix
    double x_rot = x_trans * std::cos(angle_rad) - y_trans * std::sin(angle_rad);
    double y_rot = x_trans * std::sin(angle_rad) + y_trans * std::cos(angle_rad);

    // Translate back to original position
    return { x_rot + xori, y_rot + yori, 0 };
}

/*
 * Function to get the 4 outer corners of the regsurf (regular 2D grid)
 *
 * 3                        4
 *  x----------------------x
 *  |                      |
 *  |                      |
 *  x----------------------x
 * 1                        2
 *
 *
 * @param xori The x-coordinate of the origin
 * @param yori The y-coordinate of the origin
 * @param xinc The x-increment
 * @param yinc The y-increment
 * @param ncol The number of columns
 * @param nrow The number of rows
 * @param angle_deg The angle of rotation in degrees
 * @return A tuple of 4 points representing the outer corners of the regsurf
 */
std::tuple<Point, Point, Point, Point>
get_outer_corners(const RegularSurface &regsurf)
{
    // Convert the angle to radians
    double angle_rad = regsurf.rotation * M_PI / 180.0;

    // Calculate the unrotated corners of the cell (i, j)
    Point bottom_left = { regsurf.xori + 0 * regsurf.xinc,
                          regsurf.yori + 0 * regsurf.yinc, 0 };
    Point bottom_right = { regsurf.xori + regsurf.ncol * regsurf.xinc,
                           regsurf.yori + 0 * regsurf.yinc, 0 };
    Point top_right = { regsurf.xori + regsurf.ncol * regsurf.xinc,
                        regsurf.yori + regsurf.nrow * regsurf.yinc, 0 };
    Point top_left = { regsurf.xori + 0 * regsurf.xinc,
                       regsurf.yori + regsurf.nrow * regsurf.yinc, 0 };

    // Get the outer corners if the 2D grid,
    Point bl_rot = rotate_point(bottom_left, regsurf.xori, regsurf.yori, angle_rad);
    Point br_rot = rotate_point(bottom_right, regsurf.xori, regsurf.yori, angle_rad);
    Point tr_rot = rotate_point(top_right, regsurf.xori, regsurf.yori, angle_rad);
    Point tl_rot = rotate_point(top_left, regsurf.xori, regsurf.yori, angle_rad);
    return { bl_rot, br_rot, tl_rot, tr_rot };  // note order
}

/*
 * Function to get the X and Y given I and J. Z in the returned Point will be 0
 * @param regsurf The RegularSurface struct representing the surface
 * @param i The i-coordinate of the origin
 * @param j The j-coordinate of the origin
 * @param yflip The flip factor for the Y-coordinate
 * @return A tuple of 4 points representing the outer corners of the regsurf
 */
Point
get_xy_from_ij(const RegularSurface &regsurf,
               const size_t i,
               const size_t j,
               const int yflip)
{
    // Convert the angle to radians
    double angle_rad = regsurf.rotation * M_PI / 180.0;

    // Calculate the unrotated corners of the cell (i, j)
    Point point = { regsurf.xori + i * regsurf.xinc,
                    regsurf.yori + j * regsurf.yinc * yflip, 0 };

    // Get the position of the point in the rotated grid
    Point point_rot = rotate_point(point, regsurf.xori, regsurf.yori, angle_rad);
    return point_rot;
}

// Function to transform a point from world to grid coordinates
static Point
inverse_rotate_and_translate(const Point p,
                             const double xori,
                             const double yori,
                             const double cos_a,
                             const double sin_a)
{
    // Translation
    double x_trans = p.x - xori;
    double y_trans = p.y - yori;
    // Rotation
    double x_rot = x_trans * cos_a + y_trans * sin_a;
    double y_rot = -x_trans * sin_a + y_trans * cos_a;

    return { x_rot, y_rot };
}

// Function to get the grid index, ensuring it is within bounds
static int
get_index(const double coord, const double increment, const int max_index)
{
    int index = round(coord / increment);

    // Ensure the index is within bounds
    if (index < 0)
        index = 0;
    if (index >= max_index)
        index = max_index - 1;

    return index;
}

using regsurf::RegularSurface;

/*
 * Find the range of grid cells within the box defined xmin, xmax, ymin, ymax
 * @param regsurf The RegularSurface object representing the surface
 * @param xmin The minimum x-coordinate of the box
 * @param xmax The maximum x-coordinate of the box
 * @param ymin The minimum y-coordinate of the box
 * @param ymax The maximum y-coordinate of the box
 * @param expand The number of cells to expand the range by
 * @return A tuple of 4 integers representing the range of grid cells (i_min, i_max,
 * j_min, j_max)
 */
std::tuple<int, int, int, int>
find_cell_range(const RegularSurface &regsurf,
                const double xmin,
                const double xmax,
                const double ymin,
                const double ymax,
                const int expand)
{
    // Convert rotation to radians
    double angle = regsurf.rotation * M_PI / 180.0;
    double cos_a = std::cos(angle);
    double sin_a = std::sin(angle);

    // Corners of the bounding box
    Point corners[4] = {
        { xmin, ymin, 0 },  // bottom-left
        { xmin, ymax, 0 },  // top-left
        { xmax, ymin, 0 },  // bottom-right
        { xmax, ymax, 0 }   // top-right
    };

    // Variables to hold min/max indices
    int i_min = regsurf.ncol;
    int i_max = -1;
    int j_min = regsurf.nrow;
    int j_max = -1;

    // Iterate over each corner, transform it, and find grid indices
    for (size_t i = 0; i < 4; ++i) {
        // Transform the corner from world to grid coordinates
        Point grid2d_coord = inverse_rotate_and_translate(corners[i], regsurf.xori,
                                                          regsurf.yori, cos_a, sin_a);

        // Find grid indices for columns (i for NCOL) and rows (j for NROW)
        int i_grid = get_index(grid2d_coord.x, regsurf.xinc, regsurf.ncol);
        int j_grid = get_index(grid2d_coord.y, regsurf.yinc, regsurf.nrow);

        // Update the range of indices, ensuring they are within the grid bounds
        i_min = std::min(i_min, i_grid);
        i_max = std::max(i_max, i_grid);
        j_min = std::min(j_min, j_grid);
        j_max = std::max(j_max, j_grid);
    }
    // Expand the range of indices if espand is greater than 0
    i_min -= expand;
    i_max += expand;
    j_min -= expand;
    j_max += expand;

    // Ensure the indices are within the grid bounds
    i_min = std::max(0, i_min);
    i_max = std::min(static_cast<int>(regsurf.ncol - 1), i_max);
    j_min = std::max(0, j_min);
    j_max = std::min(static_cast<int>(regsurf.nrow - 1), j_max);

    if (i_min > i_max || j_min > j_max) {
        // Return an invalid range if the indices are out of bounds
        return { -1, -1, -1, -1 };
    }

    return { i_min, i_max, j_min, j_max };
}

/*
 * Function to get the Z value at a given X, Y position on a regular 2D grid
 *
 * @param regsurf The RegularSurface object representing the surface
 * @param x The x-coordinate of the point
 * @param y The y-coordinate of the point
 * @return The interpolated Z value at the given X, Y position
 */

double
get_z_from_xy(const RegularSurface &regsurf,
              const double x,
              const double y,
              const double tolerance)
{
    // Convert the angle to radians
    double angle_rad = regsurf.rotation * M_PI / 180.0;

    Point p = { x, y, 0 };
    Point p_rel = inverse_rotate_and_translate(
      p, regsurf.xori, regsurf.yori, std::cos(angle_rad), std::sin(angle_rad));

    // Find the indices of the grid cell containing the point
    int i_temp = static_cast<int>(p_rel.x / regsurf.xinc);
    int j_temp = static_cast<int>(p_rel.y / regsurf.yinc);

    // check inside status, with a tolerance
    bool is_inside = geometry::is_xy_point_in_quadrilateral(
      p_rel.x, p_rel.y, { 0.0, 0.0, 0.0 },
      { (regsurf.ncol - 1) * regsurf.xinc, 0.0, 0.0 },
      { (regsurf.ncol - 1) * regsurf.xinc, (regsurf.nrow - 1) * regsurf.yinc, 0.0 },
      { 0.0, (regsurf.nrow - 1) * regsurf.yinc, 0.0 }, tolerance);

    // Check if the point is outside the grid, and return NaN if it is
    // (a bit double checking here)
    if (!is_inside || i_temp < 0 || i_temp > static_cast<int>(regsurf.ncol - 1) ||
        j_temp < 0 || j_temp > static_cast<int>(regsurf.nrow - 1)) {

        return std::numeric_limits<double>::quiet_NaN();
    }

    // Convert to size_t
    size_t i = static_cast<size_t>(i_temp);
    size_t j = static_cast<size_t>(j_temp);
    // Access the array without bounds checking
    auto values = regsurf.values.unchecked<2>();
    auto mask = regsurf.mask.unchecked<2>();

    // Get the values at the corners of the cell, beware edge effects
    size_t i2 = std::min(i + 1, regsurf.ncol - 1);
    size_t j2 = std::min(j + 1, regsurf.nrow - 1);

    double z11 = values(i, j), z12 = values(i, j2);
    double z21 = values(i2, j), z22 = values(i2, j2);

    if (mask(i, j) || mask(i, j2) || mask(i2, j) || mask(i2, j2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Perform bilinear interpolation
    double x1 = i * regsurf.xinc;
    double x2 = (i + 1) * regsurf.xinc;
    double y1 = j * regsurf.yinc;
    double y2 = (j + 1) * regsurf.yinc;

    auto res = geometry::interpolate_z_4p_regular(p_rel.x, p_rel.y, { x1, y1, z11 },
                                                  { x2, y1, z21 }, { x1, y2, z12 },
                                                  { x2, y2, z22 }, tolerance);
    return res;

}  // get_z_from_xy

}  // namespace xtgeo::regsurf
