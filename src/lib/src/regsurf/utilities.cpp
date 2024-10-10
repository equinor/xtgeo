#include <algorithm>  // Include for std::max and std::min
#include <cmath>      // Include for std::isnan
#include <cstddef>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <xtgeo/geometry.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/point.hpp>
#include <xtgeo/regsurf.hpp>

namespace py = pybind11;

namespace xtgeo::regsurf {

// Function to rotate a point (x, y) around the origin (xori, yori) by a given angle (in
// radians)
static Point
rotate_point(Point p, double xori, double yori, double angle_rad)
{
    // Translate point back to origin
    double x_trans = p.x - xori;
    double y_trans = p.y - yori;

    // Apply rotation matrix
    double x_rot = x_trans * cos(angle_rad) - y_trans * sin(angle_rad);
    double y_rot = x_trans * sin(angle_rad) + y_trans * cos(angle_rad);

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
 * @param A_deg The angle of rotation in degrees
 * @return A tuple of 4 points representing the outer corners of the regsurf
 */
std::tuple<Point, Point, Point, Point>
get_outer_corners(double xori,
                  double yori,
                  double xinc,
                  double yinc,
                  size_t ncol,
                  size_t nrow,
                  double A_deg)
{
    // Convert the angle to radians
    double A_rad = A_deg * M_PI / 180.0;

    // Vector to store corners of each cell
    std::vector<std::vector<Point>> cells;

    // Calculate the unrotated corners of the cell (i, j)
    Point bottom_left = { xori + 0 * xinc, yori + 0 * yinc, 0 };
    Point bottom_right = { xori + ncol * xinc, yori + 0 * yinc, 0 };
    Point top_right = { xori + ncol * xinc, yori + nrow * yinc, 0 };
    Point top_left = { xori + 0 * xinc, yori + nrow * yinc, 0 };

    // Get the outer corners if the 2D grid,
    Point bl_rot = rotate_point(bottom_left, xori, yori, A_rad);
    Point br_rot = rotate_point(bottom_right, xori, yori, A_rad);
    Point tr_rot = rotate_point(top_right, xori, yori, A_rad);
    Point tl_rot = rotate_point(top_left, xori, yori, A_rad);
    return { bl_rot, br_rot, tl_rot, tr_rot };  // note order
}

/*
 * Function to get the X and Y given I and J. Z in the returned Point will be 0
 * @param xori The x-coordinate of the origin
 * @param yori The y-coordinate of the origin
 * @param xinc The x-increment
 * @param yinc The y-increment
 * @param ncol The number of columns
 * @param nrow The number of rows
 * @param A_deg The angle of rotation in degrees
 * @return A tuple of 4 points representing the outer corners of the regsurf
 */
Point
get_xy_from_ij(size_t i,
               size_t j,
               double xori,
               double yori,
               double xinc,
               double yinc,
               size_t ncol,
               size_t nrow,
               double A_deg)
{
    // Convert the angle to radians
    double A_rad = A_deg * M_PI / 180.0;

    // Calculate the unrotated corners of the cell (i, j)
    Point point = { xori + i * xinc, yori + j * yinc, 0 };

    // Get the position of the point in the rotated grid
    Point point_rot = rotate_point(point, xori, yori, A_rad);
    return point_rot;
}

// Function to transform a point from world to grid coordinates
static Point
inverse_rotate_and_translate(Point p,
                             double xori,
                             double yori,
                             double cos_a,
                             double sin_a)
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
get_index(double coord, double increment, int max_index)
{
    int index = round(coord / increment);

    // Ensure the index is within bounds
    if (index < 0)
        index = 0;
    if (index >= max_index)
        index = max_index - 1;

    return index;
}

/*
 * Find the range of grid cells within the box defined xmin, xmax, ymin, ymax
 * @param xmin The minimum x-coordinate of the box
 * @param xmax The maximum x-coordinate of the box
 * @param ymin The minimum y-coordinate of the box
 * @param ymax The maximum y-coordinate of the box
 * @param xori The x-coordinate of the origin
 * @param yori The y-coordinate of the origin
 * @param xinc The x-increment
 * @param yinc The y-increment
 * @param rotation_degrees The angle of rotation in degrees
 * @param ncol The number of columns
 * @param nrow The number of rows
 * @param expand The number of cells to expand the range by
 * @return A tuple of 4 integers representing the range of grid cells (i_min, i_max,
 * j_min, j_max)
 */
std::tuple<int, int, int, int>
find_cell_range(double xmin,
                double xmax,
                double ymin,
                double ymax,
                double xori,
                double yori,
                double xinc,
                double yinc,
                double rotation_degrees,
                size_t ncol,
                size_t nrow,
                int expand)
{
    // Convert rotation to radians
    double angle = rotation_degrees * M_PI / 180.0;
    double cos_a = cos(angle);
    double sin_a = sin(angle);

    // Corners of the bounding box
    Point corners[4] = {
        { xmin, ymin, 0 },  // bottom-left
        { xmin, ymax, 0 },  // top-left
        { xmax, ymin, 0 },  // bottom-right
        { xmax, ymax, 0 }   // top-right
    };

    // Variables to hold min/max indices
    int i_min = ncol;
    int i_max = -1;
    int j_min = nrow;
    int j_max = -1;

    // Iterate over each corner, transform it, and find grid indices
    for (int i = 0; i < 4; ++i) {
        // Transform the corner from world to grid coordinates
        Point grid_coord =
          inverse_rotate_and_translate(corners[i], xori, yori, cos_a, sin_a);

        // Find grid indices for columns (i for NCOL) and rows (j for NROW)
        int i_grid = get_index(grid_coord.x, xinc, ncol);
        int j_grid = get_index(grid_coord.y, yinc, nrow);

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
    i_max = std::min(static_cast<int>(ncol - 1), i_max);
    j_min = std::max(0, j_min);
    j_max = std::min(static_cast<int>(nrow - 1), j_max);

    if (i_min > i_max || j_min > j_max) {
        // Return an invalid range if the indices are out of bounds
        return { -1, -1, -1, -1 };
    }

    return { i_min, i_max, j_min, j_max };
}

}  // namespace xtgeo::regsurf
