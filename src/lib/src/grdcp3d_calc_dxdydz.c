/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_dxdy.c
 *
 * DESCRIPTION:
 *    Computes the DY and DX per cell.
 *
 *    The DX/DY of a cell is the average of edge length in the x/y direction.
 *
 *    The output vector dx to be initialized to zero and have size nx*ny*nz.
 *    coordsv should have size (nx+1)*(ny+1)*6.
 *    zcornsv should have size (nx+1)*(ny+1)*(nz+1)*4.
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     Dimensions
 *    coordsv        i     Coordinates (with size)
 *    zcornsv        i     Z corners (with size)
 *    dx/dy/dz       i/o   Array to be updated
 *    m              i     Metric to use for distance (function pointer of type metric)
 *
 * RETURNS:
 *    Success (0) or failure. Pointers to arrays are updated
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <math.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>

/**
 * The planar map of a corner line maps height to coordinates in the plane,
 * i.e.
 *
 * f(z) = (bot_x + (z-bot_z)*slope_x, (y - bot_y) * (z-bot_z)*slope_y)
 *
 */
typedef struct PlanarMap
{
    double bot_x;
    double bot_y;
    double bot_z;
    double slope_x;
    double slope_y;
} PlanarMap;

/**
 * Gives the planar map of the ith coordinate line.
 */
inline int
pm_from_corner_line(const double *const coordsv, const size_t i, PlanarMap *const pm)
{

    size_t ind = i * 6;
    pm->bot_x = coordsv[ind + 0];
    pm->bot_y = coordsv[ind + 1];
    pm->bot_z = coordsv[ind + 2];
    double top_x = coordsv[ind + 3];
    double top_y = coordsv[ind + 4];
    double total_diff_z = coordsv[ind + 5] - pm->bot_z;
    if (fabs(total_diff_z) < 1e-10) {
        if (fabs(top_x - pm->bot_x) >= 1e-3 || fabs(top_y - pm->bot_y) >= 1e-3) {
            throw_exception("Grid has near zero height, but different top and bottom.");
            return EXIT_FAILURE;
        }
        pm->slope_x = 0.0;
        pm->slope_y = 0.0;
    } else {
        pm->slope_x = (top_x - pm->bot_x) / total_diff_z;
        pm->slope_y = (top_y - pm->bot_y) / total_diff_z;
    }
    return EXIT_SUCCESS;
}

/**
 * evaluates the planar map at the given z value, ie. gives
 * the x,y coordinate of the corresponding corner line at
 * the given height.
 */
inline int
pm_evaluate(const PlanarMap *const pm, const double z, double *x, double *y)
{
    double diff_z = z - pm->bot_z;
    *x = pm->bot_x + diff_z * pm->slope_x;
    *y = pm->bot_y + diff_z * pm->slope_y;
    return EXIT_SUCCESS;
}

double
euclid_length(const double x1,
              const double y1,
              const double z1,
              const double x2,
              const double y2,
              const double z2)
{
    return sqrt(powf(x2 - x1, 2) + powf(y2 - y1, 2) + powf(z2 - z1, 2));
}

double
horizontal_length(const double x1,
                  const double y1,
                  const double z1,
                  const double x2,
                  const double y2,
                  const double z2)
{
    return sqrt(powf(x2 - x1, 2) + powf(y2 - y1, 2));
}

double
east_west_vertical_length(const double x1,
                          const double y1,
                          const double z1,
                          const double x2,
                          const double y2,
                          const double z2)
{
    return sqrt(powf(x2 - x1, 2) + powf(z2 - z1, 2));
}

double
north_south_vertical_length(const double x1,
                            const double y1,
                            const double z1,
                            const double x2,
                            const double y2,
                            const double z2)
{
    return sqrt(powf(y2 - y1, 2) + powf(z2 - z1, 2));
}

double
x_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2)
{
    return x2 - x1;
}

double
y_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2)
{
    return y2 - y1;
}

double
z_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2)
{
    return z2 - z1;
}

int
grdcp3d_calc_dy(int nx,
                int ny,
                int nz,
                double *coordsv,
                long ncoord,
                double *zcornsv,
                long nzcorn,
                double *dy,
                long ndy,
                metric m)

{
    if (ncoord != (nx + 1) * (ny + 1) * 6) {
        throw_exception("Incorrect size of coordsv.");
        return EXIT_FAILURE;
    }
    if (nzcorn != (nx + 1) * (ny + 1) * (nz + 1) * 4) {
        throw_exception("Incorrect size of zcornsv.");
        return EXIT_FAILURE;
    }
    if (ndy != nx * ny * nz) {
        throw_exception("Incorrect size of dx.");
        return EXIT_FAILURE;
    }
    if (ndy <= 0) {
        return EXIT_SUCCESS;
    }

    // The following algorithm goes through all subsequent pairs of corner
    // lines in y direction, calculates the length of the lines between the
    // their corners and adds it contribution to the average of the cells
    // it is an edge of.

    // The initial first corner line in the pair
    PlanarMap pm1;
    if (pm_from_corner_line(coordsv, 0, &pm1) == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }
    size_t corner_line1_start = 0;

    size_t num_plane = (nx + 1) * (ny + 1);
    for (size_t pair_index = 1, cell_plane_index = 0; pair_index < num_plane;
         pair_index++) {

        // The next corner line in y direction (second in the pair)
        PlanarMap pm2;
        if (pm_from_corner_line(coordsv, pair_index, &pm2) == EXIT_FAILURE) {
            return EXIT_FAILURE;
        }

        // Corresponding to the corner lines in the pair, there are two pillars
        // of cells. Those having the edges of that pair to the west and those
        // having them to the east (west/east is x direction). We go through
        // each layer in that pillar/line pair.
        size_t west_pillar_start = nz * cell_plane_index;
        size_t corner_line1_end = corner_line1_start + (nz + 1);
        size_t corner_line2_start = (nz + 1) * pair_index;

        // Skip the pair of corners going from the end of a row
        // to the start of the next row
        if (pair_index % (ny + 1) != 0) {

            char west_in_bounds = pair_index >= ny + 1;
            char east_in_bounds = pair_index < num_plane - (ny + 1);

            for (size_t j = corner_line1_start, k = corner_line2_start,
                        jj = west_pillar_start;
                 j < corner_line1_end; j += 1, k += 1, jj += 1) {
                // In each layer there is the east and west (x direction)
                // z values (west is l=0, east is l=1)
                for (size_t l = 0; l < 2; l += 1) {

                    // calculate the x and y values for
                    // this layer and east/west in both
                    // corner lines of the pair.
                    double z1 = zcornsv[4 * j + l + 2];
                    double x1, y1;
                    if (pm_evaluate(&pm1, z1, &x1, &y1) == EXIT_FAILURE) {
                        return EXIT_FAILURE;
                    }

                    double z2 = zcornsv[4 * k + l];
                    double x2, y2;
                    if (pm_evaluate(&pm2, z2, &x2, &y2) == EXIT_FAILURE) {
                        return EXIT_FAILURE;
                    }

                    // The corresponding line contributes 1/4 of its length to the
                    // dx value of up to two surrounding cells it is an edge of
                    double vector_len = 0.25 * m(x1, y1, z1, x2, y2, z2);

                    char at_top = j == (corner_line1_end - 1);
                    char at_bottom = j == corner_line1_start;

                    if (l == 0 && west_in_bounds) {
                        // the cell to the west and above the edge
                        if (!at_top) {
                            dy[jj - (ny * nz)] += vector_len;
                        }

                        // the cell to the west and below the edge
                        if (!at_bottom) {
                            dy[(jj - (ny * nz)) - 1] += vector_len;
                        }
                    } else if (l == 1 && east_in_bounds) {
                        // the cell to the east and above the edge
                        if (!at_top) {
                            dy[jj] += vector_len;
                        }

                        // index of the cell to the easts and below the edge
                        if (!at_bottom) {
                            dy[jj - 1] += vector_len;
                        }
                    }
                }
            }
            // Since there are one more corner line in the y direction then there
            // are cells we have to keep a separate index for cells
            cell_plane_index++;
        }
        // update the first corner line map and start
        pm1 = pm2;
        corner_line1_start = corner_line2_start;
    }
    return EXIT_SUCCESS;
}

int
grdcp3d_calc_dx(int nx,
                int ny,
                int nz,
                double *coordsv,
                long ncoord,
                double *zcornsv,
                long nzcorn,
                double *dx,
                long ndx,
                metric m)

{
    if (ncoord != (nx + 1) * (ny + 1) * 6) {
        throw_exception("Incorrect size of coordsv.");
        return EXIT_FAILURE;
    }
    if (nzcorn != (nx + 1) * (ny + 1) * (nz + 1) * 4) {
        throw_exception("Incorrect size of zcornsv.");
        return EXIT_FAILURE;
    }
    if (ndx != nx * ny * nz) {
        throw_exception("Incorrect size of dx.");
        return EXIT_FAILURE;
    }
    if (ndx <= 0) {
        return EXIT_SUCCESS;
    }

    // The following algorithm goes through all subsequent pairs of corner
    // lines in x direction, calculates the length of the lines between the
    // their corners and adds it contribution to the average of the cells
    // it is an edge of.

    size_t num_line_pairs = nx * (ny + 1);

    for (size_t pair_index = 0, cell_plane_index = 0; pair_index < num_line_pairs;
         pair_index++) {
        if (pair_index % (ny + 1) != 0) {
            // There is one more line in the y direction then there
            // are cells, so we have to keep two indices.
            cell_plane_index++;
        }

        // The first corner line in the pair
        PlanarMap pm1;
        if (pm_from_corner_line(coordsv, pair_index, &pm1) == EXIT_FAILURE) {
            return EXIT_FAILURE;
        }

        // The next corner line in x direction (second in the pair)
        PlanarMap pm2;
        if (pm_from_corner_line(coordsv, pair_index + ny + 1, &pm2) == EXIT_FAILURE) {
            return EXIT_FAILURE;
        }

        // Corresponding to the corner lines in the pair, there are two pillars
        // of cells. Those having the edges of that pair to the north and those
        // having them to the south (north/south is y direction). We go through
        // each layer in that pillar/line pair.
        size_t north_pillar_start = nz * cell_plane_index;
        size_t corner_line1_start = (nz + 1) * pair_index;
        size_t corner_line1_end = corner_line1_start + (nz + 1);
        size_t corner_line2_start = (nz + 1) * (pair_index + (ny + 1));

        char north_in_bounds = pair_index % (ny + 1) != ny;
        char south_in_bounds = pair_index % (ny + 1) != 0;

        for (size_t j = corner_line1_start, k = corner_line2_start,
                    jj = north_pillar_start;
             j < corner_line1_end; j += 1, k += 1, jj += 1) {
            // In each layer there is the north and south (y direction)
            // z values (south is l=0, north is l=2)
            for (size_t l = 0; l < 4; l += 2) {

                // calculate the x and y values for
                // this layer and north/south in both
                // corner lines of the pair.
                double z1 = zcornsv[4 * j + l + 1];
                double x1, y1;
                if (pm_evaluate(&pm1, z1, &x1, &y1) == EXIT_FAILURE) {
                    return EXIT_FAILURE;
                }

                double z2 = zcornsv[4 * k + l];
                double x2, y2;
                if (pm_evaluate(&pm2, z2, &x2, &y2) == EXIT_FAILURE) {
                    return EXIT_FAILURE;
                }

                // The corresponding line contributes 1/4 of its length to the
                // dx value of up to two surrounding cells it is an edge of
                double vector_len = 0.25 * m(x1, y1, z1, x2, y2, z2);

                char at_top = j == (corner_line1_end - 1);
                char at_bottom = j == corner_line1_start;

                if (l == 0 && south_in_bounds) {
                    // the cell to the south and above the edge
                    if (!at_top) {
                        dx[jj - nz] += vector_len;
                    }

                    // the cell to the south and below the edge
                    if (!at_bottom) {
                        dx[(jj - nz) - 1] += vector_len;
                    }
                } else if (l == 2 && north_in_bounds) {
                    // the cell to the south and above the edge
                    if (!at_top) {
                        dx[jj] += vector_len;
                    }

                    // index of the cell to the south and below the edge
                    if (!at_bottom) {
                        dx[jj - 1] += vector_len;
                    }
                }
            }
        }
    }
    return EXIT_SUCCESS;
}

int
grdcp3d_calc_dz(int nx,
                int ny,
                int nz,
                double *coordsv,
                long ncoord,
                double *zcornsv,
                long nzcorn,
                double *dx,
                long ndx,
                metric m)

{
    if (ncoord != (nx + 1) * (ny + 1) * 6) {
        throw_exception("Incorrect size of coordsv.");
        return EXIT_FAILURE;
    }
    if (nzcorn != (nx + 1) * (ny + 1) * (nz + 1) * 4) {
        throw_exception("Incorrect size of zcornsv.");
        return EXIT_FAILURE;
    }
    if (ndx != nx * ny * nz) {
        throw_exception("Incorrect size of dx.");
        return EXIT_FAILURE;
    }
    if (ndx <= 0) {
        return EXIT_SUCCESS;
    }

    // The following algorithm goes through all corner lines,
    // calculates coordinates of subsequent pairs of corners in
    // z direction, and adds the contribution of length average to
    // corresponding cells.

    size_t num_corner_lines = (nx + 1) * (ny + 1);

    for (size_t pair_index = 0, cell_plane_index = 0; pair_index < num_corner_lines;
         pair_index++) {
        if (pair_index % (ny + 1) != 0) {
            // There is one more line in the y direction then there
            // are cells, so we have to keep two indecies.
            cell_plane_index++;
        }

        PlanarMap pm;
        if (pm_from_corner_line(coordsv, pair_index, &pm) == EXIT_FAILURE) {
            return EXIT_FAILURE;
        }

        size_t north_east_pillar_start = nz * cell_plane_index;
        size_t corner_line_start = (nz + 1) * pair_index;
        size_t corner_line_end = corner_line_start + (nz + 1);

        char north_in_bounds = pair_index % (ny + 1) != ny;
        char south_in_bounds = pair_index % (ny + 1) != 0;
        char west_in_bounds = pair_index >= ny + 1;
        char east_in_bounds = pair_index < num_corner_lines - (ny + 1);

        for (size_t j = corner_line_start, jj = north_east_pillar_start;
             j < corner_line_end - 1; j += 1, jj += 1) {
            for (size_t l = 0; l < 4; l++) {

                double z1 = zcornsv[4 * j + l];
                double x1, y1;
                if (pm_evaluate(&pm, z1, &x1, &y1) == EXIT_FAILURE) {
                    return EXIT_FAILURE;
                }

                double z2 = zcornsv[4 * (j + 1) + l];
                double x2, y2;
                if (pm_evaluate(&pm, z2, &x2, &y2) == EXIT_FAILURE) {
                    return EXIT_FAILURE;
                }

                // The corresponding line contributes 1/4 of its length to the
                // dx value of one edge depending on l.
                double vector_len = 0.25 * m(x1, y1, z1, x2, y2, z2);

                if (l == 0 && south_in_bounds && west_in_bounds) {
                    dx[jj - nz * (ny + 1)] += vector_len;
                } else if (l == 1 && south_in_bounds && east_in_bounds) {
                    dx[jj - nz] += vector_len;
                } else if (l == 2 && north_in_bounds && west_in_bounds) {
                    dx[jj - nz * ny] += vector_len;
                } else if (l == 3 && north_in_bounds && east_in_bounds) {
                    dx[jj] += vector_len;
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
