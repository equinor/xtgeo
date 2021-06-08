
/*
****************************************************************************************
*
* NAME:
*    grd3d_conv_roxapi_grid.c
*
* DESCRIPTION:
*    Retrieve the grid
*
* ARGUMENTS:
*    nx, ny, nz     i     Dimensions
*    cact
*    crds           i     Input array flatened with 8 corners (* 3 = 24 xyz)
*                         for all cells, length = nx * ny * nz * 24
*    coordsv        o     Updated coord array
*    zcornsv        o     Updated zcorn array
*    actnumsv       o     Updated actnum array
*    debug          i     Debug level
*
* RETURNS:
*    Result vectors are updated
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_conv_roxapi_grid(int nx,
                       int ny,
                       int nz,
                       long nxyz,
                       int *cact,
                       long ncactin,
                       double *crds,
                       long ncrdsin,
                       double *coordsv,
                       long ncoordin,
                       double *zcornsv,
                       long nzcornin,
                       int *actnumsv,
                       long nactin)
{

    /* locals */
    double **cellcorners;  // note dynamic x[nxyz][24] gave segfaults
    long id, nc, ib, ibt, ibb, ia;
    int ic, i, j, k, ix, jy, kz;

    logger_info(LI, FI, FU, "Convert ROXAPI grid...");

    cellcorners = (double **)malloc(nxyz * sizeof(double *));

    for (ia = 0; ia < nxyz; ia++) {
        cellcorners[ia] = (double *)malloc(24 * sizeof(double));
    }

    /* convert crds* to cellcorners repr, C to Fortran order */

    nc = 0;
    id = 0;
    ia = 0;
    for (ix = 1; ix <= nx; ix++) {
        for (jy = 1; jy <= ny; jy++) {
            for (kz = 1; kz <= nz; kz++) {

                id = x_ijk2ib(ix, jy, kz, nx, ny, nz, 0);
                if (id < 0) {
                    for (ia = 0; ia < nxyz; ia++) {
                        free(cellcorners[ia]);
                    }
                    free(cellcorners);
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_conv_roxapi_grid");
                    return;
                }
                for (ic = 0; ic < 24; ic++) {
                    cellcorners[id][ic] = crds[nc++];
                }
                actnumsv[id] = cact[ia++];
            }
        }
    }

    /* extract the COORD lines for all cells */
    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {

            /* top: */
            k = 1;
            ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
            if (ib < 0) {
                for (ia = 0; ia < nxyz; ia++) {
                    free(cellcorners[ia]);
                }
                free(cellcorners);
                throw_exception("Loop through grid resulted in index outside boundary "
                                "in grd3d_conv_roxapi_grid");
                return;
            }
            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 0] = cellcorners[ib][0];
            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 1] = cellcorners[ib][1];
            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 2] = cellcorners[ib][2];

            if (i == nx) {
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 0] = cellcorners[ib][3];
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 1] = cellcorners[ib][4];
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 2] = cellcorners[ib][5];
            }
            if (j == ny) {
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 0] = cellcorners[ib][6];
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 1] = cellcorners[ib][7];
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 2] = cellcorners[ib][8];

                if (i == nx) {
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 0] = cellcorners[ib][9];
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 1] = cellcorners[ib][10];
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 2] = cellcorners[ib][11];
                }
            }

            /* base: */
            k = nz;
            ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
            if (ib < 0) {
                for (ia = 0; ia < nxyz; ia++) {
                    free(cellcorners[ia]);
                }
                free(cellcorners);
                throw_exception("Loop through grid resulted in index outside boundary "
                                "in grd3d_conv_roxapi_grid");
                return;
            }

            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 3] = cellcorners[ib][12];
            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 4] = cellcorners[ib][13];
            coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 5] = cellcorners[ib][14];

            if (i == nx) {
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 3] = cellcorners[ib][15];
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 4] = cellcorners[ib][16];
                coordsv[6 * ((j - 1) * (nx + 1) + i - 0) + 5] = cellcorners[ib][17];
            }
            if (j == ny) {
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 3] = cellcorners[ib][18];
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 4] = cellcorners[ib][19];
                coordsv[6 * ((j - 0) * (nx + 1) + i - 1) + 5] = cellcorners[ib][20];

                if (i == nx) {
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 3] = cellcorners[ib][21];
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 4] = cellcorners[ib][22];
                    coordsv[6 * ((j - 0) * (nx + 1) + i - 0) + 5] = cellcorners[ib][23];
                }
            }
        }
    }

    /* ZCORN values: */

    for (kz = 1; kz <= nz; kz++) {
        for (jy = 1; jy <= ny; jy++) {
            for (ix = 1; ix <= nx; ix++) {

                /* cell and cell below*/
                ibt = x_ijk2ib(ix, jy, kz, nx, ny, nz + 1, 0);
                ibb = x_ijk2ib(ix, jy, kz + 1, nx, ny, nz + 1, 0);
                ib = x_ijk2ib(ix, jy, kz, nx, ny, nz, 0);
                if (ibt < 0 || ibb < 0 || ib < 0) {
                    for (ia = 0; ia < nxyz; ia++) {
                        free(cellcorners[ia]);
                    }
                    free(cellcorners);
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_conv_roxapi_grid");
                    return;
                }

                zcornsv[4 * ibt + 1 * 1 - 1] = cellcorners[ib][2];
                zcornsv[4 * ibt + 1 * 2 - 1] = cellcorners[ib][5];
                zcornsv[4 * ibt + 1 * 3 - 1] = cellcorners[ib][8];
                zcornsv[4 * ibt + 1 * 4 - 1] = cellcorners[ib][11];

                zcornsv[4 * ibb + 1 * 1 - 1] = cellcorners[ib][14];
                zcornsv[4 * ibb + 1 * 2 - 1] = cellcorners[ib][17];
                zcornsv[4 * ibb + 1 * 3 - 1] = cellcorners[ib][20];
                zcornsv[4 * ibb + 1 * 4 - 1] = cellcorners[ib][23];
            }
        }
    }

    for (ia = 0; ia < nxyz; ia++) {
        free(cellcorners[ia]);
    }
    free(cellcorners);

    logger_info(LI, FI, FU, "Convert ROXAPI grid... done");
}
