/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_cellvol.c
 *
 * DESCRIPTION:
 *    Compute bulk volume of a cell. Tests shows that this is very close to what
 *    RMS will compute; almost identical
 *
 * ARGUMENTS:
 *    ncol,nrow,nlay   i     Grid dimensions nx ny nz
 *    coordsv          i     Grid Z coord for input
 *    zcornsv          i     Grid Z corners for input
 *    actnumsv         i     Actnum array
 *    cellvolsv        o     Array, cellvol as property
 *    option           i     0: do not compute for inactive cells (assign UNDEF)
 *
 *
 * RETURNS:
 *    Update pointer cellvolsv, _xtgformat=2
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf. XTGeo License
 ***************************************************************************************
 */

#include "libxtg.h"
#include "logger.h"

void
grdcp3d_cellvol(long ncol,
                long nrow,
                long nlay,
                double *coordsv,
                long ncoord,
                float *zcornsv,
                long nzcorn,
                int *actnumsv,
                long nact,
                double *cellvolsv,
                long ncell,
                int presision,
                int option)

{

    logger_info(LI, FI, FU, "Cell bulk volume...");
    double *corners = calloc(24, sizeof(double));

    long i, j, k;
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < nrow; j++) {
            for (k = 0; k < nlay; k++) {

                long ic = i * nrow * nlay + j * nlay + k;

                if (option == 0 && actnumsv[ic] == 0) {
                    cellvolsv[ic] = UNDEF;
                    continue;
                }

                grdcp3d_corners(i, j, k, ncol, nrow, nlay, coordsv, ncoord, zcornsv,
                                nzcorn, corners);

                cellvolsv[ic] = x_hexahedron_volume(corners, 24, presision);
            }
        }
    }

    free(corners);
    logger_info(LI, FI, FU, "Cell bulk volume... done");
}