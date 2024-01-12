/*
****************************************************************************************
*
* NAME:
*    grd3d_get_vtk_grid_arrays.c
*
* DESCRIPTION:
*    Get cell corners on VTK / pyvista array format. The format is designed for
*    ExplicitStructuredGrid in pyvista which is described here:
*    https://docs.pyvista.org/api/core/_autosummary/pyvista.ExplicitStructuredGrid.html
*
*    The grid corners are sampled along each direction where I_DIR are looping
*    fastest, then J_DIR and then K_DIR i.e. (aka F order).
*
* ARGUMENTS:
*    ncol, nrow, nlay     i     Dimensions
*    coordsv              i     Coordinate vector (xtgeo fmt)
*    zcornsv              i     ZCORN vector (xtgeo fmt)
*    xarr .. zarr         o     Single vectors for all X Y Z
*
* RETURNS:
*    Update pointers xarr, yarr, zarr
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    CF XTGeo's LICENSE
***************************************************************************************
*/
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

void
grdcp3d_get_vtk_grid_arrays(long ncol,
                            long nrow,
                            long nlay,

                            double *coordsv,
                            long ncoordin,
                            float *zcornsv,
                            long nlaycornin,

                            double *xarr,
                            long nxarr,
                            double *yarr,
                            long nyarr,
                            double *zarr,
                            long nzarr)

{
    double crs[24];
    long i, j, k, ic;

    long ib = 0;

    double *ccr;

    ccr = calloc(ncol * nrow * nlay * 24, sizeof(double));

    // collect first here to avoid multiple recomputing below
    for (k = 0; k < nlay; k++) {
        for (j = 0; j < nrow; j++) {
            for (i = 0; i < ncol; i++) {
                grdcp3d_corners(i, j, k, ncol, nrow, nlay, coordsv, 0, zcornsv, 0, crs);
                for (ic = 0; ic < 24; ic++)
                    ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + ic] = crs[ic];
            }
        }
    }

    for (k = 0; k < nlay; k++) {
        // top
        for (j = 0; j < nrow; j++) {
            for (i = 0; i < ncol; i++) {
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 0];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 1];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 2];
                ib++;
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 3];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 4];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 5];
                ib++;
            }
            for (i = 0; i < ncol; i++) {
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 6];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 7];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 8];
                ib++;
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 9];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 10];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 11];
                ib++;
            }
        }
        // base
        for (j = 0; j < nrow; j++) {
            for (i = 0; i < ncol; i++) {
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 12];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 13];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 14];
                ib++;
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 15];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 16];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 17];
                ib++;
            }
            for (i = 0; i < ncol; i++) {
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 18];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 19];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 20];
                ib++;
                xarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 21];
                yarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 22];
                zarr[ib] = ccr[i * nrow * nlay * 24 + j * nlay * 24 + k * 24 + 23];
                ib++;
            }
        }
    }
    free(ccr);
}
