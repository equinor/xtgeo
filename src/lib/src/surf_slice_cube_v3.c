/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_slice_cube_v3.c
 *
 * DESCRIPTION:
 *    As v2 but faster...
 *    Given a map and a cube, sample cube values to the map and return a
 *    map copy with cube values sampled.
 *
 *    In this version, the surface and cube share geometry, which makes calculations
 *    way much simpler and probably also much faster
 *
 * ARGUMENTS:
 *    ncol, nrow...  i     cube dimensions and relevant increments
 *    p_cubeval_v    i     1D Array of cube values of ncx*ncy*ncz size
 *    ncube          i     Length of cube array
 *    zslicev        i     map array with Z values
 *    nslice         i     Length of slice array
 *    surfsv        i/o    map to update
 *    nmap           i     Length of map array
 *    maskv         i/o    mask array for map, may be updated!
 *    nmap           i     Length of map array
 *    optnearest     i     If 1 use nerest node, else do interpolation aka trilinear
 *    optmask        i     If 1 then masked cells (undef) are made if cube is UNDEF
 *                         (surface outside cube will not occur in this case)
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    See XTGeo lisence
 *
 ***************************************************************************************
 */
#include <stdbool.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

int
surf_slice_cube_v3(int ncol,
                   int nrow,
                   int nlay,
                   double czori,
                   double czinc,
                   float *cubevalsv,
                   long ncube,
                   double *zslicev,
                   long nslice,
                   double *surfsv,
                   long nsurf,
                   bool *maskv,  // *maskv,
                   long nmask,
                   int optnearest,
                   int optmask)

{

    double zd[2];
    double czvals[2];

    logger_info(LI, FI, FU, "Enter %s", FU);

    int icol, jrow;
    for (icol = 1; icol <= ncol; icol++) {
        for (jrow = 1; jrow <= nrow; jrow++) {

            long icmap = x_ijk2ic(icol, jrow, 1, ncol, nrow, 1, 0);
            if (icmap < 0) {
                throw_exception("Error in surf_slice_cube_v3");
                return EXIT_FAILURE;
            }

            if (maskv[icmap] != 0)
                continue;

            double zval = zslicev[icmap];

            // find vertical index of node right above
            int k1 = (int)((zval - czori) / czinc);
            if (k1 < 0 || k1 > (nlay - 1)) {
                surfsv[icmap] = UNDEF;
                maskv[icmap] = 1;
                continue;
            }
            int k2 = k1 + 1;

            // end cases
            if (k1 == 0 && zval < czori)
                k2 = k1;
            if (k1 == nlay - 1)
                k2 = k1;

            long icc1, icc2;
            icc1 = x_ijk2ic(icol, jrow, k1 + 1, ncol, nrow, nlay, 0);
            icc2 = x_ijk2ic(icol, jrow, k2 + 1, ncol, nrow, nlay, 0);
            if (icc1 < 0 || icc2 < 0) {
                throw_exception("Index outside boundary in surf_slice_cube_v3");
                return EXIT_FAILURE;
            }
            czvals[0] = cubevalsv[icc1];
            czvals[1] = cubevalsv[icc2];
            zd[0] = czori + k1 * czinc;
            zd[1] = czori + k2 * czinc;

            // interpolate, either with nearest sample or linear dependent on optnearest
            surfsv[icmap] = x_vector_linint1d(zval, zd, czvals, 2, optnearest);

            if (surfsv[icmap] > UNDEF_LIMIT && optmask == 1)
                maskv[icmap] = 1;
        }
    }
    logger_info(LI, FI, FU, "Exit from %s", FU);

    return EXIT_SUCCESS;
}
