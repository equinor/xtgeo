/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_slice_cube_v2.c
 *
 * DESCRIPTION:
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

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
surf_slice_cube_v2(int ncol,
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
                   mbool *maskv,  // *maskv,
                   long nmask,
                   int optnearest,
                   int optmask)

{

    // define cube Z depth array which is constant
    double *zd, *czvals;

    int nzlay = nlay + 2;  // add ends

    zd = calloc(nzlay, sizeof(double));
    czvals = calloc(nzlay, sizeof(double));

    zd[0] = czori - 0.5 * czinc;
    zd[nzlay - 1] = zd[0] + nlay * czinc;

    int kcol;
    for (kcol = 1; kcol <= nlay; kcol++)
        zd[kcol] = czori + (kcol - 1) * czinc;

    int icol, jcol;
    for (icol = 1; icol <= ncol; icol++) {
        for (jcol = 1; jcol <= nrow; jcol++) {

            long icmap = x_ijk2ic(icol, jcol, 1, ncol, nrow, 1, 0);
            if (icmap < 0) {
                free(zd);
                free(czvals);
                throw_exception("Loop through surface gave index outside boundary in "
                                "surf_slice_cube_v2");
                return EXIT_FAILURE;
            }

            double zval = zslicev[icmap];

            if (maskv[icmap] != 0 || zval < zd[0] || zval > zd[nzlay - 1])
                continue;

            for (kcol = 1; kcol <= nlay; kcol++) {
                long iccube = x_ijk2ic(icol, jcol, kcol, ncol, nrow, nlay, 0);
                if (iccube < 0) {
                    free(zd);
                    free(czvals);
                    throw_exception(
                      "Loop gave index outside boundary in surf_slice_cube_v2");
                    return EXIT_FAILURE;
                }
                czvals[kcol] = cubevalsv[iccube];
            }
            czvals[0] = czvals[1];
            czvals[nzlay - 1] = czvals[nzlay - 2];

            // interpolate, either with nearest sample or linear dependetn on optnearest
            surfsv[icmap] = x_vector_linint1d(zval, zd, czvals, nzlay, optnearest);

            if (surfsv[icmap] > UNDEF_LIMIT && optmask == 1)
                maskv[icmap] = 1;
        }
    }

    free(zd);
    free(czvals);

    return EXIT_SUCCESS;
}
