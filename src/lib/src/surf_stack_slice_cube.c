/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_stack_slice_cube.c
 *
 * DESCRIPTION:
 *    Slicing a stack of surfaces and return cube values
 *
 *    In this version, the surface and cube share geometry, which makes calculations
 *    way much simpler and probably also much faster
 *
 * ARGUMENTS:
 *    ncol, nrow...  i     Cube dimensions and relevant increments
 *    czori, czinc   i     Cube settings
 *    cubevalsv      i     Cube values
 *    stack         i/o    stacked map array stack with Z values
 *    rmask          i     stacked map array stack with masks
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
surf_stack_slice_cube(int ncol,
                      int nrow,
                      int nlay,
                      int nstack,
                      double czori,
                      double czinc,
                      float *cubevalsv,
                      double **stack,
                      bool **rmask,
                      int optnearest,
                      int optmask)

{

    double zd[2];
    double czvals[2];

    int icol, jcol;
    for (icol = 1; icol <= ncol; icol++) {
        for (jcol = 1; jcol <= nrow; jcol++) {

            int nd;
            for (nd = 0; nd < nstack; nd++) {

                long icmap = x_ijk2ic(icol, jcol, 1, ncol, nrow, 1, 0);
                if (icmap < 0) {
                    throw_exception("Loop through surface gave index outside boundary "
                                    "in surf_stack_slice_cube");
                    return EXIT_FAILURE;
                }

                if (rmask[icmap][nd] != 0)
                    continue;

                double zval = stack[icmap][nd];
                if (zval > UNDEF_LIMIT)
                    continue;

                // find vertical index of node right above
                int k1 = (int)((zval - czori) / czinc);

                if (k1 < 0 || k1 > (nlay - 1)) {
                    stack[icmap][nd] = UNDEF;
                    rmask[icmap][nd] = 0;
                    continue;
                }

                int k2 = k1 + 1;

                // end cases
                if (k1 == 0 && zval < czori)
                    k2 = k1;
                if (k1 == nlay - 1)
                    k2 = k1;

                long icc1, icc2;
                icc1 = x_ijk2ic(icol, jcol, k1 + 1, ncol, nrow, nlay, 0);  // yes k+1
                icc2 = x_ijk2ic(icol, jcol, k2 + 1, ncol, nrow, nlay, 0);
                if (icc1 < 0 || icc2 < 0) {
                    throw_exception(
                      "Index outside boundary in surf_stack_slice_cube, no k+1");
                    return EXIT_FAILURE;
                }
                czvals[0] = cubevalsv[icc1];
                czvals[1] = cubevalsv[icc2];
                zd[0] = czori + k1 * czinc;
                zd[1] = czori + k2 * czinc;

                // interpolate, either with nearest sample or linear dependent on
                // optnearest
                stack[icmap][nd] = x_vector_linint1d(zval, zd, czvals, 2, optnearest);

                if (stack[icmap][nd] > UNDEF_LIMIT && optmask == 1)
                    rmask[icmap][nd] = 1;
            }
        }
    }

    return EXIT_SUCCESS;
}
