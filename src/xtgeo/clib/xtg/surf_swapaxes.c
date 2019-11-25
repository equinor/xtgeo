/*
 ******************************************************************************
 *
 * NAME:
 *    surf_swapaxes.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Do swapping of surface axes; will switch from a left handed system
 *    to a right handed system (depth pos. down in both cases) or back.
 *    Origo will be the same,  but xinc, yinc, nx, ny will be swapped,
 *    and rotation updated.
 *
 * ARGUMENTS:
 *    nx, ny        i/o    Surf dimensions
 *    yflip         i/o    Surf YFLIP index (pointer, will be updated)
 *    xori ...      i/o    Surf origin and increments
 *    rotation      i/o    Surf rotation
 *    p_map_v        i     1D Array of surf values of ncx*ncy size
 *    option         i     For future use
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    - Robust?
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"


int surf_swapaxes (
                   int *nx,
                   int *ny,
                   int *yflip,
                   double xori,
                   double *xinc,
                   double yori,
                   double *yinc,
                   double *rotation,
                   double *p_map_v,
                   long nval,
                   int option,
                   int debug
                   )
{

    /* locals */
    char s[24]="surf_swapaxes";
    int i, j, flip, nnx, nny;
    long nxyz, ib1, ib2;
    float *tmp = NULL;
    double xxinc, yyinc, rot;

    xtgverbose(debug);

    nnx = *nx;
    nny = *ny;

    nxyz = (long)nnx * (long)nny;

    /* intermediate array */
    xtg_speak(s, 2, "Allocate..");
    tmp = calloc(nxyz, sizeof(double));

    flip = *yflip;

    xtg_speak(s, 2, "Swap...");

    ib2 = 0;
    for (j=1; j<=nny; j++) {
        for (i=1; i<=nnx; i++) {
            ib1 = x_ijk2ic(i, j, 1, nnx, nny, 1, 0);
            ib2 = x_ijk2ic(j, i, 1, nny, nnx, 1, 0);

            tmp[ib2] = p_map_v[ib1];
        }
    }
    xtg_speak(s, 2, "Swap... done");

    for (i=0; i<nxyz; i++) {
        p_map_v[i] = tmp[i];
    }

    nnx = *ny;
    nny = *nx;
    xxinc = *yinc;
    yyinc = *xinc;

    *nx = nnx;
    *ny = nny;
    *xinc = xxinc;
    *yinc = yyinc;

    rot = *rotation;
    rot = rot + flip*90;
    if (rot >= 360.0) rot = rot - 360;
    if (rot < 0.0) rot = rot + 360;

    *yflip = flip * -1;
    *rotation = rot;

    free(tmp);

    return EXIT_SUCCESS;
}
