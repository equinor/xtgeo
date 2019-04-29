/*
 ******************************************************************************
 *
 * NAME:
 *    cube_swapaxes.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Do swapping of cube axes; will switch from a left handed system
 *    to a right handed system (depth pos. down in both cases). Swapping axes).
 *    Origo will be the same,  but xinc, yinc, nx, ny will be swapped,
 *    and rotation updated.
 *
 * ARGUMENTS:
 *    nx...nz       i/o    Cube dimensions
 *    yflip         i/o    Cube YFLIP index (pointer, will be updated)
 *    xori ...      i/o    Cube origin and increments
 *    rotation      i/o    Cube rotation
 *    p_val_v        i     1D Array of cube values of ncx*ncy*ncz size
 *    p_traceid_v    i     1D Array of trace id values of ncx*ncy size
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


int cube_swapaxes (
                   int *nx,
                   int *ny,
                   int nz,
                   int *yflip,
                   double xori,
                   double *xinc,
                   double yori,
                   double *yinc,
                   double *rotation,
                   float *p_val_v,
                   long nval,
                   int *p_traceid_v,
                   long nval_traceid,
                   int option,
                   int debug
                   )
{

    /* locals */
    char s[24]="cube_swapaxes";
    int i, j, k, flip, nnx, nny;
    long nxyz, nxy, ib1, ib2;
    float *tmp1 = NULL;
    int *tmp2 = NULL;
    double xxinc, yyinc, rot;

    xtgverbose(debug);
    if (debug > 2) xtg_speak(s, 3, "Entering routine %s", s);

    nnx = *nx;
    nny = *ny;

    nxyz = (long)nnx * (long)nny * (long)nz;
    nxy = (long)nnx * (long)nny;

    /* intermediate array */
    xtg_speak(s, 2, "Allocate..");
    tmp1 = calloc(nxyz, sizeof(float));
    tmp2 = calloc(nxy, sizeof(int));

    flip = *yflip;

    xtg_speak(s, 2, "Swap...");
    ib2 = 0;
    /* reverse J I by looping J before I*/
    for (j = 1; j <= nny; j++) {
        for (i = 1; i <= nnx; i++) {
            for (k = 1; k <= nz; k++) {
                ib1 = x_ijk2ic(i, j, k, nnx, nny, nz, 0);
                ib2 = x_ijk2ic(j, i, k, nny, nnx, nz, 0);
                tmp1[ib2] = p_val_v[ib1];
            }
        }
    }

    for (i = 0; i < nxyz; i++) {
        p_val_v[i] = tmp1[i];
    }

    for (j = 1; j <= nny; j++) {
        for (i = 1; i <= nnx; i++) {
            ib1 = x_ijk2ic(i, j, 1, nnx, nny, 1, 0);
            ib2 = x_ijk2ic(j, i, 1, nny, nnx, 1, 0);
            tmp2[ib2] = p_traceid_v[ib1];
        }
    }

    for (i = 0; i < nxy; i++) {
        p_traceid_v[i] = tmp2[i];
    }

    xtg_speak(s, 2, "Swap... done");

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

    free(tmp1);
    free(tmp2);

    return EXIT_SUCCESS;
}
