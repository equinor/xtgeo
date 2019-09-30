/*
 ******************************************************************************
 *
 * Convert from Eclipse ZCORN format to XTGeo simplified zcorn
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_zcorn_convert.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Eclipse style ZCORN to XTGeo style p_zcorn_v
 *
 * ARGUMENTS:
 *    nx, ny, nc     i     Dimensions
 *    zcorn          i     ZCORN as input from Eclipse
 *    p_coord_v      o     XTgeo's ZCORN repr
 *    option         i     Options flag for later usage
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ******************************************************************************
 */



void grd3d_zcorn_convert (
                          int nx,
                          int ny,
                          int nz,
                          float *zcorn,
                          double *p_zcorn_v,
                          int option
                          )
{

    int ibb = 0;
    int ibz = 0;
    int kzread = 0;
    float fvalue1;
    float fvalue2;
    int kk = 0, kz, ix, jy;

    for (kz = 1; kz <= 2 * nz; kz++) {
        if (kzread == 0) {
            kzread = 1;
        }
        else{
            kzread = 0;
        }

        if (kz == 2*nz && kzread == 0) kzread=1;

        if (kzread == 1) {
            kk += 1;
        }
        for (jy = 1; jy <= ny; jy++) {
            /* "left" cell margin */
            for (ix = 1; ix <= nx; ix++) {
                fvalue1 = zcorn[ibz++];
                fvalue2 = zcorn[ibz++];

                ibb = x_ijk2ib(ix, jy, kk, nx, ny, nz+1, 0);
                if (kzread == 1) {
                    p_zcorn_v[4*ibb+1*1-1]=fvalue1;
                    p_zcorn_v[4*ibb+1*2-1]=fvalue2;
                }
            }
            /* "right" cell margin */
            for (ix=1; ix<=nx; ix++) {
                fvalue1 = zcorn[ibz++];
                fvalue2 = zcorn[ibz++];

                ibb = x_ijk2ib(ix, jy, kk, nx, ny, nz + 1, 0);
                if (kzread == 1) {
                    p_zcorn_v[4 * ibb + 1 * 3 - 1] = fvalue1;
                    p_zcorn_v[4 * ibb + 1 * 4 - 1] = fvalue2;
                }
            }
        }
    }
}
