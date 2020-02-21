/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_create_box.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Create a simple shoebox grid, based on DX, DY rotation etc. I.e.
 *    similar to a Cube spec.
 *
 * ARGUMENTS:
 *    ncol..nlay     i     Dimensions
 *    p_coord_v     i/o    Coordinates (must be allocated in caller)
 *    p_zcorn_v     i/o    Z corners (must be allocated in called)
 *    p_actnum_v    i/o    ACTNUM (must be allocated in caller)
 *    xori..zori     i     Origins
 *    xinc..zinc     i     Increments
 *    option         i     0: input is lower left (top) corner node; 1 input
 *                         lower left (top) cell center
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void. Pointers to arrays are updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_from_cube (
                      int ncol,
                      int nrow,
                      int nlay,
                      double *p_coord_v,
                      long ncoord,
                      double *p_zcorn_v,
                      long nzcorn,
                      int *p_actnum_v,
                      long nactnum,
                      double xori,
                      double yori,
                      double zori,
                      double xinc,
                      double yinc,
                      double zinc,
                      double rotation,
                      int yflip,
                      int option
                      )

{
    /* locals */
   int nn, iok, i, j, k;
   long ibc = 0, ibz = 0, iba=0;
   double xcoord=0.0, ycoord=0.0;

   logger_info(LI, FI, FU, "Making Grid3D from cube or shoebox spec");

   if (option == 1) {  /* input is cell center, not cell corner */
       double res[8];
       x_2d_rect_corners(xori, yori, xinc, yinc, rotation, res, XTGDEBUG);
       xori = res[6];
       yori = res[7];
       if (yflip == -1) {
           xori = res[0];
           yori = res[1];
       }
       zori = zori - 0.5 * zinc;
   }

   /* make coord... */

   for (j = 1; j <= nrow + 1; j++) {
       for (i = 1; i <= ncol + 1; i++) {

           iok = cube_xy_from_ij(i, j, &xcoord, &ycoord, xori, xinc, yori,
                                 yinc, ncol + 1, nrow + 1, yflip, rotation, 0);

           if (iok != 0) logger_critical(LI, FI, FU, "Bug in %s", FU);

           p_coord_v[ibc++] = xcoord;
           p_coord_v[ibc++] = ycoord;
           p_coord_v[ibc++] = zori;

           p_coord_v[ibc++] = xcoord;
           p_coord_v[ibc++] = ycoord;
           p_coord_v[ibc++] = zori + zinc * (nlay + 1);
       }
   }

   /* make zcorn and actnum... */

   double zlevel = zori;

   for (k = 1; k <= nlay + 1; k++) {
       for (j = 1; j <= nrow; j++) {
           for (i = 1; i <= ncol; i++) {

               for (nn = 0; nn < 4; nn++) p_zcorn_v[ibz++] = zlevel;

               if (k <= nlay) p_actnum_v[iba++] = 1;

           }
       }
       zlevel += zinc;
   }

}
