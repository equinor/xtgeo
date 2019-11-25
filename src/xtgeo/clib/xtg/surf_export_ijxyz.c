/*
****************************************************************************************
 *
 * Export to IJXYZ format (OW XYZ, with inline xline in two first columns)
 *
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_export_ijxyz.c
 *
 * DESCRIPTION:
 *    Export a map on DSG .map format. Columns with INL XL X Y Z, e.g.
 *
 * 1690 2758    535122.9879395069    6761613.488389527       1954.0
 * 1691 2758    535110.6627284092    6761627.618244775       1953.6038
 * 1692 2758    535098.3375173114    6761641.748100022       1953.7563
 * 1693 2758    535086.0123062138    6761655.87795527        1954.2279
 *
 *
 * ARGUMENTS:
 *    fc             i     File handle
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    yflip          i     yflip flag
 *    ilines         i     inlines array
 *    ncol           i     dimensions of inlines (shall be = MX)
 *    xlines         i     xlines array
 *    nrow           i     dimensions of xlines (shall be = MY)
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    mxy            i     dimensions of map values as 1D array
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    See XTGeo license
 ***************************************************************************************
 */
int surf_export_ijxyz(
                      FILE *fc,
                      int mx,
                      int my,
                      double xori,
                      double yori,
                      double xinc,
                      double yinc,
                      double rot,
                      int yflip,
                      int *ilines,
                      long ncol,
                      int *xlines,
                      long nrow,
                      double *p_map_v,
                      long mxy,
                      int option
                      )
{

    /* local declarations */
    int i, j, iok;
    double xv, yv, zv;

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Write OW style map file INLINE XLINE X Y Z (%s)", __FUNCTION__);

    /* export in INLINE running fastest order */
    for (j = 1; j <= my; j++) {
	for (i = 1; i <= mx; i++) {

            iok = surf_xyz_from_ij(i, j, &xv, &yv, &zv, xori, xinc, yori,
                                   yinc, mx, my, yflip, rot, p_map_v,
                                   nrow*ncol, 0);

            if (iok != 0) {
                logger_error(__LINE__, "Error from %s", __FUNCTION__);
                exit(313);
            }

            if (zv < UNDEF_MAP_LIMIT) {

                fprintf(fc, "%d\t%d\t%lf\t%lf\t%lf\n",
                        ilines[i-1], xlines[j-1], xv, yv, zv);

            }
        }
    }
    fprintf(fc, "\n");

    return EXIT_SUCCESS;

}
