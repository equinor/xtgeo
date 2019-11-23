/*
 *******************************************************************************
 *
 * Export Irap ascii map (with rotation)
 *
 *******************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    surf_export_irap_ascii.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export a map on Irap ascii format.
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
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    option         i     Options flag for later usage
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    Issue: The surf_* routines in XTGeo will include rotation, and origins
 *           (not xmin etc ) and steps are used to define the map extent.
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */
int surf_export_irap_ascii(
                           FILE *fc,
                           int mx,
                           int my,
                           double xori,
                           double yori,
                           double xinc,
                           double yinc,
                           double rot,
                           double *p_map_v,
                           long mxy,
                           double zmin,
                           double zmax,
                           int option
                           )
{

    /* local declarations */
    int     i, j, ic, nn, fcode;
    float   myfloat, xmax, ymax;

    logger_info(__LINE__, "Write IRAP ascii map file ... (%s)", __FUNCTION__);

    /*
     * Do some computation first, to find best format
     * -------------------------------------------------------------------------
     */
    if (zmin > -10 && zmax < 10) {
        fcode = 1;
    }
    else{
        fcode = 2;
    }

    xmax = xori + (mx - 1) * xinc;
    ymax = yori + (my - 1) * yinc;

    /*
     * WRITE HEADER
     * -------------------------------------------------------------------------
     * The ascii header is
     * ID MY XINC YINC
     * XMIN XMAX YMIN YMAX     # note these are 'as if nonrotation!'
     * MX ROT X0ORI Y0ORI
     * 0 0 0 0 0 0 0
     * -------------------------------------------------------------------------
     */

    fprintf(fc, "%d %d %lf %lf\n", -996, my, xinc, yinc);
    fprintf(fc, "%lf %f %lf %f\n", xori, xmax, yori, ymax);
    fprintf(fc, "%d %lf %lf %lf\n", mx, rot, xori, yori);
    fprintf(fc, "0 0 0 0 0 0 0\n");

    nn = 0;
    /* export in F order */
    for (j=1; j<=my; j++) {
	for (i=1; i<=mx; i++) {

            /* C order input */
            ic = x_ijk2ic(i, j, 1, mx, my, 1, 0);
            myfloat = p_map_v[ic];

            if (myfloat > UNDEF_MAP_LIMIT) myfloat = UNDEF_MAP_IRAP;

            if (fcode == 1) {
                fprintf(fc, " %.7f", myfloat);
            }
            else{
                fprintf(fc, " %.4f", myfloat);
            }

            nn++;

            if (nn > 5) {
                fprintf(fc, "\n");
                nn = 0;
            }
	}
    }
    fprintf(fc, "\n");

    return EXIT_SUCCESS;

}
