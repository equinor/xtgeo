/*
 *******************************************************************************
 *
 * Export Irap ascii map (with rotation)
 *
 *******************************************************************************
 */

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
 *    filename       i     File name, character string
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
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
                           char *filename,
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
                           int option,
                           int debug
                           )
{

    /* local declarations */
    int     i, j, ic, nn, fcode;
    float   myfloat, xmax, ymax;

    char    s[24]="surf_export_irap_ascii";

    FILE    *fc;

    xtgverbose(debug);
    xtg_speak(s,1,"Write IRAP ascii map file ...",s);

    xtg_speak(s,2,"Entering %s",s);

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

    fc = fopen(filename,"wb");

    if (fc == NULL) {
        xtg_warn(s, 0, "Some thing is wrong with requested filename <%s>",
                 filename);
        xtg_error(s, "Could be: Non existing folder, wrong permissions ? ..."
                  " anyway: STOP!", s);
        return -9;
    }

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

    fclose(fc);
    return EXIT_SUCCESS;

}
