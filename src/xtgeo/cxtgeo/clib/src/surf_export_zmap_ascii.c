/*
 ******************************************************************************
 *
 * Export ZMAP plus ascii map (no rotation)
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    surf_export_zmap_ascii.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export a map to zmap plus ascii. Note the map must be unrotated!
 *    The spec is according to
 *    http://lists.osgeo.org/pipermail/gdal-dev/2011-June/029173.html
 *
 * ARGUMENTS:
 *    filename       i     File name, character string
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
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
 ******************************************************************************
 */
int surf_export_zmap_ascii(
                           char *filename,
                           int mx,
                           int my,
                           double xori,
                           double yori,
                           double xinc,
                           double yinc,
                           double *p_map_v,
                           long mxy,
                           double zmin,
                           double zmax,
                           int option,
                           int debug
                           )
{

    /* local declarations */
    int     i, j, ib, nn, fcode;
    float   myfloat, xmax, ymax;

    char    s[24]="surf_export_zmap_ascii";

    FILE    *fc;

    xtgverbose(debug);
    xtg_speak(s,1,"Write ZMAP plus ascii map file ...",s);

    xtg_speak(s,2,"Entering %s",s);

    /*
     * Do some computation first, to find best format
     * ------------------------------------------------------------------------
     */
    if (zmin > -10 && zmax < 10) {
        fcode = 1;
    }
    else{
        fcode = 2;
    }

    xmax = xori + (mx - 1) * xinc;
    ymax = yori + (my - 1) * yinc;


    fc = fopen(filename,"wb");

    if (fc == NULL) {
        xtg_warn(s, 0, "Some thing is wrong with requested filename <%s>",
                 filename);
        xtg_error(s, "Could be: Non existing folder, wrong permissions ? ..."
                  " anyway: STOP!", s);
        return -9;
    }

    /* header */
    fprintf(fc, "! Export from XTGeo\n");
    fprintf(fc, "@ GRIDFILE, GRID, 5\n");
    fprintf(fc, "15, %f,  , 4, 1\n", UNDEF_MAP_ZMAP);
    fprintf(fc, "%d, %d, %lf, %lf, %lf, %lf\n",
            my, mx, xori, xmax, yori, ymax);
    fprintf(fc, "0.0, 0.0, 0.0\n");
    fprintf(fc, "@\n");
    ib = 0;
    nn = 0;

    /* data, the format start in upper left corner and goes fastest along
       the y axis */

    for (i=1; i<=mx; i++) {

        if (nn != 0) {
            fprintf(fc, "\n");
            nn = 0;
        }

        for (j=my; j>=1; j--) {

            ib = x_ijk2ib(i, j, 1, mx, my, 1, 0);
	    myfloat = p_map_v[ib];

            if (myfloat > UNDEF_MAP_LIMIT) myfloat = UNDEF_MAP_ZMAP;

            if (fcode == 1) {
                fprintf(fc, " %15.4f", myfloat);
            }
            else{
                fprintf(fc, " %15.8f", myfloat);
            }
            nn++;

            if (nn > 4) {
                fprintf(fc, "\n");
                nn = 0;
            }
	}
    }
    fprintf(fc, "\n");

    fclose(fc);

    return EXIT_SUCCESS;

}
