/*
****************************************************************************************
*
* NAME:
*    surf_export_zmap_ascii.c
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
***************************************************************************************
*/
#include <stdio.h>
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

#include "logger.h"

int
surf_export_zmap_ascii(FILE *fc,
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
                       int option)
{

    /* local declarations */
    int i, j, nn, fcode;
    double node_value, xmax, ymax;

    logger_info(LI, FI, FU, "Write ZMAP plus ascii map file ... (%s)", FU);

    /*
     * Do some computation first, to find best format
     * ---------------------------------------------------------------------------------
     */

    /* number of decimal points */
    if (zmin > -10 && zmax < 10) {
        fcode = 4;
    } else {
        fcode = 8;
    }

    int nfrow = 5;
    if (my < nfrow) {
        nfrow = my;
    }

    xmax = xori + (mx - 1) * xinc;
    ymax = yori + (my - 1) * yinc;

    if (fc == NULL)
        return -1;
    /* header */
    fprintf(fc, "! Export from XTGeo (cxtgeo engine)\n");
    fprintf(fc, "@ GRIDFILE, GRID, %d\n", nfrow);
    fprintf(fc, "20, %f,  , %d, 1\n", UNDEF_MAP_ZMAP, fcode);
    fprintf(fc, "%d, %d, %lf, %lf, %lf, %lf\n", my, mx, xori, xmax, yori, ymax);
    fprintf(fc, "0.0, 0.0, 0.0\n");
    fprintf(fc, "@\n");
    nn = 0;

    /* data, the format start in upper left corner and goes fastest along the y axis
     * ---------------------------------------------------------------------------------
     */

    for (i = 1; i <= mx; i++) {

        if (nn != 0) {
            fprintf(fc, "\n");
            nn = 0;
        }

        for (j = my; j >= 1; j--) {

            long ic = x_ijk2ic(i, j, 1, mx, my, 1, 0);
            if (ic < 0) {
                throw_exception("Index outside boundary in surf_export_zmap_ascii");
                return EXIT_FAILURE;
            }

            node_value = p_map_v[ic];

            if (node_value > UNDEF_MAP_LIMIT)
                node_value = UNDEF_MAP_ZMAP;

            if (fcode == 4) {
                fprintf(fc, " %19.4f", node_value);
            } else {
                fprintf(fc, " %19.8f", node_value);
            }
            nn++;

            if (nn >= nfrow || j == 1) {
                fprintf(fc, "\n");
                nn = 0;
            }
        }
    }

    return EXIT_SUCCESS;
}
