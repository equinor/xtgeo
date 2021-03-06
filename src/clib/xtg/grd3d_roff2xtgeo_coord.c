/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_coord.c
 *
 * DESCRIPTION:
 *    Convert from ROFF internal spec to XTGeo spec for coordinate lines.
 *    Pillars (COORD lines). ROFF order them from base to top in C order, while XTGeo
 *    use the Eclipse style; top (X Y Z) to base (X Y Z) looping in F order.
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     NCOL, NROW, NLAY dimens
 *    *offset          i     Offsets in XYZ spesified in ROFF
 *    *scale           i     Scaling in XYZ spesified in ROFF
 *    p_cornerlines_v  i     Input cornerlines array ROFF fmt
 *    coordsv          o     Output cornerlines array XTGEO fmt
 *
 * RETURNS:
 *    Function: 0: upon success. Update pointers
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_roff2xtgeo_coord(int nx,
                       int ny,
                       int nz,
                       float xoffset,
                       float yoffset,
                       float zoffset,
                       float xscale,
                       float yscale,
                       float zscale,
                       float *p_cornerlines_v,
                       double *coordsv,
                       long ncoord)

{

    long ib, ic;
    int i, j;

    logger_info(LI, FI, FU, "Enter %s", FU);
    logger_info(LI, FI, FU, "Transforming grid ROFF coords -> XTG representation...");

    ib = 0;
    for (j = 0; j <= ny; j++) {
        for (i = 0; i <= nx; i++) {

            ic = 6 * (i * (ny + 1) + j);

            coordsv[ib++] = (p_cornerlines_v[ic + 3] + xoffset) * xscale;
            coordsv[ib++] = (p_cornerlines_v[ic + 4] + yoffset) * yscale;
            coordsv[ib++] = (p_cornerlines_v[ic + 5] + zoffset) * zscale;
            coordsv[ib++] = (p_cornerlines_v[ic + 0] + xoffset) * xscale;
            coordsv[ib++] = (p_cornerlines_v[ic + 1] + yoffset) * yscale;
            coordsv[ib++] = (p_cornerlines_v[ic + 2] + zoffset) * zscale;
        }
    }

    logger_info(LI, FI, FU, "Transforming grid ROFF coords -> XTG representation done");

    return EXIT_SUCCESS;
}
