/*
 ***************************************************************************************
 *
 * Convert from ROFF grid cornerpoint spec to XTGeo cornerpoint grid
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    name.c
 *
 * DESCRIPTION:
 *    Desciption, short or longer
 *
 * ARGUMENTS:
 *    points_v       i     a [9] matrix with X Y Z of 3 points
 *    nvector        o     a [4] vector with A B C D
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
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
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int grd3d_roff_to_xtgeo (
                         int nx,
                         int ny,
                         int nz,
                         float xoffset,
                         float yoffset,
                         float zoffset,
                         float xscale,
                         float yscale,
                         float zscale,
                         float *p_cornerlines_v,
                         char *p_splitenz_v,
                         float *p_zdata_v,
                         double *p_coord_v,
                         double *p_zcorn_v,
                         int *p_actnum_v,
                         int debug
                         )

{

    long ib, ic;
    int i, j;

    char sbn[24] = "grd3d_roff_to_xtgeo";
    xtgverbose(debug);

    xtg_speak(sbn, 2, "Transforming grid ROFF --> XTG representation ...");

    /*
     *----------------------------------------------------------------------------------
     * Pillars (COORD lines). ROFF order them from base to top in C order, while XTGeo
     * use the Eclipse style; top (X Y Z) to base (X Y Z) looping in F order.
     *----------------------------------------------------------------------------------
     */

    for (j = 1;j <= ny + 1; j++) {
        for (i = 1; i <= nx + 1; i++) {
            ib = x_ijk2ib(i, j, 1, nx + 1, ny + 1, 1, 0);  /* F order */
            ic = x_ijk2ic(i, j, 1, nx + 1, ny + 1, 1, 0);  /* C order */

            p_coord_v[ib + 0] = (p_cornerlines_v[ic + 3] + xoffset) * xscale;
            p_coord_v[ib + 1] = (p_cornerlines_v[ic + 4] + yoffset) * yscale;
            p_coord_v[ib + 2] = (p_cornerlines_v[ic + 5] + zoffset) * zscale;
            p_coord_v[ib + 3] = (p_cornerlines_v[ic + 0] + xoffset) * xscale;
            p_coord_v[ib + 4] = (p_cornerlines_v[ic + 1] + yoffset) * yscale;
            p_coord_v[ib + 5] = (p_cornerlines_v[ic + 2] + zoffset) * zscale;
        }
    }
}
