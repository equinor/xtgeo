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
                         int *p_splitenz_v,
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

    xtg_speak(sbn, 2, "Transforming grid ROFF coords --> XTG representation ...");

    /*
     * ---------------------------------------------------------------------------------
     * Pillars (COORD lines). ROFF order them from base to top in C order, while XTGeo
     * use the Eclipse style; top (X Y Z) to base (X Y Z) looping in F order.
     * ---------------------------------------------------------------------------------
    */
    ib = 0;
    for (j = 0;j <= ny; j++) {
        for (i = 0; i <= nx; i++) {

            ic = 6 * (i * (ny + 1) + j);

            p_coord_v[ib++] = (p_cornerlines_v[ic + 3] + xoffset) * xscale;
            p_coord_v[ib++] = (p_cornerlines_v[ic + 4] + yoffset) * yscale;
            p_coord_v[ib++] = (p_cornerlines_v[ic + 5] + zoffset) * zscale;
            p_coord_v[ib++] = (p_cornerlines_v[ic + 0] + xoffset) * xscale;
            p_coord_v[ib++] = (p_cornerlines_v[ic + 1] + yoffset) * yscale;
            p_coord_v[ib++] = (p_cornerlines_v[ic + 2] + zoffset) * zscale;
        }
    }

    for (i = 0; i < 12; i++) {
        printf("NEW %f\n", p_coord_v[i]);
    }
    return EXIT_SUCCESS;
}
