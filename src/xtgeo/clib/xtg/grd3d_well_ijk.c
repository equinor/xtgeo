/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_well_ijk.c
 *
 * DESCRIPTION:
 *    Look along a well trajectory (X Y Z coords), and for each point find
 *    which I J K it has. Return as 3 IJK 1D arrays.
 *
 *    Note, the cell index is 1 based.
 *
 * ARGUMENTS:
 *    nx,ny,nz           i     Grid dimensions ncol, nrow, nlay
 *    coordsv            i     Grid coordinate lines
 *    zcornsv            i     Grid Z corners
 *    actnumsv           i     Grid ACTNUM parameter
 *    p_zcorn_onelay_v   i     Grid Z corners, top bot only
 *    p_actnum_onelay_v  i     Grid ACTNUM parameter top bot only
 *    nval               i     Position of last point for well log
 *    p_utme_v           i     East coordinate vector for well log
 *    p_utmn_v           i     North coordinate vector for well log
 *    p_tvds_v           i     TVD (SS) coordinate vector for well log
 *    ivector            o     Returning I coordinates (UNDEF if not in grid)
 *    jvector            o     Returning J coordinates (UNDEF if not in grid)
 *    kvector            o     Returning K coordinates (UNDEF if not in grid)
 *    iflag              i     Options flag
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems
 *    Updated *vector variables
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

#define DEBUG 0

int grd3d_well_ijk(
    int nx,
    int ny,
    int nz,

    double *coordsv,
    long ncoordin,
    double *zcornsv,
    long nzcornin,
    int *actnumsv,
    long nactin,

    double *p_zcorn_onelay_v,
    long nzcornonein,
    int *p_actnum_onelay_v,
    long nactonein,

    int nval,
    double *p_utme_v,
    double *p_utmn_v,
    double *p_tvds_v,
    int *ivector,
    int *jvector,
    int *kvector,
    int iflag
    )

{

    logger_info(LI, FI, FU, "Entering %s", FU);

    /*
     * Must be sure that grid is consistent in z, and also has
     * a small separation for each cell-layer, to avoid trouble with
     * zero cells
     */

    double zconst = 0.000001;
    grd3d_make_z_consistent(nx, ny, nz, zcornsv, 0, zconst);

    /*
     * ========================================================================
     * Need to loop through each well point and sample zonelog from grid
     * ========================================================================
     */

    /* find a smart global startcell; middle of IJ and K=1 */
    long ibstart0 = x_ijk2ib(nx / 2, ny / 2, 1, nx, ny, nz, 0);
    long ibstart  = ibstart0;
    long ibstart2 = ibstart0;

    int outside = -999;

    /* initial search options in grd3d_point_in_cell */
    int maxradsearch = 5;
    int nradsearch;
    int sflag = 1;  /* SFLAG=1 means that search shall take full grid as a last
		       attempt */

    int mnum;
    int icol = 0, jrow = 0, klay = 0;

    for (mnum = 0; mnum < nval; mnum++) {
	double xcor = p_utme_v[mnum];
	double ycor = p_utmn_v[mnum];
	double zcor = p_tvds_v[mnum];
        logger_debug(LI, FI, FU, "Check point %lf   %lf   %lf", xcor, ycor, zcor);

	ivector[mnum] = 0;
	jvector[mnum] = 0;
	kvector[mnum] = 0;

        /* now check that the point is between top and base to avoid
           unneccasary searching and looping; hence use a one layer grid...
        */

        logger_debug(LI, FI, FU,"Check via grid envelope");

        /* loop cells in simplified (one layer) grid */
        long ib1 = grd3d_point_in_cell(ibstart2, 0, xcor, ycor, zcor,
                                       nx, ny, 1,
                                       coordsv,
                                       p_zcorn_onelay_v, p_actnum_onelay_v,
                                       maxradsearch,
                                       sflag, &nradsearch,
                                       0, DEBUG);

        if (ib1 >= 0) {
            outside = 0;
            ibstart2 = ib1;
        }
        else{
            outside = -777;
        }

        logger_info(LI, FI, FU, "Check grid envelope DONE, outside status: %d", outside);

        /* now go further if the point is inside the single layer grid */

        if (outside == 0) {


            /* loop cells in full grid */
            long ib2 = grd3d_point_in_cell(ibstart, 0, xcor, ycor, zcor,
                                           nx, ny, nz,
                                           coordsv,
                                           zcornsv, actnumsv,
                                           maxradsearch,
                                           sflag, &nradsearch,
                                           0, DEBUG);

            if (ib2 >= 0) {

                x_ib2ijk(ib2, &icol, &jrow, &klay, nx, ny, nz, 0);

                if (actnumsv[ib2] == 1) {

                    ivector[mnum] = icol;
                    jvector[mnum] = jrow;
                    kvector[mnum] = klay;
                }

                ibstart  = ib2;

            }
            else{
                /* outside grid */
                ibstart = ibstart0;
            }
        }

    }

    logger_info(LI, FI, FU, "Exit from %s", FU);
    return EXIT_SUCCESS;

}
