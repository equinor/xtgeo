/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_well_ijk.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Look along a well trajectory (X Y Z coords), and for each point find
 *    which I J K it has. Return as 3 IJK 1D arrays.
 *
 *    Note, the cell index is 1 based.
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions
 *    p_coord_v      i     Grid coordinate lines
 *    p_zcorn_v      i     Grid Z corners
 *    p_actnum_v     i     Grid ACTNUM parameter
 *    nval           i     Position of last point for well log
 *    p_utme_v       i     East coordinate vector for well log
 *    p_utmn_v       i     North coordinate vector for well log
 *    p_tvds_v       i     TVD (SS) coordinate vector for well log
 *    ivector        o     Returning I coordinates (UNDEF if not in grid)
 *    jvector        o     Returning J coordinates (UNDEF if not in grid)
 *    kvector        o     Returning K coordinates (UNDEF if not in grid)
 *    iflag          i     Options flag
 *    debug          i     Debug level
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
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_well_ijk(
                   int nx,
                   int ny,
                   int nz,
                   double *p_coord_v,
                   double *p_zcorn_v,
                   int *p_actnum_v,
                   double *p_zcorn_onelay_v,
                   int *p_actnum_onelay_v,
                   int nval,
                   double *p_utme_v,
                   double *p_utmn_v,
                   double *p_tvds_v,
                   int *ivector,
                   int *jvector,
                   int *kvector,
                   int iflag,
                   int debug
                   )

{

    char   sbn[24] = "grd3d_well_ijk";

    xtgverbose(debug);

    xtg_speak(sbn, 2, "Entering %s", sbn);
    xtg_speak(sbn, 3, "Using IFLAG: %d", iflag);
    xtg_speak(sbn, 3, "NX NY NZ: %d %d %d", nx, ny, nz);

    /*
     * Must be sure that grid is consistent in z, and also has
     * a small separation for each cell-layer, to avoid trouble with
     * zero cells
     */

    double zconst = 0.000001;
    grd3d_make_z_consistent(nx, ny, nz, p_zcorn_v, p_actnum_v, zconst, debug);

    /*
     * ========================================================================
     * Need to loop through each well point and sample zonelog from grid
     * ========================================================================
     */

    /* find a smart global startcell; middle of IJ and K=1 */
    long ibstart0 = x_ijk2ib(nx / 2, ny / 2, 1, nx, ny, nz, 0);
    long ibstart  = ibstart0;
    long ibstart2 = ibstart0;

    xtg_speak(sbn,2,"Working ...");

    int outside = -999;

    /* initial search options in grd3d_point_in_cell */
    int maxradsearch = 5;
    int nradsearch;
    int sflag = 1;  /* SFLAG=1 means that search shall take full grid as a last
		       attempt */

    int mnum;
    int icol = 0, jrow = 0, klay = 0, countpoints = 0;

    for (mnum = 0; mnum < nval; mnum++) {
	double xcor = p_utme_v[mnum];
	double ycor = p_utmn_v[mnum];
	double zcor = p_tvds_v[mnum];
        xtg_speak(sbn, 1, "Check point %lf   %lf   %lf", xcor, ycor, zcor);

	ivector[mnum] = 0;
	jvector[mnum] = 0;
	kvector[mnum] = 0;

        /* now check that the point is between top and base to avoid
           unneccasary searching and looping; hence use a one layer grid...
        */

        xtg_speak(sbn,2,"Check via grid envelope");

        /* loop cells in simplified (one layer) grid */
        long ib1 = grd3d_point_in_cell(ibstart2, 0, xcor, ycor, zcor,
                                       nx, ny, 1,
                                       p_coord_v,
                                       p_zcorn_onelay_v, p_actnum_onelay_v,
                                       maxradsearch,
                                       sflag, &nradsearch,
                                       0, debug);

        if (ib1 >= 0) {
            outside = 0;
            ibstart2 = ib1;
            xtg_speak(sbn, 1, "INSIDE GRID, nradsearch is %d", nradsearch);
        }
        else{
            outside = -777;
        }

        xtg_speak(sbn, 2, "Check via grid envelope DONE, outside status: %d",
                  outside);

        /* now go further if the point is inside the single layer grid */

        if (outside == 0) {


            /* loop cells in full grid */
            long ib2 = grd3d_point_in_cell(ibstart, 0, xcor, ycor, zcor,
                                           nx, ny, nz,
                                           p_coord_v,
                                           p_zcorn_v, p_actnum_v,
                                           maxradsearch,
                                           sflag, &nradsearch,
                                           0, debug);

            if (ib2 >= 0) {

                x_ib2ijk(ib2, &icol, &jrow, &klay, nx, ny, nz, 0);

                if (p_actnum_v[ib2] == 1) {

                    if (nradsearch > 3 && nradsearch <= 20) {
                        xtg_speak(sbn, 2, "Search radius is > 3: %d",
                                  nradsearch);
                    }
                    if (nradsearch>maxradsearch) {
                        xtg_speak(sbn, 1, "Search radius is large, %d",
                                  nradsearch);
                    }

                    if (mnum % 1000 == 0) {
                        xtg_speak(sbn, 2, "[%d]: Point %9.2f %9.2f %8.2f, the "
                                  "index is %d (%d %d %d). "
                                  "Search radius is %d",
                                  mnum, xcor, ycor, zcor, ib2, icol,
                                  jrow, klay, nradsearch);
                    }

                    ivector[mnum] = icol;
                    jvector[mnum] = jrow;
                    kvector[mnum] = klay;


                }
                else{
                    xtg_speak(sbn,2,"INACTIVE CELL "
                              "Point %9.2f %9.2f %8.2f, the cell index is "
                              "%d (%d %d %d) but inactive cell",
                              xcor, ycor, zcor, ib2, icol, jrow, klay);


                }

                ibstart  = ib2;
                countpoints++;

            }
            else{
                xtg_speak(sbn,2,"OUTSIDE Point %9.2f %9.2f %8.2f "
                          "is outside grid",
                          xcor, ycor, zcor);
                ibstart = ibstart0;
            }
        }

    }

    // xtg_speak(sbn,1,"Number of points inside: %d", countpoints);

    return EXIT_SUCCESS;

}
