/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_rpt_zlog_vs_zon.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Reports zone log vs zone in grid mismatch.
 *    Compares a zone log for a well with the zonation in the grid. Returns
 *    a percent of match, and updates some 3D parameters that will later be
 *    used for grid adjustment
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions
 *    p_coord_v      i     Grid coordinate lines
 *    p_zcorn_v      i     Grid Z corners
 *    p_actnum_v     i     Grid ACTNUM parameter
 *    p_zon_v        i     Grid zone parameter
 *    nval           i     Position of last point for well log
 *    p_utme_v       i     East coordinate vector for well log
 *    p_utmn_v       i     North coordinate vector for well log
 *    p_tvds_v       i     TVD (SS) coordinate vector for well log
 *    p_zlog_v       i     Zone log vector
 *    zlmin,zlmax    i     min max ZLOG val to look inside for well log
 *    p_adjz_v       io    Adjust parameter vertically (neg. for --> shallower)
 *    p_confl_v      io    Parameter with flag to adress conflicts
 *    results        o     Vector with some descriptions:
 *                         results[0] = percent match of zonelog
 *                         results[1] = total N of points evaluated
 *                         results[2] = matching N of points evaluated
 *                         (if even sampling, the points ratio will eq
 *                         length ratio)
 *    iflag          i     Options flag
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems
 *
 * TODO/ISSUES/BUGS:
 *    Code is not finished
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_rpt_zlog_vs_zon(
			  int    nx,
			  int    ny,
			  int    nz,
			  double *p_coord_v,
			  double *p_zcorn_v,
			  int    *p_actnum_v,
			  int    *p_zon_v,
			  int    nval,
			  double *p_utme_v,
			  double *p_utmn_v,
			  double *p_tvds_v,
			  int    *p_zlog_v,
			  int    zlmin,
			  int    zlmax,
			  double *p_zcorn_onelay_v,
			  int    *p_actnum_onelay_v,
			  double *results,
			  int    iflag,
			  int    debug
			  )

{
    /* locals */
    double  zconst, x, y, z;
    int    m, i, j, k, ib, ibstart, ibstart0, ibstart2, lzone, nzone;
    char   s[24]="grd3d_rpt_zlog_vs_zon";
    int    *p_zsample_v, *p_icell_v, *p_jcell_v, *p_kcell_v;
    int    mtopmark, mbotmark, matchcount, totalcount, outside, nradsearch;
    int    mlimit=100;
    int    countpoints=0;
    int    maxradsearch, sflag;

    p_zsample_v = calloc(nval+1,sizeof(int));
    p_icell_v   = calloc(nval+1,sizeof(int));
    p_jcell_v   = calloc(nval+1,sizeof(int));
    p_kcell_v   = calloc(nval+1,sizeof(int));

    xtgverbose(debug);

    xtg_speak(s, 2, "Entering <grd3d_adj_z_from_zlog>");
    xtg_speak(s, 3, "Using IFLAG: %d", iflag);

    xtg_speak(s, 3, "NX NY NZ: %d %d %d", nx, ny, nz);


    /*
     * Must be sure that grid is consistent in z, and also has
     * a small separation for each cell-layer, to avoid troble with
     * zero cells
     */

    zconst=0.01;
    grd3d_make_z_consistent(
			    nx,
			    ny,
			    nz,
			    p_zcorn_v,
			    p_actnum_v,
			    zconst,
			    debug
			    );




    /*
     * =========================================================================
     * Need to loop through each well point and sample zonelog from grid
     * =========================================================================
     */

    /* find a smart global startcell; middle of IJ and K=1 */
    ibstart0=x_ijk2ib(nx/2,ny/2,1,nx,ny,nz,0);

    ibstart  = ibstart0;
    ibstart2 = ibstart0;


    xtg_speak(s,2,"Working ...");

    /*
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * need to find the first and last point of the zone log interval, and
     * mark those points.
     */

    mtopmark = -1;
    mbotmark = -1;

    for (m=0; m<=nval; m++) {
	lzone = p_zlog_v[m];

	if (lzone >= zlmin && lzone <= zlmax && mtopmark<0) {
	    mtopmark = m;
	    xtg_speak(s,2,"First valid zonelog value at position %d", mtopmark);
	}
	if (lzone >= zlmin && lzone <= zlmax) {
	    mbotmark = m;
	    xtg_speak(s,2,"Last valid zonelog value at position %d", mbotmark);
	}
    }

    if (mtopmark>mbotmark) {
	xtg_error(s,"Something is wrong with MTOPMARK and MBOTMARK. Call JRIV");
    }

    if (mtopmark==-1 || mbotmark==-1) {
	xtg_warn(s,1,"No zonation for well?");
	return(2);
    }

    /*
     * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
     * loop from mtopmark and go below
     */

    xtg_speak(s,2,"MTOPMARK is %d and MBOTMARK is %d (n=%d)",
	      mtopmark, mbotmark, mbotmark - mtopmark +1);


    outside=-999;

    /* initial search options in grd3d_point_in_cell */
    maxradsearch=5;
    sflag=1; /* SFLAG=1 means that search shall take full grid as a last
		attempt */


    for (m=mtopmark; m<=mbotmark; m++) {
	x     = p_utme_v[m];
	y     = p_utmn_v[m];
	z     = p_tvds_v[m];
	lzone = p_zlog_v[m];

	p_icell_v[m] = 0;
	p_jcell_v[m] = 0;
	p_kcell_v[m] = 0;

	if (lzone >= zlmin && lzone <= zlmax) {


	    /* now check that the point is between top and base to avoid
	       unneccasary searching and looping; hence use a one layer grid...
	    */


	    xtg_speak(s,2,"Check via grid envelope");


	    /* loop cells in simplified (one layer) grid */
	    ib=grd3d_point_in_cell(ibstart2, 0, x, y, z, nx, ny, 1,
				   p_coord_v,
				   p_zcorn_onelay_v, p_actnum_onelay_v,
				   maxradsearch,
				   sflag, &nradsearch,
				   0, debug);

	    if (ib>=0) {
		outside=0;
		ibstart2=ib;
	    }
	    else{
		outside=-777;
	    }

	    xtg_speak(s,2,"Check via grid envelope DONE, outside status: %d",
		      outside);


	    /* now go further if the point is inside the single layer grid
	     */

	    if (outside==0) {

		xtg_speak(s,2,"Well point zone (%d) is:  %9.2f   %9.2f   %8.2f"
			  "(ZLMIN ZLMAX is %d %d)",
			  lzone, x, y, z, zlmin, zlmax);



		/* loop cells in simplified (one layer) grid */
		ib=grd3d_point_in_cell(ibstart, 0, x, y, z, nx, ny, nz,
				       p_coord_v,
				       p_zcorn_v, p_actnum_v,
				       maxradsearch,
				       sflag, &nradsearch,
				       0, debug);



		if (ib>=0) {
		    x_ib2ijk(ib,&i,&j,&k,nx,ny,nz,0);
		    if (p_actnum_v[ib]==1) {
			nzone=p_zon_v[ib];

			if (nradsearch>3 && nradsearch<=20) {
			    xtg_speak(s,2,"Search radius is >3: %d",nradsearch);
			}
			if (nradsearch>maxradsearch) {
			    xtg_speak(s,1,"Search radius is large, %d",
				      nradsearch);
			}

			if (m%1000==0) {
			    xtg_speak(s,2,"[%d]: Point %9.2f %9.2f %8.2f, the "
				      "index is %d (%d %d %d) "
				      "and zone is %d [wzone = %d]."
				      " Search radius is %d",
				      m, x,y,z,ib,i,j,k,nzone,lzone,nradsearch);
			}
			p_zsample_v[m]=nzone;

			p_icell_v[m]=i;
			p_jcell_v[m]=j;
			p_kcell_v[m]=k;


		    }
		    else{
			xtg_speak(s,2,"INACTIVE CELL "
				  "Point %9.2f %9.2f %8.2f, the cell index is "
				  "%d (%d %d %d) but inactive cell",
				  x,y,z,ib,i,j,k);

			p_zsample_v[m]=-777;

		    }

		    ibstart  = ib;
		    countpoints++;

		}
		else{
		    xtg_speak(s,2,"OUTSIDE Point %9.2f %9.2f %8.2f "
			      "is outside grid",
			      x,y,z);
		    p_zsample_v[m]=-999;
		    ibstart=ibstart0;
		}
	    }
	}

    }

    xtg_speak(s,1,"Number of points inside: %d", countpoints);


    /*
     * ========================================================================
     * Compare zonations actual well vs well sampled in 3D grid and report
     * ========================================================================
     */

    xtg_speak(s,2,"---------------------------------------------------------");
    xtg_speak(s,2,"Well zonation vs grid zonation:");
    xtg_speak(s,2," >>    %4s %4s  (%9s %9s %8s) [cell %4s %4s %4s]",
	      "WELL", "ZONE", "x", "y", "z", "I", "J", "K");

    /* count match to compute match % */
    matchcount=0;
    totalcount=0;

    if (debug>1) mlimit=99999;

    for (m=0;m<=nval;m++) {

	x     = p_utme_v[m];
	y     = p_utmn_v[m];
	z     = p_tvds_v[m];
	lzone = p_zlog_v[m];
	nzone = p_zsample_v[m];

	i = p_icell_v[m];
	j = p_jcell_v[m];
	k = p_kcell_v[m];

	if (lzone>=zlmin && lzone<=zlmax) {

	    /* report (restrict to found zonation for wells and/or grid) */

	    if (p_zlog_v[m]>-9 || p_zsample_v[m]>-9) {

		totalcount++;
		if (p_zlog_v[m] == p_zsample_v[m]) {
		    matchcount++;
		}

		if (totalcount < mlimit) {
		    xtg_speak(s,2," >>   %4d %4d  (%9.2f %9.2f %8.2f) "
			      "[cell %4d %4d %4d]",
			      lzone, nzone, x, y, z, i, j, k);
		}
		else if (totalcount == mlimit) {
		    xtg_speak(s,2,"Etc... (The rest is not displayed)");
		}

	    }
	}
    }

    results[0]=100*(double)matchcount/(double)totalcount;
    results[1]=(double)totalcount;
    results[2]=(double)matchcount;

    xtg_speak(s,2,"Match count is %7.2f percent",results[0]);


    xtg_speak(s,2,"Adjusting grid to zlog ... DONE!");
    xtg_speak(s,2,"Exiting <grd3d_adj_z_from_zlog>");


    return EXIT_SUCCESS;

}
