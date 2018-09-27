/*
 * ############################################################################
 * grd3d_pol_ftrace.c
 * Make a polygon based on a fault trace property grid
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                          GRD3D_POL_FTRACE
 * ****************************************************************************
 * Look for fault traces...
 * ----------------------------------------------------------------------------
 *
 */
int grd3d_pol_ftrace(
		     int    nx,
		     int    ny,
		     int    nz,
		     double  *p_coord_v,
		     double  *p_zcorn_v,
		     int    *p_actnum_v,
		     double  *p_fprop_v,
		     double  *p_fprop2_v,
		     int    klayer,
		     int    nelem,
		     int    *fid,
		     double  *sfr,
		     double  *dir,
		     int    *ext,
		     int    *pri,
		     double  *xw1,
		     double  *xw2,
		     double  *yw1,
		     double  *yw2,
		     int    *np,
		     double  *p_xp_v,
		     double  *p_yp_v,
		     int    maxradius,
		     int    ntracemax,
		     int    option,
		     int    debug
		     )
{
    int      i, j, k, ib, il, il_first, istart, incr, nfirst, iflag=0, iprev;
    double    firstsfr, x;
    int      *imark_v, npcounter, ic, jc, kc, *nfprop_v, ntrace, *nftype_v;
    double    *rank_v, maxrank;
    double   xg, yg, zg, vlen, azi_rad, azi_deg, xg0, yg0, avgdir;
    double   currentdir, nextdir, currentx1, currentx2, currenty1, currenty2;
    double   nextx1, nextx2, nexty1, nexty2;
    double   angdiff;
    int      currentfid, nextfid, nextpri, nradius, currentext;
    int      istatus, ib_use=0, il_use=0, ncurrentfault;
    int      nlastfault, ncount4eval=0;
    int      ierr;

    char     s[24]="grd3d_pol_ftrace";

    xtgverbose(debug);

    xtg_speak(s,1,"Compute polygons from 3D grid (layer) fault traces ...");
    xtg_speak(s,2,"NX NY NZ is %d %d %d", nx, ny, nz);


    /*
     *-------------------------------------------------------------------------
     * Some checks and correction of inputs (more tests probably needed)
     *-------------------------------------------------------------------------
     */
    for (i=0;i<nelem;i++) {
	if (dir[i]==0) { /* had some issues with 0 angle */
	    dir[i]=1;
	}
	if (dir[i]==360) {
	    dir[i]=359;
	}
	if (dir[i]<0 || dir[i]>360) {
	    xtg_error(s,"Direction out of limits (0..360): %5.2f",dir[i]);
	}
    }


    /*
     *-------------------------------------------------------------------------
     * Work in a given K layer
     * Identify first cell (starting point). Need to set up some help vectors
     * to identify which cells that has been traced (imark_v). Also need to
     * store the first coordinate (for closing the polygon)
     *-------------------------------------------------------------------------
     */

    imark_v   = calloc(nx*ny, sizeof(int));
    rank_v    = calloc(nx*ny, sizeof(double));
    nfprop_v  = calloc(nx*ny*nz, sizeof(int));
    nftype_v  = calloc(nx*ny*nz, sizeof(int)); /* fault type extracted cf _import_grd3d_grdecl_prop in Zone/_Etc*/

    /* status */
    ierr=1;


    /* initialize: */
    xtg_speak(s,2,"Inialize ... for KLAYER %d",klayer);
    for (i=0;i<nx*ny;i++) {
	imark_v[i] = 0;
	rank_v[i]  = 0.0;
    }

    /* "round" the properties from 3002 etc to 3000 e.g; only needed for current klayer */

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {

	    ib=x_ijk2ib(i,j,klayer,nx,ny,nz,0);

	    /* make actnum neutral ... */
	    if (p_actnum_v[ib]<=1) {
		if (p_fprop_v[ib]>999) {
		    x=p_fprop_v[ib]/1000.0;
		    nfprop_v[ib]=(int)x*1000;
		    nftype_v[ib]=(int)p_fprop_v[ib] - nfprop_v[ib];
		}
		/* polygon: */
		else if (p_fprop_v[ib]>899 && p_fprop_v[ib]<1000) {
		    x=p_fprop_v[ib];
		    nfprop_v[ib]=(int)x;
		    nftype_v[ib]=0;
		}
		else{
		    nfprop_v[ib]=0;
		    nftype_v[ib]=0;
		}

		/* detect edges 1 (i=1) 2 (j=1), 3 (i=nx) 4 (j=ny) and mark them*/
		if (nfprop_v[ib]==0) {
		    if (i==1)  nfprop_v[ib]=1;
		    if (i==nx) nfprop_v[ib]=2;
		    if (j==1)  nfprop_v[ib]=3;
		    if (j==ny) nfprop_v[ib]=4;

		    if (i==1)  nftype_v[ib]=1;
		    if (i==nx) nftype_v[ib]=2;
		    if (j==1)  nftype_v[ib]=3;
		    if (j==ny) nftype_v[ib]=4;
		}

	    }
	}
    }

    xtg_speak(s,2,"Inialize ... DONE");


    /* establish the way to search for first coordinate */
    ncurrentfault=0;
    nlastfault=nelem-1; /* last fault when counting from 0 */
    il_first=0;

    nfirst=fid[ncurrentfault]; /* first fault id to look for */
    xtg_speak(s,2,"Code for nfirst is %10d", nfirst);

    if (nfirst > 1000) {
        xtg_warn(s,0,"Code for nfirst is %10, expected 1000 ...", nfirst);
    }

    /*
     *-------------------------------------------------------------------------
     * Find the first point in the polygon (I search), first fault
     *-------------------------------------------------------------------------
     */

    /* window */
    currentx1  = xw1[0];
    currentx2  = xw2[0];
    currenty1  = yw1[0];
    currenty2  = yw2[0];


    /* decode the sfr ... */
    firstsfr=sfr[0];
    if (firstsfr < 2 && firstsfr >-2) {

	xtg_speak(s,2,"Code for firstsfr is %10.6f", firstsfr);
	/* then loop from I */
	x = fabs(firstsfr);
	x = x - 1;
	x = x * 100000 + 0.001;
	istart=(int)x;
	incr=1;
	if (firstsfr<0) {
	    incr=incr*-1;
	}
	xtg_speak(s,1,"Istart is %d with incr %d (x is %10.6f)", istart,incr,x);

	iflag=-2;

	/* for I direction, we need to work spin J fastest (inner) */
	for (i=istart;i<=nx;i+=incr) {
            xtg_speak(s,2,"Searchfrom (Idir): I index is %d ...", i);
	    /* in case of revert search, must stop at cell i = 1 */
	    if (i<1) {
		xtg_warn(s,1, "<searchfrom> Range of i index out of bound "
                         "(i=%d). Wrong specification or missing?", i);
                iflag=-1;
	    }
	    for (j=1;j<=ny;j++) {
		ib=x_ijk2ib(i,j,klayer,nx,ny,nz,0);
		il=x_ijk2ib(i,j,1,nx,ny,1,0);
		xtg_speak(s,3,"Fault mark for I J (%d %d) is %d", i,j, (int)p_fprop_v[ib]);

		if (p_actnum_v[ib]<=1 && nfprop_v[ib]==nfirst && iflag==-2) {

		    /* get correct point of this cell */
		    grd3d_cellpoint(i,j,klayer,nx,ny,nz,nftype_v[ib],p_coord_v,
				   p_zcorn_v,&xg,&yg,&zg,debug);

		    xtg_speak(s,2,"Check window: %f %f %f %f (<< %f %f >>)",
			      currentx1, currentx2, currenty1, currenty2, xg, yg);

		    if (xg>=currentx1 && xg<=currentx2 && yg>=currenty1 && yg<=currenty2) {

			xtg_speak(s,1,"First active cell (nfirst) is %d : %d %d", nfirst,i,j);
			imark_v[il]=nfirst;
			iflag=ib;
			il_first=il;

			/* set first point */
			npcounter=0;
			p_xp_v[npcounter]=xg;
			p_yp_v[npcounter]=yg;
			xtg_speak(s,1,"First coordinate: %12.1f %12.1f", xg, yg);

			p_fprop2_v[ib]=p_fprop_v[ib];


			if (iflag>=-1) break;
		    }
		}
		if (iflag>=-1) break;
	    }
	    if (iflag>=-1) break;
	}
    }

    /*
     *-------------------------------------------------------------------------
     * Find the first point in the polygon (J search)
     *-------------------------------------------------------------------------
     */
    /* decode the sfr ... */
    firstsfr=sfr[0];
    if (firstsfr > 2 || firstsfr < -2) {

	xtg_speak(s,2,"Code for firstsfr is %10.6f", firstsfr);
	/* then loop from J */
	x = fabs(firstsfr);
	x = x - 2;
	x = x * 100000 + 0.001;
	istart = (int)x;
	incr = 1;
	if (firstsfr<0) {
	    incr=incr * -1;
	}
	xtg_speak(s,1,"Jstart is %d with incr %d (x is %10.6f)", istart,incr,x);

	iflag=-2;

	/* for J direction, we need to work spin I fastest (inner) */
	for (j=istart;j<=ny;j+=incr) {
            xtg_speak(s,2,"HEI Searchfrom (Jdir): J index is %d ...", j);
	    /* in case of revert search, must stop at cell j = 1 */
	    if (j < 1) {
		xtg_warn(s, 1, "<searchfrom> Range of j index out of bound "
                         "(j=%d). Wrong specification or missing? Or fault "
                         "not found? Anyway... try more...", j);
                iflag = -1;
	    }
	    for (i=1;i<=nx;i++) {
		ib=x_ijk2ib(i,j,klayer,nx,ny,nz,0);
		il=x_ijk2ib(i,j,1,nx,ny,1,0);
                if (p_fprop_v[ib] > 0) {
                    xtg_speak(s,2,"Fault mark for I J K (%d %d %d) is %d (%d)",
                              i, j, klayer, (int)p_fprop_v[ib], nfprop_v[ib]);
                    xtg_speak(s,2,"ACTNUM, nfirst and iflag: %d %d %d",
                              p_actnum_v[ib], nfirst, iflag);
                }
		if (p_actnum_v[ib]<=1 && nfprop_v[ib]==nfirst && iflag==-2) {

                    xtg_speak(s,2,"Cell found! at %d %d %d", i, j, klayer);

		    /* get correct point of this cell */
		    grd3d_cellpoint(i,j,klayer,nx,ny,nz,nftype_v[ib],p_coord_v,
				   p_zcorn_v,&xg,&yg,&zg,debug);


		    xtg_speak(s,2,"Check window: %f %f %f %f (<< %f %f >>)",
			      currentx1, currentx2, currenty1, currenty2, xg, yg);

		    if (xg>=currentx1 && xg<=currentx2 && yg>=currenty1 &&
                        yg<=currenty2) {

			xtg_speak(s,1,"First active cell (nfirst) is %d : "
                                  "%d %d", nfirst,i,j);
			imark_v[il]=nfirst;
			iflag=ib;
			il_first=il;

			/* set first point */
			npcounter=0;
			p_xp_v[npcounter]=xg;
			p_yp_v[npcounter]=yg;
			xtg_speak(s,1,"First coordinate: %12.1f %12.1f", xg, yg);

			p_fprop2_v[ib]=p_fprop_v[ib];

			if (iflag >= -1) break;
		    }
		}
		if (iflag >= -1) break;
	    }
	    if (iflag >= -1) break;
	}
    }


    if (iflag<0) {
	xtg_warn(s, 2, "Warning; did not find first cell of first fault");
	xtg_warn(s, 2, "No success for this layer...");
        ierr = 0;
        goto finalise;
    }


    /*
     *-------------------------------------------------------------------------
     * Find the next points in the polygon...
     *-------------------------------------------------------------------------
     */

    /* search for next iflag is the the previous cell; need to search around this*/

    iprev=iflag;
    currentfid = fid[ncurrentfault];
    nextfid    = fid[ncurrentfault+1];

    /* azimuth */
    currentdir = dir[0];
    nextdir = dir[1];

    /* priority */
    nextpri = pri[1];

    /* extend flag (1 means extand before 2 means extend after, 3 means both) */
    currentext = ext[0];

    /* window */
    currentx1  = xw1[0];
    currentx2  = xw2[0];
    currenty1  = yw1[0];
    currenty2  = yw2[0];

    nextx1  = xw1[1];
    nextx2  = xw2[1];
    nexty1  = yw1[1];
    nexty2  = yw2[1];


    /* remember the coords */
    xg0 = p_xp_v[npcounter];
    yg0 = p_yp_v[npcounter];


    /* ntrace is the number of points in the polygon; this is a max number */
    for (ntrace=0; ntrace<ntracemax; ntrace++) {
	x_ib2ijk(iprev,&ic,&jc,&kc,nx,ny,nz,0);

	/* the nradius is the search area around a current cell; it gets bigger and bigger */
	for (i=0;i<nx*ny;i++) {
	    rank_v[i]  = 0.0;
	}


	for (nradius=1;nradius<=maxradius;nradius++) {

	    xtg_speak(s,2,"===== LAYER %5d ===============================================",klayer);
	    xtg_speak(s,2,"Current cell <%d,%d,%d>: Search radius is %d", ic, jc, klayer, nradius);
	    xtg_speak(s,2,"=================================================================");

	    /* number of potential cells within the give nradius needs reset*/
	    ncount4eval=0;

	    for (j=jc-nradius;j<=jc+nradius;j++) {

		for (i=ic-nradius;i<=ic+nradius;i++) {

		    if (j>=1 && j<=ny && i>=1 && i<=nx && ! (i==ic && j==jc)) {

			ib=x_ijk2ib(i,j,klayer,nx,ny,nz,0);
			il=x_ijk2ib(i,j,1,nx,ny,1,0);

			xtg_speak(s,3,"Neighbour cell with this fault is %d %d (IB %d, IL %d)",
				  i, j, ib, il);


			/* Need to find if the cell within the radius is within the approx direction
			 * of the current fault. Searching in wrong directions should be given
			 * zero weight
			 */

			/* find point of that cell... */
			grd3d_cellpoint(i,j,klayer,nx,ny,nz,nftype_v[ib],p_coord_v,
				   p_zcorn_v,&xg,&yg,&zg,debug);

			/* find the lenght and angle of the vector */
			istatus=x_vector_info2(xg0, xg, yg0, yg, &vlen, &azi_rad, &azi_deg, 0, debug);


			/* the first test is to check if the cell is a candidate, and it a candidate
			 * if: - active (or not)
			 *     - nfprop_v[ib] is > 0 (a fault trace)
			 *     - it is not already used (skip if imark_v[il]>0)
			 * this will return an array of possible cells; next, we need to rank them:
			 *     - within range of azimuth
                         *     - is the same fault
			 *     - prioritize cell in direction instead of corners
                         *     - is a next fault in the list (and also rank by azimuth, which may
			 *     - be a turning point (bend) between previous fault and the next)
			 */

			if (p_actnum_v[ib]<=1 && nfprop_v[ib]>0 && (nfprop_v[ib]==currentfid || nfprop_v[ib]==nextfid)
			    && imark_v[il]==0) {


			    /* a possible candidate is found; now need to give it a rank */
			    grd3d_cellpoint(i,j,klayer,nx,ny,nz,nftype_v[ib],p_coord_v,
					   p_zcorn_v,&xg,&yg,&zg,debug);

			    /* compute length and azimuth */

			    istatus=x_vector_info2(xg0, xg, yg0, yg, &vlen, &azi_rad, &azi_deg, 0, debug);
			    xtg_speak(s,2,"Testing coordinate... %10.1f %10.1f", xg, yg);


			    /* rank should be a number that shall be largest for closest cells
			     * and for cells with preferable azimuth
			     */

			    xtg_speak(s,2,"SOURCE cell <%d,%d> .... Possible CELL <%d,%d>",ic,jc, i,j);
			    xtg_speak(s,2,"Possible CELL <%d,%d> has ACTNUM %d, NFPROP %d and IMARK %d",
				      i,j,p_actnum_v[ib],nfprop_v[ib],imark_v[il]);

			    rank_v[il]=1;

			    xtg_speak(s,2,"Initial rank is %5.2f",rank_v[il]);


			    if (nfprop_v[ib]==currentfid) {
				/*
				 * Overview of ranks:
				 * Cell within searchdir 44 deg           = 1+1+1+1 = 4
				 * ...and i og j aligned                     +1     = 5
				 * ...and closest cell                       +5     = 10
				 *
				 */


				xtg_speak(s,2,"CURRENTDIR %5.2f   AZI_DEG %5.2f",currentdir,azi_deg);

				angdiff=x_diff_angle(currentdir,azi_deg,1,debug);
				xtg_speak(s,2,"ANGLEDIFF <%d,%d,%d> is %5.2f",i,j,klayer,angdiff);

				/* increase the rank according to aperture */
				if (fabs(angdiff)<111)   rank_v[il] +=1 ;
				if (fabs(angdiff)<89)    rank_v[il] +=1 ;
				if (fabs(angdiff)<44)    rank_v[il] +=1 ;

				if (fabs(angdiff)>=111)  rank_v[il] =0 ;


				/* also, rank is added if the cell is neighbour in i */
				if (rank_v[il]>0 && i==ic)  {
				    rank_v[il] +=1 ;
				    /* and in particular if closest neighbour */
				    if (abs(jc-j)==1) {
					rank_v[il] += 5;
					xtg_speak(s,2,"Same I, J-diff=1, for <%d,%d,%d>",i,j,klayer);
				    }
				}
				/* ...or, rank is added if the cell is neighbour in j */
				if (rank_v[il]>0 && j==jc)  {
				    rank_v[il] +=1 ;
				    /* and in particular if closest neighbour */
				    if (abs(ic-i)==1) {
					rank_v[il] += 5;
					xtg_speak(s,2,"Same J, I-diff=1, for <%d,%d,%d>",i,j,klayer);
				    }

				}


				/* if the point is outside the requested UTM window, it gets rank 0 */
				if (xg<currentx1 || xg>currentx2 || yg<currenty1 || yg>currenty2) {
				    rank_v[il]=0;
				    xtg_speak(s,2,"X: %10.1f << %10.1f >> %10.1f    Y: %10.1f << %10.1f >> %10.1f  ",
					      currentx1,xg,currentx2, currenty1,yg,currenty2);
				    xtg_speak(s,2,"RANK (3b) %5.2f",rank_v[il]);
				}

				xtg_speak(s,2,"RANK for cell (THIS FAULT ID): %5.2f",rank_v[il]);

			    }
			    /*
			     *-------------------------------------------------
			     * Next fault:
			     *-------------------------------------------------
			     */

			     /*
			     * Overview of ranks:
			     * Cell within searchdir (which is next search angle) 44 deg           = 1+1+1+1 = 4
			     * nextpri=force=medium:                                                  +5     = 9
			     * or high...
			     */

			    else if (nfprop_v[ib]==nextfid) {
				/* avgdir=0.5*(currentdir+nextdir);  THIS GOES WRONG IN SPECIAL CASES */

				avgdir=nextdir;
				rank_v[il]=1;

				angdiff=x_diff_angle(avgdir,azi_deg,1,debug);

				/* increase the rank according to aperture */
				if (fabs(angdiff)<111)   rank_v[il] +=1 ;
				if (fabs(angdiff)<89)    rank_v[il] +=1 ;
				if (fabs(angdiff)<44)    rank_v[il] +=1 ;


				/* give always fault priority to next fault (which is first, no 0) if this
				 *  is last fault; the loop is closed
				 */


				if (ncurrentfault==nlastfault) {
				    nextpri=1;
				}

				/* give extra pri of priority is 'force'  (or last) */
				if (nextpri>=1) {
				    xtg_speak(s,2,"Nextpri is active");
				    if (rank_v[il]>2) {
					rank_v[il]+=5;
				    }
				    else{
					rank_v[il]+=1;
				    }
				}

				if (nextpri==2) {
				    xtg_speak(s,2,"Nextpri is active");
				    if (rank_v[il]>2) {
					rank_v[il]+=2;
				    }
				    else{
					rank_v[il]+=1;
				    }
				}


				if (nextpri==1) {
				    /* also, rank is added if the cell is neighbour in i */
				    if (rank_v[il]>0 && i==ic)  {
					rank_v[il] +=1 ;
					/* and in particular if closest neighbour */
					if (abs(jc-j)==1) {
					    rank_v[il] += 5;
					    xtg_speak(s,2,"Same I, J-diff=1, for <%d,%d,%d>",i,j,klayer);
					}
				    }
				    /* ...or, rank is added if the cell is neighbour in j */
				    if (rank_v[il]>0 && j==jc)  {
					rank_v[il] +=1 ;
					/* and in particular if closest neighbour */
					if (abs(ic-i)==1) {
					    rank_v[il] += 5;
					    xtg_speak(s,2,"Same J, I-diff=1, for <%d,%d,%d>",i,j,klayer);
					}

				    }
				}


				/* if the point is outside the requested UTM window, it get rank 0 */
				if (xg<nextx1 || xg>nextx2 || yg<nexty1 || yg>nexty2) {
				    rank_v[il]=0;
				}

				xtg_speak(s,2,"RANK for cell (NEXT FAULT ID): %5.2f",rank_v[il]);

			    }
			    else{
				/* do not accept other faults */
				rank_v[il]=0;
			    }



			    if (rank_v[il]>0) {
				ncount4eval++;
				xtg_speak(s,2,"currentfid %d nextfid %d, curretext %d. Rank for %d %d is  %6.2f",
					  currentfid, nextfid, currentext, i, j, rank_v[il]);
				xtg_speak(s,2,"X0 X1 Y0 Y1 %6.2f %6.2f %6.2f %6.2f", xg0, xg, yg0, yg);
				xtg_speak(s,2,"Length and azimuth: %6.2f %6.2f (%6.2f)", vlen, azi_deg, currentdir);
				xtg_speak(s,2,"NCOUNT4EVAL: %d", ncount4eval);
				xtg_speak(s,2,"-------------------------------------\n");
			    }
			}
			else{
			    xtg_speak(s,2,"Cell <%d,%d> is outside search cone", i,j);
			}
		    }
		}
	    }
	    if (ncount4eval>0) {
		break;
	    }
	}

	xtg_speak(s,2,"Number of cells to be evaluated: %d: ",ncount4eval);
	xtg_speak(s,2,"Eval nradius and maxradius: %d %d: ",nradius, maxradius);

	if (nradius>=maxradius) {
	    xtg_warn(s,2,"(KLAYER %d) Max radius ... I surrender (give up). Return...",klayer);
	    ierr=0;
	    goto finalise;
	}


	xtg_speak(s,2,"Current radius for CELL <%d,%d,%d> is %d. Now determine ranks for surrounding cells:", ic, jc, klayer, nradius);

	maxrank=0;
	/* a rank is made; need to find the highest one with in the current radius*/
	for (j=jc-nradius;j<=jc+nradius;j++) {
	    for (i=ic-nradius;i<=ic+nradius;i++) {
		if (j>=1 && j<=ny && i>=1 && i<=nx) {

		    ib=x_ijk2ib(i,j,klayer,nx,ny,nz,0);
		    il=x_ijk2ib(i,j,1,nx,ny,1,0);

		    xtg_speak(s,2,"   CELL <%d,%d,%d>    RANK  %3.1f", i,j,klayer,rank_v[il]);


		    if (rank_v[il]>0) {
			if (rank_v[il]>maxrank) {
			    il_use=il;
			    ib_use=ib;
			    maxrank=rank_v[il];
			}
		    }
		}
	    }
	}

	/* cell is identified cell and coordinate will be stored */
	x_ib2ijk(ib_use,&i,&j,&k,nx,ny,nz,0);
	xtg_speak(s,2,"Cell %d %d has max rank: %6.3f", i, j, maxrank);


	grd3d_cellpoint(i,j,k,nx,ny,nz,nftype_v[ib_use],p_coord_v,
			p_zcorn_v,&xg,&yg,&zg,debug);


	npcounter++;
	p_xp_v[npcounter]=xg;
	p_yp_v[npcounter]=yg;
	iprev=ib_use;
	imark_v[il_use]=1;

	p_fprop2_v[ib_use]=p_fprop_v[ib_use];


	if (p_xp_v[npcounter]==p_xp_v[0] && p_yp_v[npcounter]==p_yp_v[0]) {
	    xtg_speak(s,2,"Polygon is closed!");
	    goto finalise;
	}

	/* find if the proposed cell is another fault, ie do an increment of ncurrentfault */
	if (nfprop_v[ib_use] != currentfid) {
	    if (ncurrentfault<nlastfault) {
		ncurrentfault=ncurrentfault+1;
		currentfid = fid[ncurrentfault];
		currentdir = dir[ncurrentfault];
		currentext = ext[ncurrentfault];
		currentx1  = xw1[ncurrentfault];
		currentx2  = xw2[ncurrentfault];
		currenty1  = yw1[ncurrentfault];
		currenty2  = yw2[ncurrentfault];

		if (ncurrentfault==nlastfault) {
		    /* the circle will be closed ... */
		    nextfid=fid[0];
		    nextdir=dir[0];
		    nextpri=pri[0];
		    nextx1=xw1[0];
		    nextx2=xw2[0];
		    nexty1=yw1[0];
		    nexty2=yw2[0];
		    imark_v[il_first]=0;  /* make the first cell with code 0 so it can be tracked once more */
		}
		else{
		    nextfid=fid[ncurrentfault+1];
		    nextdir=dir[ncurrentfault+1];
		    nextpri=pri[ncurrentfault+1];
		    nextx1=xw1[ncurrentfault+1];
		    nextx2=xw2[ncurrentfault+1];
		    nexty1=yw1[ncurrentfault+1];
		    nexty2=yw2[ncurrentfault+1];
		}
	    }
	}


	xtg_speak(s,2,"Cell %d %d, coords are %10.1f %10.1f", i, j, xg, yg);

	/* remember the coords */
	xg0 = xg;
	yg0 = yg;

    }

 finalise:
    *np=npcounter+1;


    return(ierr);

}
