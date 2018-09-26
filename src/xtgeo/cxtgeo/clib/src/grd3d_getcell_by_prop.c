/*
 * ##################################################################################################
 * Name:      grd3d_getcell_by_prop.c
 * Author:    JRIV@statoil.com
 * Created:   2001-12-12
 * Updates:   
 * #################################################################################################
 * Search prop (interval) based on coded criteria, and return the ib code 
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     i1, ... k2       cell IJK range
 *     p_xxx_v          Property to evaluate (double)
 *     criteria code    1: most to the north, 2: most east, 3:most south, 4:most west
 *     debug            debug/verbose flag
 *
 * Return:
 *     ib               Cell index (1D array counting)
 * Caveeats/issues:
 *     ACTNUM is ignored.
 * #################################################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


int grd3d_getcell_by_prop(
			  int      nx,
			  int      ny,
			  int      nz,
			  int      i1,
			  int      i2,
			  int      j1,
			  int      j2,
			  int      k1,
			  int      k2,
			  double   *p_coord_v,
			  double   *p_zcorn_v,
			  int      *p_actnum_v,
			  double    *p_xxx_v,
			  double    pmin,
			  double    pmax,
			  int      criteria,
			  int      option,
			  int      debug
			  )

{
    /* locals */
    int     i, j, k, ib, ibuse;
    double  nmax, emax, nmin, emin, x, y, z;
    char    s[24]="grd3d_getcell_by_prop";

    nmax=-9999999;
    nmin=9999999;
    emax=-9999999;
    emin=9999999;
    
    ibuse=-1;

    xtgverbose(debug);
    xtg_speak(s,2,"Entering %s",s);

    xtg_speak(s,2,"Finding parameter based on criteria %d ...",criteria);


    for (k = k1; k <= k2; k++) {
	xtg_speak(s,3,"Working with layer %d of %d",k,nz);
	for (j = j1; j <= j2; j++) {
	    for (i = i1; i <= i2; i++) {

		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		xtg_speak(s,5,"IB is %d",ib);
		
		if (p_xxx_v[ib]>=pmin && p_xxx_v[ib]<=pmax) {
		    /* need the cell midpoint coordinates */
		
		    xtg_speak(s,3,"PROP is %f (%d %d %d)",p_xxx_v[ib],i,j,k);

		    grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v, p_zcorn_v, &x, &y, &z, debug);
		    /* 1, look for most northern cell */
		    if (criteria==1) {
			if (y>nmax) {
			    nmax=y;
			    ibuse=ib;
			}
		    }			
		
		    /* 2, look for most eastern cell */
		    if (criteria==2) {
			if (x>emax) {
			    emax=x;
			    ibuse=ib;
			}
		    }			
		
		    /* 1, look for most southern cell */
		    if (criteria==3) {
			if (y<nmin) {
			    nmin=y;
			    ibuse=ib;
			}
		    }			
		
		    /* 1, look for most western cell */
		    if (criteria==4) {
			if (x<emin) {
			    emin=x;
			    ibuse=ib;
			}
		    }			
		    
		}	   
	    }
	}
    }
    xtg_speak(s,3,"IBUSE is %d",ibuse);
    xtg_speak(s,2,"Exiting <%s>",s);

    return(ibuse);
}
