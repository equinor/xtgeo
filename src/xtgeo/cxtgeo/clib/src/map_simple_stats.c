/*
 * ############################################################################
 * map_simple_stats.c
 * Simple statistics for a map
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 * General description:
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void map_simple_stats (
		       double *p_map_in_v,
		       int    nx,
		       int    ny,
		       double *zmin,
		       double *zmax,
		       double *mean,
		       int    *ndef,
		       int    *sign,
		       int    debug
		       )
{



    int i, j, ib, n, mj, mi, nsign;
    double sum, value, limit;
    char s[24]="map_simple_stats";
    
    xtgverbose(debug);


    xtg_speak(s,2,"Do some basic statistics ...");
    /* initially */
    limit=UNDEF_MAP_LIMIT;
    *zmin=limit;
    *zmax=-limit;
    sum=0.0;
    n=0;
    nsign=0;

    for (j=1;j<=ny;j++) {
	mj=j % 2; /* modulus 2 */
	for (i=1;i<=nx;i++) {

	    mi=i % 4; /* modulus 4 */

	    /*compute actual cell in 1D array*/
	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);
	    
	    value=p_map_in_v[ib];
	    /*skip UNDEF values*/
	    if (value < UNDEF_MAP_LIMIT) {
		n++;
		sum+=value;
		if (value >= *zmax) *zmax=value;
		if (value <= *zmin) *zmin=value;
		
	    }
	    
	    /* the follwing is used to compute the "SIGN" */
	    if (value < UNDEF_MAP_LIMIT && mj==0 && mi==0) {
		nsign++;
	    }
	    
	}
    }
    *mean=sum / n;
    *ndef=n;
    *sign=nsign+nx;
    xtg_speak(s,2,"Do some basic statistics ... DONE!");
}


