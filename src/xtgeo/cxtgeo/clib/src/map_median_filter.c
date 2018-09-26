/*
 * ############################################################################
 * map_median_filter.c
 * Median filter for maps.
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: map_median_filter.c,v 1.1 2000/12/12 17:24:54 bg54276 Exp $
 * $Source: /h/bg54276/jcr/prg/lib/gplext/GPLExt/RCS/map_median_filter.c,v $
 *
 * $Log: map_median_filter.c,v $
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 * General description:
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                          GRD2D_MEDIAN_FILTER
 * ****************************************************************************
 * For a map, collect values within a  search radius, sort, and
 * return the median value. Return modified map. The "mode" determines
 * sub-options.
 * ----------------------------------------------------------------------------
 *
 */
int map_median_filter (
			   double *map_in_v,
			   int nx,
			   int ny,
			   int nradius,
			   int mode
			   )
{



    const int mx = MAXPSTACK;
    double pstack[MAXPSTACK];
    int i, ii, j, jj, ip, jp, ib=0, ibb, k, nqi, nqj;

    double *map_tmp;

    /* Allocate memory for map_tmp */
    map_tmp=calloc(nx*ny, sizeof(double));

    if (mode == 1) {
	mode=1; /* so what ...? */
    }


    if (pow((2*nradius),2) > mx) {
	return -99;
    }

    /* test every point */

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {

	    /*compute actual cell in 1D array*/
	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);

	    /*skip UNDEF values*/
	    if (map_in_v[ib] < UNDEF_MAP_LIMIT) {

		/*now ry +- nradius values and save in pstack if defined*/
		k=0;
		for (jj=-nradius;jj<=nradius;jj++) {
		    for (ii=-nradius;ii<=nradius;ii++) {
			ip=i+ii;
			jp=j+jj;

			nqi=0;
			if (ip >= 1 && ip <= nx) nqi=1;
			nqj=0;
			if (jp >= 1 && jp <= ny) nqj=1;

			ibb=x_ijk2ib(ip,jp,1,nx,ny,1,0);

			if (nqi && nqj && map_in_v[ibb] < UNDEF_MAP_LIMIT) {
			    k++;
			    pstack[k]=map_in_v[ibb];
			}
		    }
		}

		/* To get the median, I need to sort pstack*/
		/* Use the built-in qsort function */
		qsort(pstack, k, sizeof(double), x_cmp_sort);

		/*find median*/
		if (k % 2  == 0) {
		    map_tmp[ib]=(pstack[k/2 - 1] + pstack[k/2]) * (double).5;
		}
		else{
		    map_tmp[ib]=pstack[(k + 1) / 2 - 1];
		}
	    }
	    else {
                map_tmp[ib] = map_in_v[ib]; /*UNDEF*/
            }
	}
    }

    for (i=0;i<ib;i++) map_in_v[ib] = map_tmp[ib]; /* copy */

    /* free memory */
    free (map_tmp);

    return 1;
}
