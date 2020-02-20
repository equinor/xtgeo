/*
 * ############################################################################
 * pol_refine.c
 *
 * Description:
 * Refine a polygon by a given max distance
 *
 * ############################################################################
 */

#include "logger.h"
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

int pol_refine (
    int    np,
    int    npmax,
    double *p_x_v,
    double *p_y_v,
    double *p_z_v,
    double dist,
    int    option  /* 0: look in 3D, 1: look in 2d XY */
    )
{

    int     i, n, m, ii, newnpmax, iostat;
    double  dist2d, dist3d, usedist, len, frac, x, y, z;
    double  *xv, *yv, *zv;

    logger_info(LI, FI, FU, "Entering routine %s", __FUNCTION__);

    /* allocate the tmp arrays to something big */
    xv=calloc(99999,sizeof(double));
    yv=calloc(99999,sizeof(double));
    zv=calloc(99999,sizeof(double));


    /* look at each segment and measure distance */
    m=-1;
    for (i=0; i<np; i++) {

	m=m+1;
	xv[m]=p_x_v[i];
	yv[m]=p_y_v[i];
	zv[m]=p_z_v[i];

	if (i==(np-1)) break; /*last point */

	dist2d=sqrt(pow(p_x_v[i]-p_x_v[i+1],2) + pow(p_y_v[i]-p_y_v[i+1],2));
	dist3d=sqrt(pow(p_x_v[i]-p_x_v[i+1],2) + pow(p_y_v[i]-p_y_v[i+1],2) +
                    pow(p_z_v[i]-p_z_v[i+1],2));

	if (option==1) {
	    usedist=dist2d;
	}
	else{
	    usedist=dist3d;
	}

	/* find he amount to refine */
	n = 1 + (usedist/dist);

	if (n>1) {
	    len=usedist/n;
	    for (ii=1;ii<n; ii++) {

		frac = (len/usedist)*ii;

		if (frac>1) {
		    logger_critical(LI, FI, FU,"Bug in %s (frac > 1)", __FUNCTION__);
		}

		iostat=x_vector_linint(p_x_v[i],p_y_v[i],p_z_v[i],
				       p_x_v[i+1],p_y_v[i+1],p_z_v[i+1],
				       frac, &x, &y, &z, 0);
		if (iostat<0) {
		    logger_critical(LI, FI, FU,"Bug in %s (iostat < 0)", __FUNCTION__);
		}


		/* add the new point to the array */
		m=m+1;
		xv[m]=x;
		yv[m]=y;
		zv[m]=z;

	    }
	}

    }

    newnpmax=m+1;
    /* the new x_v etc array has m+1 entries total. check against npmax */
    if (newnpmax > npmax) {
	free(xv);
	free(yv);
	free(zv);
	return(0);
    }
    else{

	/* update array */
	for (i=0; i<newnpmax; i++) {
	    p_x_v[i]=xv[i];
	    p_y_v[i]=yv[i];
	    p_z_v[i]=zv[i];
	}

	free(xv);
	free(yv);
	free(zv);

	return(newnpmax);
    }
}
