/*
 * ############################################################################
 * pol_refine.c
 *
 * Description:
 * Refine a polygon by a given max distance
 *
 * Author: J.C. Rivenaes JRIV@statoil.com
 * ############################################################################
 */

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
		int    option,  /* 0: look in 3D, 1: look in 2d XY */
		int    debug
		)
{

    int     i, n, m, ii, newnpmax, iostat;
    double  dist2d, dist3d, usedist, len, frac, x, y, z;
    double  *xv, *yv, *zv;

    xtgverbose(debug);

    char s[24]="pol_refine";
    xtg_speak(s,2,"Entering routine %s", s);

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
	dist3d=sqrt(pow(p_x_v[i]-p_x_v[i+1],2) + pow(p_y_v[i]-p_y_v[i+1],2) + pow(p_z_v[i]-p_z_v[i+1],2));

	if (option==1) {
	    usedist=dist2d;
	}
	else{
	    usedist=dist3d;
	}

	/* find he amount to refine */
	n = 1 + (usedist/dist);

	xtg_speak(s,3,"Distance: %9.2f, from X point %9.2f, to X point %9.2f (intervals: %d)",
		  usedist, p_x_v[i], p_x_v[i+1], n);

	if (n>1) {
	    len=usedist/n;
            xtg_speak(s, 2, "LEN is %d", len);
	    for (ii=1;ii<n; ii++) {

		frac = (len/usedist)*ii;

		if (frac>1) {
		    xtg_error(s,"Something is wrong in <%s>, contact JRIV",s);
		}

		iostat=x_vector_linint(p_x_v[i],p_y_v[i],p_z_v[i],
				       p_x_v[i+1],p_y_v[i+1],p_z_v[i+1],
				       frac, &x, &y, &z, debug);
		if (iostat<0) {
		    xtg_error(s,"Null vector. Something failed");
		}


		/* add the new point to the array */
		m=m+1;
		xv[m]=x;
		yv[m]=y;
		zv[m]=z;

		xtg_speak(s,3,"M=%d ... New X Y z: %9.2f  %9.2f",
			  m, xv[m], yv[m]);

	    }
	}

    }

    newnpmax=m+1;
    /* the new x_v etc array has m+1 entries total. check against npmax */
    if (newnpmax > npmax) {
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
