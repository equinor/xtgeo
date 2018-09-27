/*
 * ############################################################################
 * x_stretch_vector.c
 *
 * This routine takes a _sorted_ array of floats, and do a squeeze or
 * stretch between two given nodes. The stretch is proportional. Outside
 * an equal movement is done.
 * 
 * Author: Jan C. Rivenaes
 * ############################################################################
 * Arguments:
 * z_v        The float vector, input and output
 * nz         Number of elements (0..nz-1)
 * k1         Position of first change (numbering system as 1..nz)
 * k2         Position of last change (numbering system as 1..nz)
 * z1n        New value in k=k1
 * z2n        New value in k=k2
 * debug      Standard flag for debugging
 *
 *
 * Example:
 * 0.2  0.3  0.5  0.8  0.9  0.9  1.4  1.8
 *      k1                  k2
 *
 * z1n=0.2 z2n=1.4   >>>  interval from 0.6 to 1.2. Stretch factor fz=2 total
 *
 * 0.1  0.2  0.6  1.2  1.4  1.4  1.9  2.3 
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void x_stretch_vector (
		       double *z_v,
		       int   nz,
		       int   k1,
		       int   k2,
		       double z1n,
		       double z2n,
		       int debug
		       )
{
    /* locals */
    int k;
    double fz, *zn_v, zndiff, diff1, diff2;
    char sub[24]="x_stretch_vector";
    
    //    xtgverbose(debug);

    xtg_speak(sub,4,"Entering routine");

    k=1;

    /* check */
    for (k=1;k<nz;k++) {
	if (z_v[k]< z_v[k-1]) {
	    xtg_error(sub,"Error in input to routine %s. Numbers must increase", sub);
	    return;
	}
    }

    if (debug==5) {
	xtg_speak(sub,1,"XX K1 and K2 is %d %d %f %f",k1,k2,z_v[k1-1],z_v[k2-1]);
    }
    
    zn_v=calloc(nz,sizeof(double)); /* new vector */

    zndiff=(z_v[k2-1]-z_v[k1-1]);
    if (debug==5) printf("XX zndiff %f\n",zndiff);
    if (zndiff<FLOATEPS) zndiff=FLOATEPS;

    if (debug==5) {
	printf("XX zndiff %f AFTER floateps\n",zndiff);
    }
    

    fz=(z2n-z1n)/zndiff;  /* stretch factor */


    diff1=z1n-z_v[k1-1];
    diff2=z2n-z_v[k2-1];

    if (k1==k2) diff1=0;
    if (k1==k2) diff2=0;

    for (k=0;k<nz;k++) {
	if (k<=(k1-1)) {

	    zn_v[k]=z_v[k]+diff1;

	}else if (k>=(k2-1)) {
	    zn_v[k]=z_v[k]+diff2;
	}
	else{
	    zn_v[k]=zn_v[k-1]+fz*(z_v[k]-z_v[k-1]);
	}
    }

    if (debug==5) {
	xtg_speak(sub,1,"XX k1 = %d  k2 = %d   factor = %f8.5",k1,k2,fz);
	for (k=0;k<nz;k++) {
	    xtg_speak(sub,1,"XX %d : Old value %f   New value %f", k+1, z_v[k], zn_v[k]);
	}
    }

    /* final mapping */
    for (k=0;k<nz;k++) {
	z_v[k] = zn_v[k];
    }

    free (zn_v);
    
}


