/*
 * ############################################################################
 * pox_operation_pox.c
 *
 * Description:
 * Do operations (add, subtract, etc) between points and points, leaving the
 * the result in points. If point sets are different, then the points that
 * have th same XY are used (unless option is...)
 * iop=1  add
 * iop=2  subtract
 * iop=3  multiply
 * iop=4  divide
 * Bugs or potential problems:
 *
 * The routine return updated versions of p_*1_v and the number of points
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

int pox_operation_pox (
		       double *p_x1_v,
		       double *p_y1_v,
		       double *p_z1_v,
		       int   np1,
		       double *p_x2_v,
		       double *p_y2_v,
		       double *p_z2_v,
		       int   np2,
		       int   iop,
		       double toldfloat,
		       int   debug
		       )
{

    int     i, j, k, npnew;
    double   tol;
    double   *p_tmpx_v, *p_tmpy_v, *p_tmpz_v;

    char s[24]="pox_operation_pox";
    xtg_speak(s,2,"Entering routine...");

    tol=toldfloat;

    xtgverbose(debug);

    if (iop==1) {
	xtg_speak(s,2,"Adding ...");
    }
    else if (iop==2) {
	xtg_speak(s,2,"Subtracting ...");
    }
    else if (iop==3) {
	xtg_speak(s,2,"Multiply ...");
    }
    else if (iop==4) {
	xtg_speak(s,2,"Divide ...");
    }
    else if (iop==5) {
	xtg_speak(s,2,"Merge ...");
    }
    else {
	xtg_error(s,"Illegal operation! STOP!");
    }

    /* this is a bit rough ... allocate a tmp pointer using same size as largest np */
    npnew=np1;
    if (np1 < np2) npnew=np2;
    p_tmpx_v=calloc(npnew,sizeof(double));
    p_tmpy_v=calloc(npnew,sizeof(double));
    p_tmpz_v=calloc(npnew,sizeof(double));

    /* for each point in set1, a search must be done to find similar XY point in set 2 */

    k=0;
    for (i=0;i<np1;i++) {
	if (p_x1_v[i] < UNDEF_LIMIT) {
	    if (iop==5) {
		k=i;
		p_tmpx_v[k]=p_x1_v[i];
		p_tmpy_v[k]=p_y1_v[i];
		p_tmpz_v[k]=p_z1_v[i];
	    }

	    for (j=0;j<np2;j++) {
		if ((fabs(p_x1_v[i]-p_x2_v[j]))<tol && (fabs(p_y1_v[i]-p_y2_v[j]))<tol) {

		    /* add */
		    if (iop==1) {
			p_tmpz_v[k]=p_z1_v[i] + p_z2_v[j];
			p_tmpx_v[k]=p_x1_v[i];
			p_tmpy_v[k]=p_y1_v[i];
			k++;
		    }

		    /* subtract */
		    if (iop==2) {
			p_tmpz_v[k]=p_z1_v[i] - p_z2_v[j];
			p_tmpx_v[k]=p_x1_v[i];
			p_tmpy_v[k]=p_y1_v[i];
			k++;
		    }

		    /* multiply */
		    if (iop==3) {
			p_tmpz_v[k]=p_z1_v[i] * p_z2_v[j];
			p_tmpx_v[k]=p_x1_v[i];
			p_tmpy_v[k]=p_y1_v[i];
			k++;
		    }

		    /* divide */
		    if (iop==4) {
			/* avoid division on zero */
			if (fabs(p_z2_v[j]) < FLOATEPS) p_z2_v[i]=FLOATEPS;
			p_tmpz_v[k]=p_z1_v[i] / p_z2_v[j];
			p_tmpx_v[k]=p_x1_v[i];
			p_tmpy_v[k]=p_y1_v[i];
			k++;
		    }

		    /* merge */
		    if (iop==5) {
			p_tmpz_v[k]=p_z2_v[j];
			// xtg_speak(s,3,"Replacing %f with %f", p_z1_v[k], p_tmpz_v[k]);
		    }
		}
	    }
	}
    }


    if (iop==5) k=k+1; // just to get numbering right


    for (i=0;i<k;i++) {
	p_x1_v[i] = p_tmpx_v[i];
	p_y1_v[i] = p_tmpy_v[i];
	p_z1_v[i] = p_tmpz_v[i];
	xtg_speak(s,2,"Coord %d   %f %f %f",k,p_x1_v[i],p_y1_v[i],p_z1_v[i]);
    }

    /* return number of points */

    xtg_speak(s,2,"Returning %d points",k);
    xtg_speak(s,2,"Exiting ...");
    return(k);

}
