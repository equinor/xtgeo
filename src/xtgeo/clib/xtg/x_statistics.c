/*
 * ############################################################################
 * Some simple routines for statistics of arrays
 * JRIV
 * ############################################################################
 */



#include "libxtg.h"
#include "libxtg_.h"


/*
 * ############################################################################
 * Basic statistics of a float array
 * n        Number of elements
 * undef    Undef value for that actual array (to be ignored in stats)
 * min      Minimum value (pointer return)
 * max      Maximum value (pointer return)
 * avg      Average (aritm. mean) value (pointer return)
 * debug    Debug flag
 * JRIV
 * ############################################################################
 */

void x_basicstats (
		   int n,
		   double undef,
		   double *v,
		   double *min,
		   double *max,
		   double *avg,
		   int debug
		   )
{
    int   i, m;
    double sum, vmin, vmax;

    vmin=VERYLARGEFLOAT;
    vmax=VERYSMALLFLOAT;

    m=0;
    sum=0.0;

    for (i=0;i<n;i++) {
	if (v[i] != undef) {
	    if (v[i]<vmin) vmin=v[i];
	    if (v[i]>vmax) vmax=v[i];
	    sum=sum+v[i];
	    m++;
	}
    }

    /* results */
    if (m>0) {
	* avg = sum/m;
    }
    *min = vmin;
    *max = vmax;

}

void x_basicstats2 (
		   int n,
		   float undef,
		   float *v,
		   float *min,
		   float *max,
		   float *avg,
		   int debug
		   )
{
    int   i, m;
    float sum, vmin, vmax;

    vmin=VERYLARGEFLOAT;
    vmax=VERYSMALLFLOAT;

    m=0;
    sum=0.0;

    for (i=0;i<n;i++) {
	if (v[i] != undef) {
	    if (v[i]<vmin) vmin=v[i];
	    if (v[i]>vmax) vmax=v[i];
	    sum=sum+v[i];
	    m++;
	}
    }

    /* results */
    if (m>0) {
	* avg = sum/m;
    }
    *min = vmin;
    *max = vmax;

}
