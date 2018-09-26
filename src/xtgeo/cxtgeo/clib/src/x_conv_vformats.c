/*
 * ############################################################################
 * Some simple routines for converting vector formats
 * n      Number of entries
 * v1     input vector of format X
 * v2     output vector of format y
 * JRIV
 * ############################################################################
 * ############################################################################
 */



#include "libxtg.h"
#include "libxtg_.h"

int x_conv_int2float (
		      int n,
		      int *v1,
		      float *v2,
		      int debug
		      )
{
    int i;
    
    
    
    for (i=0;i<n;i++) {
	v2[i]=(float)v1[i];
    }
    
    return 1;
}

int x_conv_int2double (
		       int n,
		       int *v1,
		       double *v2,
		       int debug
		      )
{
    int i;
    
    
    
    for (i=0;i<n;i++) {       
	v2[i]=(double)v1[i];
	if (v2[i] > UNDEF_INT_LIMIT) v2[i]=UNDEF;
    }
    
    return 1;
}


int x_conv_double2float (
			 int n,
			 double *v1,
			 float  *v2,
			 int debug
			 )
{
    int i;
    
    
    
    for (i=0;i<n;i++) {
	v2[i]=(float)v1[i];
    }
    
    return 1;
}

