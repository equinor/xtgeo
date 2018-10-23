/*
 * ############################################################################
 * map_set_value.c
 * Set constant values for maps. 
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id:  $ 
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

   
void map_set_value (
		    int nx,
		    int ny,
		    double *p_zval_v,
		    double value,
		    int debug
		    )
{
    int i, j, ib;

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {
	
	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);
	    p_zval_v[ib]=value;
	}
    }
}

