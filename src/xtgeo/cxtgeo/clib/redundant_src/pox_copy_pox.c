/*
 * ############################################################################
 * pox_copy_pox.c
 * 
 * Description:
 * Copy one polygon (xyz arrays) to another
 *
 * The routine return updated versions of p_*1_v. The vector must be allocated
 * before this happens.
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

int pox_copy_pox (
		  int   np, 
		  double *p_x1_v, 
		  double *p_y1_v, 
		  double *p_z1_v, 
		  double *p_x2_v, 
		  double *p_y2_v, 
		  double *p_z2_v, 
		  int   debug
		  ) 
{
    
    int     i;
    
    char s[24]="pox_copy_pox"; 
    xtg_speak(s,2,"Entering routine...");
    
    xtgverbose(debug);
	
    xtg_speak(s,2,"Numbers of points are: %d",np);

    for (i=0;i<np;i++) {
	if (p_x2_v[i] < UNDEF_LIMIT) {
	    p_x1_v[i]=p_x2_v[i];
	    p_y1_v[i]=p_y2_v[i];
	    p_z1_v[i]=p_z2_v[i];
	}
    }

    xtg_speak(s,2,"Returning %d points",np);
    xtg_speak(s,2,"Exiting ...");
    return(1);
}

