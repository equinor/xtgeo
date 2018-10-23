/*
 * ############################################################################
 * pox_zsum.c
 * 
 * Description:
 * Returns the sum of the Z values (normally useful in testing other routines)
 * 
 * Bugs or potential problems:
 *
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

double pox_zsum (
		 double *p_z1_v, 
		 int     np1, 
		 int     debug
		 ) 
{
    
    int     i;
    double  sum;

    char s[24]="pox_zsum"; 
    xtg_speak(s,2,"Entering routine...");
      
    xtgverbose(debug);
	
    xtg_speak(s,2,"Summing Z values ...");
    
    sum=0;

    for (i=0;i<np1;i++) {
	if (p_z1_v[i] < UNDEF_LIMIT) {
	    sum=sum+p_z1_v[i];
	}
    }

    xtg_speak(s,2,"Exiting ...");
    return(sum);

}

