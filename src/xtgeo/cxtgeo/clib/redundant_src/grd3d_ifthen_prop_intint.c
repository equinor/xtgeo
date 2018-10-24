/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Copy from one integer array to another
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_ifthen_prop_intint(
			      int   nx,
			      int   ny,
			      int   nz,
			      int   i1,
			      int   i2,
			      int   j1,
			      int   j2,
			      int   k1,
			      int   k2,
			      int   *p_other_v,
			      int   *p_this_v,
			      int   *p_array,
			      int   alen,
			      int   newvalue,
			      int   elsevalue,
			      int   rmin,
			      int   rmax,
			      int   debug
			      )
    
{
    /* locals */
    int ib, nn,i,j,k;
    char s[24]="grd3d_ifthen_prop_inti-";


    /* values in range rmin, rmax will be replaced, but skip this test if both rmin and rmax are -999 */

    xtgverbose(debug);

    for (k=k1;k<=k2;k++) {
	for (j=j1;j<=j2;j++) {
	    for (i=i1;i<=i2;i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		for (nn = 0; nn < alen; nn++) {
		    if (p_other_v[ib]==p_array[nn]) {
			/* xtg_speak(s,2,"Cell %5d (value %5d) is assigned value %5d",ib,p_other_v[ib],newvalue); */
			
			if (rmin == -999 && rmin == -999){ 
			    p_this_v[ib]=newvalue;
			}
			else{
			    if (p_this_v[ib]>=rmin && p_this_v[ib]<=rmin) {
				p_this_v[ib]=newvalue;
			    }
			}
		    }
		    else{
			if (elsevalue != -999) {
			    if (rmin == -999 && rmin == -999){ 
				p_this_v[ib]=elsevalue;
			    }
			    else{
				if (p_this_v[ib]>=rmin && p_this_v[ib]<=rmin) {
				    p_this_v[ib]=elsevalue;
				}
			    }		    
			}
		    }
		}
	    }			    
	}
    }
    xtg_speak(s,2,"Exit ifthen (int int mode) for properties");
}

