/*
 * ############################################################################
 * Name: map_sample_grd3d_prop
 * By:   JCR
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Input is an integher map, wher the value of each node is the ib value
 * of the 3D grid to sample from. Type=1 means double
 * ############################################################################
 */

void map_sample_grd3d_prop(
			   int   nx,
			   int   ny,
			   int   nz,
			   int   type,
			   int   *p_iprop_v,
			   double *p_fprop_v,
			   int   mx,
			   int   my,
			   double *p_zval_v, 
			   int   *p_ib_v,   
			   int   debug
			   )
     
{
    /* locals */
    char  s[24]="map_sample_grd3d_prop"; 
    int   im, ib, ibmax;
  
    xtgverbose(debug);

    xtg_speak(s,2,"Entering routine %s ...",s);
    
    xtg_speak(s,2,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,2,"MX and MY is: %d %d", mx, my);

    ibmax=nx*ny*nz;

    for (im=0; im < mx*my-1; im++) {
	ib=p_ib_v[im];
	if (ib<UNDEF_INT_LIMIT && ib>=0 && ib<=ibmax) {
	    if (type==1) {
		p_zval_v[im]=p_fprop_v[ib];
	    }
	    else{
		p_zval_v[im]=p_iprop_v[ib];
	    }
	}
    }
    xtg_speak(s,2,"Exiting routine");
}
