/*
 * #############################################################################
 * Name:      grd3d_interp_prop_vertii.c (INT version of 
 *            grd3d_interp_prop_verti.c)
 * Author:    JRIV@statoil.com
 * Created:   2015-09-30
 * Updates:   
 * #############################################################################
 * Copy the dominant K prop value to a full K column in another property. 
 * Output array could be the same as input array. 
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_zcorn_v        grid geometry ZCORN (needed for DZ)
 *     p_actnum_v       ACTNUM array
 *     p_xxx_v          array (pointer) of input/output (integer)
 *     bgval            back ground value (to be replaced)
 *     option           1 if ACTNUM is ignored
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     - strange mix of double and double
 * #############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_interp_prop_vertii(
			      int    nx,
			      int    ny,
			      int    nz,
			      int    i1,
			      int    i2,
			      int    j1,
			      int    j2,			      
			      double *p_zcorn_v,
			      int    *p_actnum_v,
			      int    *p_xxx_v,
			      int    bgval,
			      double  dzmax,
			      int    option,
			      int    debug
			      )
    
{
    
    /* locals */
    int ib, i, j, k, use_ib, use_k, trigger;
    int kk, ka, kb, ibka, ibkb, kfound, krad, kwork;
    double *dz;
    char s[24]="grd3d_interp...vertii";
    
    xtgverbose(debug);

    xtg_speak(s,2,"Enter %s",s);

    /* allocate temporary memomery */

    dz=calloc(nx*ny*nz,sizeof(double));

    /* compute dz and sum_dz ...*/
    grd3d_calc_dz(nx, ny, nz, p_zcorn_v, p_actnum_v, dz, 1, 0, debug);

    /*
    * ==========================================================================
    * loop all cells, and concentrate on K column operations
    */

    for (i=i1;i<=i2;i++) {
	for (j=j1;j<=j2;j++) {

	    /* scan (to avoid wasting time) */
	    kwork=0;
	    for (k=1;k<=nz;k++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (p_xxx_v[ib] != bgval) kwork=1;
	    }

	    if (kwork==0) break;

	    /* initial find codes present in a column, collect and 
	       find max value for later looping */
	    for (k=1;k<=nz;k++) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		trigger=0;

		if (option==1) {
		    if (p_xxx_v[ib]==bgval && dz[ib]<=dzmax) {
			trigger=1;
		    }
		}
		else{
		    if (p_actnum_v[ib]>=1 && p_xxx_v[ib]==bgval && 
			dz[ib]<=dzmax) {
			trigger=1;
		    }
		}

		if (trigger==1) {

		    if (debug>2 ) {
			xtg_speak(s,3,"TRIGGER for I J %d %d K=%d "
				  "value= %7.2f (DZ=%8.5f) ",
				  i,j,k, p_xxx_v[ib],dz[ib]);
		    }
		    /* a KRADIUS search */
		    kfound=0;
		    for (krad=1;krad<nz;krad++) {
			for (kk=1; kk<=krad; kk++) {
			    ka=k-kk;
			    if (ka<1) ka=1;

			    kb=k+kk;
			    if (kb>nz) kb=nz;

			    /* now check above and below cell */
			    ibka=x_ijk2ib(i,j,ka,nx,ny,nz,0);
			    ibkb=x_ijk2ib(i,j,kb,nx,ny,nz,0);
			    
			    use_ib=-1;
			    use_k=0;

			    if (ibka!=ib && ((option==0 && p_actnum_v[ibka]>=1) 
					     || (option==1 
						 &&  p_actnum_v[ibka]>=0)) 

				&& (dz[ibka]>dzmax || p_xxx_v[ibka] != bgval)) {

				use_ib=ibka;
				use_k=ka;
			    }

			    /* if above is not found; try below */
			    if (use_ib<0 && ibkb!=ib && 
				((option==0 && p_actnum_v[ibkb]>=1) 
				 || (option==1 && p_actnum_v[ibkb]>=0)) 

				&& (dz[ibkb]>dzmax || p_xxx_v[ibka] != bgval)) {

				use_ib=ibkb;
				use_k=kb;
			    }

			    if (use_ib>=0)  {
				p_xxx_v[ib]=p_xxx_v[use_ib];
				kfound=1;
				if (debug>2 ) {
				    xtg_speak(s,3,"Cell I J %d %d K=%d "
					      "inherits value %7.2f from K=%d",
					      i,j,k, p_xxx_v[ib],use_k);
				}
			    }
			    if (kfound==1) break;
			}
			if (kfound==1) break;			
		    }
		    if (kfound==0 && trigger==1 && debug>2) {
			xtg_speak(s,3,"No valid value found for cell "
				  "%d %d %d (value %7.2f, DZ=%6.5f) ACTNUM=%d",
				  i,j,k, p_xxx_v[ib],dz[ib],p_actnum_v[ib]);
		    }
		    trigger=0;
		}
	    }	    
	} 
    }

    
    free(dz);

    xtg_speak(s,2,"Exit %s",s);
}
