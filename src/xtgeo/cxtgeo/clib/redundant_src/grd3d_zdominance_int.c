/*
 * #############################################################################
 * Name:      grd3d_zdominance_int.c
 * Author:    JRIV@statoil.com
 * Created:   2015-09-11
 * Updates:   2015-09-16 Rewrote algorithm so it is general for several 
 *                       codes in one K column
 * #############################################################################
 * Copy the dominant K prop value to a full K column in another property. 
 * Output array could be the same as input array.
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_zcorn_v        grid geometry ZCORN (needed for DZ)
 *     p_actnum_v       ACTNUM array
 *     p_xxx_v          array (pointer) of input/output (integer)
 *     option           1 if dz weigthing, 0 otherwise
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     - strange mix of double and double
 * #############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_zdominance_int(
			  int   nx,
			  int   ny,
			  int   nz,
			  double *p_zcorn_v,
			  int   *p_actnum_v,
			  int   *p_xxx_v,
			  int   option,
			  int   debug
			  )

{

    /* locals */
    int ib, i, j, k, maxcode, ic, use_ic, trigger;
    double allsum, mostsum;
    double *dz, *dzsum, *xsum, *frac ;
    int   *codes;
    char s[24]="grd3d_zdominance_int";
    
    xtgverbose(debug);

    xtg_speak(s,1,"Entering %s ",s);

    if (option>0) {
	xtg_speak(s,2,"Use thickness weighting");
    }

    /* allocate temporary memomery */
    codes=calloc(nz+1,sizeof(int));

    xsum=calloc(256,sizeof(double)); /*assume max 255 codes 0..255 */
    frac=calloc(256,sizeof(double)); 

    dz=calloc(nx*ny*nz,sizeof(double));
    dzsum=calloc(nx*ny*nz,sizeof(double));

    /* compute dz and sum_dz ...*/
    grd3d_calc_dz(nx, ny, nz, p_zcorn_v, p_actnum_v, dz, 1, 0, debug);
    grd3d_calc_sum_dz(nx, ny, nz, p_zcorn_v, p_actnum_v, dzsum, 1, debug);

    /*
    * ==========================================================================
    * loop all cells, and concentrate on K column operations
    */

    xtg_speak(s,2,"Looping %s ",s);

    for (i=1;i<=nx;i++) {
        xtg_speak(s,3,"Looping I row: %d  ",i);

	for (j=1;j<=ny;j++) {

	    allsum=0.0;
	    maxcode=0;
	    xtg_speak(s,4,"Looping J column: %d  ",j);

	    /* initial find codes present in a column, collect and 
	       find max value for later looping */
	    xtg_speak(s,4,"Looping J column: %d  ",j);
	    for (k=1;k<=nz;k++) {
	      if (j==136) {
		xtg_speak(s,4,"Looping K : %d  ",k);
	      }

		codes[k]=-1;
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);


		if (option==0 && p_actnum_v[ib]==1) {
		    allsum=allsum+k;
		}
		else if (option==1 && allsum==0.0 && p_actnum_v[ib]==1) {
		    /* as dzsum is the same for all cells in a column...*/
		    allsum=dzsum[ib];
		}

		if (p_actnum_v[ib]==1) {
		    codes[k]=p_xxx_v[ib];
		}
		if (codes[k]>maxcode) maxcode=codes[k];
		
		if (maxcode>255) {
		    xtg_warn(s,2,"Warning: maxcode is %d in <%s> cell "
			     "(%d %d %d). Reset to 255.",maxcode,s, i,j,k); 
		    maxcode=255;
		}

	    }


	    /* for each code, do summing (and possibly weighted with dz) */
	    for (ic=0; ic<=maxcode; ic++) {
		xsum[ic]=0.0;
		for (k=1;k<=nz;k++) {
		    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		    if (codes[k]==ic){
			if (option==0) {
			    xsum[ic]=xsum[ic]+1.0;
			}
			else{
			    xsum[ic]=xsum[ic]+1.0*dz[ib];
			}
		    }
		}
	    }
	
	    if (allsum<0.01) allsum=0.01; /* avoid zero division */

	    /* now find which xsum which is largest and use that one */
	    mostsum=0.0;
	    use_ic=0;
	    trigger=0;
	    for (ic=0; ic<=maxcode; ic++) {		
		/* fractions */
		frac[ic]=xsum[ic]/allsum;

		if (frac[ic]>0.0001 && frac[ic]<0.9999) trigger=1;

		if (frac[ic]>mostsum) {
		    mostsum=frac[ic];
		    use_ic=ic;
		}

		if (debug > 2 && frac[ic] > 0.01 && frac[ic] < 0.99) {
		    xtg_speak(s,3,"Frac for %d %d column, code %d, is %4.3f", 
			      i,j,ic,frac[ic]);
		}

		
	    }

	    if (debug > 2 && frac[ic] > 0.01 && frac[ic] < 0.99) {
		xtg_speak(s,3,"Use code <%d> for %d %d column", use_ic, i,j);
	    }


	    /* now populate the full cell column with one single value, 
	       if trigger is true */
	    if (trigger==1) {
		for (k=1;k<=nz;k++) {
		    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		    p_xxx_v[ib]=use_ic;
		}
	    }
	}
    }
    
    xtg_speak(s,2,"Freeing memory ... ");
    xtg_speak(s,2,"... dz");
    free(dz);
    xtg_speak(s,3,"... dzsum");
    free(dzsum);
    xtg_speak(s,3,"... frac");
    free(frac);
    xtg_speak(s,3,"... xsum");
    free(xsum);
    xtg_speak(s,3,"... codes");
    free(codes);
    xtg_speak(s,2,"Freeing memory ... done");

    xtg_speak(s,2,"Exit grd3d_zdominance_int");
}
