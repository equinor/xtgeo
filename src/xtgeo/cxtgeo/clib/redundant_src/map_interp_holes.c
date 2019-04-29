/*
 *******************************************************************************
 *
 * Interpolate holes in map
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    map_interp_holes.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Tries to make undefined map values defined by interpolation in both
 *    direction, and use inverse distance weighting
 *
 * ARGUMENTS:
 *    mx, my         i     map dimensions
 *    p_zval_v      i/o    map values
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success.
 *    p_zval_v is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */
int map_interp_holes(int mx, int my, double *p_zval_v, int option,
		     int debug)
{

    int i, ii, j, jj, ib=0, ibn=0;
    double px1, px2, py1, py2, px1w, px2w, py1w, py2w, sumw;
    char s[24]="map_interp_holes";

    xtgverbose(debug);

    /* test every point */

    for (j=1;j<=my;j++) {
	for (i=1;i<=mx;i++) {

	    ib=x_ijk2ib(i,j,1,mx,my,1,0);

	    /* find UNDEF values*/
	    if (p_zval_v[ib] > UNDEF_MAP_LIMIT) {
		xtg_speak(s,3,"Hole for node %d %d found ...",i,j);

		px1=0.0; px1w=VERYLARGEFLOAT;
		px2=0.0; px2w=VERYLARGEFLOAT;
		py1=0.0; py1w=VERYLARGEFLOAT;
		py2=0.0; py2w=VERYLARGEFLOAT;

		for (ii=i; ii>=1; ii--) {
		    ibn=x_ijk2ib(ii,j,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			px1=p_zval_v[ibn];
			px1w=i-ii;
			ii=0; /* to quit the loop */
		    }
		}

		for (ii=i; ii<=mx; ii++) {
		    ibn=x_ijk2ib(ii,j,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			px2=p_zval_v[ibn];
			px2w=ii-i;
			ii=mx+1;
		    }
		}

		for (jj=j; jj>=1; jj--) {
		    ibn=x_ijk2ib(i,jj,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			py1=p_zval_v[ibn];
			py1w=j-jj;
			jj=0;
		    }
		}



		for (jj=j; jj<=my; jj++) {
		    ibn=x_ijk2ib(i,jj,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			py2=p_zval_v[ibn];
			py2w=jj-j;
			jj=my+1;
		    }
		}



		/* now I have potentially 4 values, with 4 weights
		   indicating distance in each direction
		   e.g.
		   px1 = 2400 px1w=3 weight=(1/3)
		   px2 = 3000 px2w=8 --> 1/8
		   py1 = 2600 py1w=1 --> 1/1
		   py2 = 2800 py2w=5 --> 1/5

		   px1 shall have 1/3 influence, px2 1/8, py1 1/1 py2 1/5

		   sum of scales are 1/3+1/8+1/1+1/5 = X = 1.65833

		   hence p1xw_actual = 0.333/1.65833=0.20098
		   hence p2xw_actual = 0.125/1.65833=0.07538
		   hence p1xw_actual = 1/1.65833=0.6030
		   hence p1xw_actual = 0.2/1.65833=0.1206
		   0.20098 + 0.07538+0.6030+0.1206=1.0 ...?

		*/

		px1w=1.0/px1w; px2w=1.0/px2w; py1w=1.0/py1w; py2w=1.0/py2w;
		sumw=px1w+px2w+py1w+py2w;
		px1w=px1w/sumw; px2w=px2w/sumw; py1w=py1w/sumw; py2w=py2w/sumw;

		sumw=px1w+px2w+py1w+py2w;
		if (sumw<0.98 || sumw> 1.02) {
		    xtg_error(s,"Wrong sum for weights. STOP");
		}

		/* assign value */
		p_zval_v[ib]=px1*px1w+px2*px2w+py1*py1w+py2*py2w;
	    }
	}
    }

    return 0;
}
