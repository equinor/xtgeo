/*
 *******************************************************************************
 *
 * NAME:
 *    map_create.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Create an UNDEF map from scratch (no file import). Note that allocation
 *    shall be done in the calling routine.
 *
 * ARGUMENTS:
 *    mx, my         i     Map dimension
 *    xori,xstep     i     Maps X settings
 *    yori,ystep     i     Maps Y settings
 *    rotation       i     Map rotation (not in use!)
 *    p_zval_v       o     Maps depth/whatever values vector (pointer)
 *    value          i     Constant value of map (may also pass UNDEF)
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void + Changed pointer to map property (z values)
 *
 * TODO/ISSUES/BUGS:
 *    Todo, need test if xori etc are sane.
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */



#include "libxtg.h"
#include "libxtg_.h"

void map_create (
		 int    mx,
		 int    my,
		 double xori,
		 double xstep,
		 double yori,
		 double ystep,
		 double rotation,
		 double *p_zval_v,
		 double value,
		 int    option,
		 int    debug
		 )
{

    /* locals */
    int     ib;
    char    s[24]="map_create";

    xtgverbose(debug);
    xtg_speak(s,1,"Create map ...");


    for (ib=0; ib<mx*my; ib++) {
	p_zval_v[ib]=value;
	if (ib==0) {
	    printf("P_ZVAL_V: %f\n",p_zval_v[0]);
	}
    }

}
