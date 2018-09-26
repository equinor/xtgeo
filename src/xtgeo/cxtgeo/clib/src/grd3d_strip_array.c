/*
 * ############################################################################
 * grd3d_strip_array_xxxx 
 * Basic routines to strip single properties from a long string
 * In principle:
 * nxyz          Number of cells (total nx*ny*nz)
 * counter       internal numbering of the actual prop within v1 (0 for first)
 * v1            the input array (n multiples for nxyz)
 * v2            the output array (nxyz)
 * debug         debug flag
 *
 * Author:       JRIV
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

void grd3d_strip_afloat (
			 int     nxyz,
			 int     counter,
			 double  *v1,
			 float   *v2,
			 int     debug
			 )
{
    int     i, nshift;
    char    s[24]="grd3d_strip_afloat";

    xtgverbose(debug);

    xtg_speak(s,2,"Stripping FLOAT array via %s",s);

    nshift = counter*nxyz;

    for (i=0; i<nxyz; i++) {
	v2[i]=(float)v1[i+nshift];
    }

    xtg_speak(s,2,"Stripping array via %s ... DONE",s);
}

void grd3d_strip_anint (
			 int     nxyz,
			 int     counter,
			 double  *v1,
			 int     *v2,
			 int     debug
			 )
{
    int     i, nshift;
    char    s[24]="grd3d_strip_anint";

    xtgverbose(debug);

    xtg_speak(s,2,"Stripping INT array via %s",s);

    nshift = counter*nxyz;

    for (i=0; i<nxyz; i++) {
	v2[i]=(int)v1[i+nshift];
    }

    xtg_speak(s,2,"Stripping INT array via %s ... DONE",s);
}

void grd3d_strip_adouble (
			  int     nxyz,
			  int     counter,
			  double  *v1,
			  double  *v2,
			  int     debug
			 )
{
    int     i, nshift;
    char    s[24]="grd3d_strip_adouble";

    xtgverbose(debug);

    xtg_speak(s,2,"Stripping DOUBLE array via %s",s);

    nshift = counter*nxyz;

    for (i=0; i<nxyz; i++) {
	v2[i]=v1[i+nshift];
    }

    xtg_speak(s,2,"Stripping DOUBLE array via %s ... DONE",s);
}


