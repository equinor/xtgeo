/*
 * ############################################################################
 * x_ib2ijk.c
 * Converting from I,J,K to sequential block number IB
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: ib2ijk.c,v 1.1 2000/12/12 17:24:54 bg54276 Exp $
 * $Source: /h/bg54276/jcr/prg/lib/gplext/GPLExt/RCS/ib2ijk.c,v $
 *
 * $Log: ib2ijk.c,v $
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */

#include "libxtg_.h"
#include "libxtg.h"

/*
 * ****************************************************************************
 *                              IB2IJK
 * ****************************************************************************
 * Change for sequence counting to I,J,K, number. The I,J,K is always
 * base 1 offset. The IB may be base 0 (most common) or base 1, and that
 * is given by the value of ia_start.
 *
 * ----------------------------------------------------------------------------
 *
 */


void x_ib2ijk (
	       long  ib,
	       int   *i,
	       int   *j,
	       int   *k,
	       int   nx,
	       int   ny,
	       int   nz,
	       int   ia_start
	       )
{
    long ir, nxy;
    long ix=1,iy=1,iz=1;

    nxy = nx*ny;


    if (ia_start==0) ib=ib+1;  /* offset number to counter number */

    iz = ib/nxy;
    if (iz*nxy < ib) iz = iz + 1;
    ir = ib-((iz-1)*nxy);
    iy=ir/nx;
    if (iy*nx < ir) iy=iy+1;

    ix=ir-((iy-1)*nx);

    /* values to return */
    *i=ix; *j=iy; *k=iz;
}


/* C order: */
void x_ic2ijk (
	       long  ic,
	       int   *i,
	       int   *j,
	       int   *k,
	       int   nx,
	       int   ny,
	       int   nz,
	       int   ia_start
	       )
{
    long ir, nxy;
    long ix=1,iy=1,iz=1;

    nxy = nx*ny;


    if (ia_start==0) ic=ic+1;  /* offset number to counter number */

    iz = ic/nxy;
    if (iz*nxy < ic) iz = iz + 1;
    ir = ic-((iz-1)*nxy);
    ix=ir/ny;
    if (ix*ny < ir) ix=ix+1;

    iy=ir-((ix-1)*ny);

    /* values to return */
    *i=ix; *j=iy; *k=iz;
}
