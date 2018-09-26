/*
 * ############################################################################
 * x_ijk2ib.c
 * Converting from sequential block number to I,J,K cell
 * Author: J.C. Rivenaes (based on Tor Barkve's numlib)
 */

#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              IJK2IB
 * ****************************************************************************
 * From I,J,K, number to sequence IB. The I,J,K is always
 * base 1 offset. The IB may be base 0 (most common) or base 1, and that
 * is given by the value of ia_start.
 *
 * ----------------------------------------------------------------------------
 *
 */
long x_ijk2ib (
               int i,
               int j,
               int k,
               int nx,
               int ny,
               int nz,
               int ia_start
               )
{
    long nxy, ib;

    nxy  = nx*ny;


    /*some error checking*/
    if ( i>nx || j>ny || k>nz ){
	return -2;
    }
    else if( i<1 || j<1 || k<1 ) {
	return -2;
    }

    ib = (k-1)*nxy;
    ib = ib + (j - 1)*nx;
    ib = ib + i;

    /* if the C array ib is 0 index...*/
    if (ia_start == 0) ib--;


    return ib;
}

/* c order counting, where K is looping fastest, them J, then I */
long x_ijk2ic(
              int i,
              int j,
              int k,
              int nx,
              int ny,
              int nz,
              int ia_start
              )
{
    long nzy, ic;

    nzy  = nz*ny;


    /*some error checking*/
    if ( i>nx || j>ny || k>nz ){
	return -2;
    }
    else if( i<1 || j<1 || k<1 ) {
	return -2;
    }

    ic = (i-1)*nzy;
    ic = ic + (j - 1)*nz;
    ic = ic + k;

    /* if the C array ic is 0 index...*/
    if (ia_start == 0) ic--;


    return ic;
}
