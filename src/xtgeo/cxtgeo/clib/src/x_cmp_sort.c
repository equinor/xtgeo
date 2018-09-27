/*
 * ############################################################################
 * x_cmp_sort.c
 * Routine needed by qsort (see e.g. map_median_filter) 
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: cmp_sort.c,v 1.1 2000/12/12 17:24:54 bg54276 Exp $ 
 * $Source: /h/bg54276/jcr/prg/lib/gplext/GPLExt/RCS/cmp_sort.c,v $ 
 *
 * $Log: cmp_sort.c,v $
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */

#include "libxtg_.h"

/*
 * ****************************************************************************
 *                                CMP_SORT 
 * ****************************************************************************
 * Sort ascending FLOAT values
 * ----------------------------------------------------------------------------
 *
 */   


int x_cmp_sort (
		const void *vp, 
		const void *vq
		)

{

    const float         *p=vp;
    const float         *q=vq;
    float               diff;

    diff = *p - *q;

    if (diff < 0) {
	return -1;
    }
    else if (diff == 0) {
	return 0;
    }
    else{
	return 1;
    }
}
