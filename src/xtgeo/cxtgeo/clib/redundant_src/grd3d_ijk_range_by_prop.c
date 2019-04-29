/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_ijk_range_by_prop.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Get the I, J, K range from the a discrete property range (zone, facies).
 *    All ranges are inclusive.
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K
 *    p1, p2         i     Property range (interval), e.g. zone 1 - 8
 *    p_iprop_v      i     Grid property (discr) to use
 *    p_actnum_v     o     Grid ACTNUM parameter
 *    imin ... kmax  o     min and max index for I J K
 *    iflag          i     Options flag, 0=use actnum, 1= ignore ACTNUM
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_ijk_range_by_prop (
                             int    nx,
                             int    ny,
                             int    nz,
                             int    p1,
                             int    p2,
                             int    *p_iprop_v,
                             int    *p_actnum_v,
                             int    *imin,
                             int    *imax,
                             int    *jmin,
                             int    *jmax,
                             int    *kmin,
                             int    *kmax,
                             int    iflag,
                             int    debug
                             )
{
    /* locals */
    char s[24]="grd3d_ijk_range_by_prop";
    int  i, j, k, ib;
    int  imi, ima, jmi, jma, kmi, kma, qtest;

    xtgverbose(debug);

    imi = nx + 1;
    ima = 0;
    jmi = ny + 1;
    jma = 0;
    kmi = nz + 1;
    kma = 0;

    xtg_speak(s,2,"Entering routine <%s>",s);

    xtg_speak(s,2,"Property interval is %d - %d", p1, p2);

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);

                qtest = 1;
                if (p_actnum_v[ib] == 0 && iflag == 0) qtest=0;

                if (qtest == 1) {
                    if (p_iprop_v[ib] >= p1 && p_iprop_v[ib] <= p2) {
                        if (i < imi) imi = i;
                        if (i > ima) ima = i;
                        if (j < jmi) jmi = j;
                        if (j > jma) jma = j;
                        if (k < kmi) kmi = k;
                        if (k > kma) kma = k;
                    }
                }
	    }
	}
    }

    /* check that this looks sane */
    qtest = 0;
    if ((imi <= ima && jmi <= jma && kmi < kma) &&
        (imi>=1 && ima<=nx && jmi>=1 && jma<=ny && kmi>=1 && kma<=nz)) {
        qtest = 1;
    }
    else{
        xtg_warn(s, 1, "Could not make a range");
        *imin = 0;
        *imax = 0;
        *jmin = 0;
        *jmax = 0;
        *kmin = 0;
        *kmax = 0;
        return(-1);
    }

    if (qtest == 1) {
        *imin = imi;
        *imax = ima;
        *jmin = jmi;
        *jmax = jma;
        *kmin = kmi;
        *kmax = kma;
    }

    xtg_speak(s,2,"Exit from <%s>",s);

    return EXIT_SUCCESS;

}
