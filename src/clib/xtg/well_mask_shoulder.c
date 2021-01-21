/*
****************************************************************************************
*
* NAME:
*    well_mask_shoulder.c
*
* DESCRIPTION:
*    Given a TVD or a MD log, and a discrete log, return a mask log
*    based on distance criteries
*
* ARGUMENTS:
*    lvec           i     MD or TVD log
*    nvec           i     Number of elements in lvec
*    inlog          i     Input log
*    ninlog         i     Number of points in inlog
*    distance       i     Distance to evaluate
*
* RETURNS:
*    Function:  0: Upon success. If problems:
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>

int
well_mask_shoulder(double *lvec,
                   long nvec,
                   int *inlog,
                   long ninlog,
                   int *mask,
                   long nmask,
                   double distance)
{

    if (nvec != nmask || nmask != ninlog)
        return EXIT_FAILURE;

    long i;
    for (i = 0; i < nmask; i++)
        mask[i] = 0;

    for (i = 1; i < ninlog; i++) {

        int v1 = inlog[i - 1];
        int v2 = inlog[i];

        if (v1 < UNDEF_INT_LIMIT && v2 < UNDEF_INT_LIMIT && (v1 != v2)) {

            double mid = 0.5 * (lvec[i - 1] + lvec[i]);

            // look in negative direction
            int ic = i - 1;
            while (ic >= 0) {
                double dist = fabs(mid - lvec[ic]);
                if (dist <= distance && inlog[ic] < UNDEF_INT_LIMIT) {
                    mask[ic] = 1;
                } else {
                    break;
                }
                ic--;
            }

            // look in positive direction
            ic = i;
            while (ic < nmask) {
                double dist = fabs(lvec[ic] - mid);
                if (dist <= distance && inlog[ic] < UNDEF_INT_LIMIT) {
                    mask[ic] = 1;
                } else {
                    break;
                }
                ic++;
            }
        }
    }
    return EXIT_SUCCESS;
}
