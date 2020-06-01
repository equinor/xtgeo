/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_linint1d.c
 *
 * DESCRIPTION:
 *    Given an array with 1d distance vector and a 1d value vector. For a distance
 *    zd, return the snapped or interpolated value.
 *
 *    It is assumed that dist array is sorted in increasing order
 *
 * ARGUMENTS:
 *    dval             i     Distance value to be evaluated
 *    dist             i     Distance array
 *    vals             i     Values array
 *    nval             i     Number of entries
 *    option           i     1: snap to nearest, 0 interpolate
 *
 * RETURNS:
 *    Updated value, UNDEF if not possible
 *
 * LICENCE:
 *    See XTGeo lisence
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

double
x_vector_linint1d(double dval, double *dist, double *vals, int nval, int option)
{

    if (nval < 2)
        return UNDEF;

    int i;
    for (i = 0; i < nval - 1; i++) {

        if (dist[i] == dist[i + 1])
            return vals[i];

        if (dval >= dist[i] && dval < dist[i + 1]) {

            if (option == 1) {
                double zp = vals[i];

                double d1 = fabs(dval - dist[i]);
                double d2 = fabs(dval - dist[i + 1]);

                if (d2 < d1)
                    zp = vals[i + 1];

                return zp;
            }

            double zp = vals[i] + ((vals[i + 1] - vals[i]) / (dist[i + 1] - dist[i])) *
                                    (dval - dist[i]);
            return zp;
        }
    }

    return UNDEF;
}
