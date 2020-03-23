/*
 ***************************************************************************************
 *
 * NAME:
 *    pol_resample.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Take in an existing polyline, and return a resampled polyline. Probably
 *    best to avoid closed polygons?
 *    This can also be used for making fence from well paths!
 *
 *    From oct 10, 2018: It is important that sampling is horizontally uniform,
 *    hence smpl is recalculated as well as the extensions. Hence input will be
 *    estimates only.
 *
 * ARGUMENTS:
 *    nlen           i     lenght of input vector
 *    xv             i     X array
 *    yv             i     Y array
 *    zv             i     Z array
 *    smpl           i     Sample distance input
 *    next           i     Extension in both ends distance (as next*smpl) input
 *    nbuf           i     Allocated length of output vectors
 *    nolen          o     Actual length of output vectors
 *    xov            o     X array output
 *    yov            o     Y array output
 *    zov            o     Z array output
 *    hlen           o     Horizontal length vector, relative to FIRST input
 *                   o     point.
 *    option         i     Options...
 *    debug          i     Debug flag
 *
 * RETURNS:
 *    Function:  0: Upon success. If problems:
 *              -8: NBUF is too small (not enough length allocated)
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <stdlib.h>
#include <string.h>

int
pol_resample(int nlen,
             double *xv,
             double *yv,
             double *zv,
             double smpl,
             int next,
             int nbuf,
             int *nolen,
             double *xov,
             double *yov,
             double *zov,
             double *hlen,
             int option,
             int debug)
{
    // int i, n, ier, nsam, nnext;
    // double x0, y0, z0, x1 = 0.0, y1 = 0.0, z1, x2, y2, z2, xr, yr, zr;
    // double length1, length2, delta, dscaler;
    // double *tmp_tlen, *tmp_dtlen;
    // double *tmp_hlen, *tmp_dhlen, xsmpl, xhlen, xect;
    // double x00, x01, x10, x11, y00, y01, y10, y11;
    // double *tmptt, *tmpdt, *tmpdh;

    double *tmp_tlen = calloc(nlen, sizeof(double));
    double *tmp_dtlen = calloc(nlen, sizeof(double));
    double *tmp_hlen = calloc(nlen, sizeof(double));
    double *tmp_dhlen = calloc(nlen, sizeof(double));

    /* find the tmp_hlen vector, which is the horizontal cumulative length */
    pol_geometrics(xv, nlen, yv, nlen, zv, nlen, tmp_tlen, nlen, tmp_dtlen, nlen,
                   tmp_hlen, nlen, tmp_dhlen, nlen);

    double xhlen = tmp_hlen[nlen - 1]; /* total horizontal length */
    if (xhlen < 1.0)
        xhlen = 1.0;

    /* the extension vector input is based on the the first points within a
     * sample unit, or if the total hlen is less than the 0.5*sample, then the
     * end points will be used. Point x0* are START, points x1* etc are END.
     * i.e. x00 is first point and x01 are second X points for START extension
     */

    double x01 = xv[0]; /* second point on START extension */
    double y01 = yv[0];

    double x11 = xv[nlen - 1]; /* second point on END extension */
    double y11 = yv[nlen - 1];

    double x00 = -999;
    double y00 = -999;
    double x10 = -999;
    double y10 = -999;

    /* recompute sampling and extension so it becomes uniform */
    int nsam = 1 + (int)(xhlen / smpl);
    if (nsam < 4)
        nsam = 4;
    double xsmpl = xhlen / nsam;

    double xect = next * smpl;
    int nnext = xect / xsmpl;

    if (xhlen < xsmpl) { /* never occur? */
        x00 = xv[nlen - 1];
        y00 = yv[nlen - 1];
        x10 = xv[0];
        y10 = yv[0];
    } else {
        int i;
        for (i = 0; i < nlen; i++) {
            if (tmp_hlen[i] > xsmpl) {
                x00 = xv[i];
                y00 = yv[i];
                break;
            }
        }

        for (i = nlen - 1; i >= 0; i--) {
            if ((xhlen - tmp_hlen[i]) > xsmpl) {
                x10 = xv[i];
                y10 = yv[i];
                break;
            }
        }
    }
    /* Create the first extension first: */
    double z0 = zv[0];
    double z1 = z0;

    int n = 0;
    if (nnext > 0) {
        int i;
        for (i = nnext; i > 0; i--) {

            double xr, yr, zr;

            int ier =
              x_vector_linint2(x00, y00, z0, x01, y01, z1, xsmpl * i, &xr, &yr, &zr, 1);

            if (ier != 0) {
                logger_error(LI, FI, FU, "Something went wrong in %s IER = %d", FU,
                             ier);
            }

            xov[n] = xr;
            yov[n] = yr;
            zov[n] = z0; /* keep constant z */
            n++;
            if (n >= nbuf)
                return -8;
        }
    } else {
        xov[n] = x01;
        yov[n] = y01;
        zov[n] = z0;
        n++;
    }

    /* now resample along the given input polyline, but beware if the total
     * length is less than sample distance*/

    if (xhlen < xsmpl) {
        xsmpl = xhlen / 4;
    }

    double x0 = xv[0];
    double y0 = yv[0];
    double z0 = zv[0];
    xov[n] = x0;
    yov[n] = y0;
    zov[n] = z0;
    n++;

    int i;
    for (i = 1; i < nlen; i++) {
        double x1 = xv[i - 1];
        double y1 = yv[i - 1];
        double z1 = zv[i - 1];

        double x2 = xv[i];
        double y2 = yv[i];
        double z2 = zv[i];

        double length1 = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2)); /* only XY length */
        double length2 = sqrt(pow(x2 - x0, 2) + pow(y2 - y0, 2)); /* only XY length */

        if (length1 <= xsmpl && length2 > xsmpl) {

            double dscaler;
            if ((length2 - length1) < 1.E-20) {
                dscaler = 1.0;
            } else {
                dscaler = (xsmpl - length1) / (length2 - length1);
            }
            double xr, yr, zr;
            int ier = x_vector_linint(x1, y1, z1, x2, y2, z2, dscaler, &xr, &yr, &zr);

            if (ier < 0) {
                logger_error(LI, FI, FU, "Something went wrong in %s IER = %d", FU,
                             ier);
            }

            xov[n] = xr;
            yov[n] = yr;
            zov[n] = zr;

            n++;
            if (n >= nbuf)
                return -8;

            x0 = xr;
            y0 = yr;
            z0 = zr;
        }
    }

    /* extension in the tail part, same principle as first extension,
     * but now x0 is the last sampled point
     */

    if (nnext > 0) {
        z0 = z1;

        /* xov[n] = x1; */
        /* yov[n] = y1; */
        /* zov[n] = z1; */
        /* n++; */

        int i;

        for (i = 0; i <= nnext; i++) {

            double xr, yr, zr;

            int ier = x_vector_linint2(x10, y10, z0, x11, y11, z1,
                                       xsmpl * i,  // xsmpl not smpl?
                                       &xr, &yr, &zr, 2);

            if (ier != 0) {
                logger_error(LI, FI, FU, "Something went wrong in %s IER = %d", FU,
                             ier);
            }

            xov[n] = xr;
            yov[n] = yr;
            zov[n] = zr;
            n++;

            if (n >= nbuf)
                return -8;
        }

        if (n >= nbuf) {
            logger_error(LI, FI, FU, "Something went wrong in %s: NBUF = %d too small",
                         FU, nbuf);
        }
    }

    if (debug > 1) {
        for (i = 0; i < n; i++) {
            double delta;
            if (i > 0)
                delta =
                  (sqrt(pow(xov[i] - xov[i - 1], 2) + (pow(yov[i] - yov[i - 1], 2))));
        }
    }

    int *nolen;
    *nolen = n;

    /* find the hlen vector */
    double *tmptt = calloc(*nolen, sizeof(double));
    double *tmpdt = calloc(*nolen, sizeof(double));
    double *tmpdh = calloc(*nolen, sizeof(double));

    int ier = pol_geometrics(xov, *nolen, yov, *nolen, zov, *nolen, tmptt, *nolen,
                             tmpdt, *nolen, hlen, *nolen, tmpdh, *nolen);

    if (debug > 1) {
        for (i = 0; i < n; i++) {
            double delta = 0.0;
            if (i > 0) {
                delta = hlen[i] - hlen[i - 1];
            }
        }
    }

    if (ier != 0) {
        logger_error(LI, FI, FU, "Something went wrong in gemetrics for %s IER = %d",
                     FU, ier);
    }

    free(tmptt);
    free(tmpdt);
    free(tmpdh);

    free(tmp_tlen);
    free(tmp_dtlen);
    free(tmp_hlen);
    free(tmp_dhlen);

    return (0);
}
