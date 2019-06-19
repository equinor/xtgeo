/*
 ******************************************************************************
 *
 * SINFO: Resample (and extend) a polygon line (e.g. a well fence)
 *
 ******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    pol_resampling.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Note: This is an recoded and better version of pol_resample.c
 *
 *    Take an input polygon, extend it at ends, and then do a
 *    uniform resampling in XY hence smpl is recalculated as well as the
 *    extensions. Hence input sampling will be estimates only.
 *
 *    The extension is estimated from the input, by extrapolating the
 *    last fraction of the sampling distance.
 *
 * ARGUMENTS:
 *    xv             i     X array
 *    nlenx          i     lenght of input vector...
 *    yv             i     Y array
 *    nleny          i     lenght of input vector...
 *    zv             i     Z array
 *    nlenz          i     lenght of input vector...
 *    smpl           i     Sample distance input as proposal (to be modified!)
 *    hext           i     Extension in both ends distance as length!
 *    xov            o     X array output
 *    nbufx          i     Allocated length of output vector...
 *    yov            o     Y array output
 *    nbufy          i     Allocated length of output vector...
 *    zov            o     Z array output
 *    nbufz          i     Allocated length of output vector...
 *    hlen           o     Horizontal length vector, relative to FIRST input
 *                   o     point.
 *    nbufh          i     Allocated length of output vector hlen...
 *    nolen          o     Actual! length of output vectors
 *    option         i     if option 1, then Z output will be contant 0
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
 ******************************************************************************
 */

#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

#define MBUFFER 10000

int pol_resampling(
                   double *xv,
                   long nlenx,
                   double *yv,
                   long nleny,
                   double *zv,
                   long nlenz,
                   double smpl,
                   double hext,
                   double *xov,
                   long nbufx,
                   double *yov,
                   long nbufy,
                   double *zov,
                   long nbufz,
                   double *hlen,
                   long nbufh,
                   int *nolen,
                   int option,
                   int debug)
{
    double *txv, *tyv, *tzv, *thlen, *tdhlenx, *ttlenx, *tdtlenx, *thlenx, *tmpdh;
    double *tmptt, *tmpdt;
    double tchlen, usmpl, angr_start, angr_end, angd, vlen, uhext;
    double length1, length2, dscaler, x1, x2, y1, y2, z1, z2, xr, yr, zr;
    double delta, x0, y0, z0, x3, y3, xs, ys, zs, xe, ye, ze, hxxd, dist;
    int ier, nsam, ic, nct = 0, i, naddr;

    char  sbn[24] = "pol_resampling";

    xtgverbose(debug);
    xtg_speak(sbn, 2, "Running %s", sbn);

    thlen = calloc(nlenx, sizeof(double));

    /* estimate additonal alloc for refinement, plus a large buffer */
    naddr = nlenx + MBUFFER; //nlenx + 2 + 2* (x_nint(hext / smpl) + 1) + MBUFFER;

    txv = calloc(naddr, sizeof(double));
    tyv = calloc(naddr, sizeof(double));
    tzv = calloc(naddr, sizeof(double));
    ttlenx = calloc(naddr, sizeof(double));
    tdtlenx = calloc(naddr, sizeof(double));
    thlenx = calloc(naddr, sizeof(double));
    tdhlenx = calloc(naddr, sizeof(double));


    /* ========================================================================
     * Part one, look at current input, re-estimate sampling distance, and
     * find extend angles using certain criteria
     * ========================================================================
     */

    /* find the hlen vector, which is the horizontal cumulative length */
    ier = pol_geometrics(xv, nlenx, yv, nlenx, zv, nlenx, ttlenx, nlenx,
                         tdtlenx, nlenx, thlen, nlenx, tdhlenx, nlenx, debug);

    if (debug > 2) {
        for (i = 0;  i < nlenx; i++) {
            delta = 0.0;
            if (i>0) delta = thlen[i] - thlen[i - 1];
            xtg_speak(sbn, 3, "Input I X Y Z H - DH: %d %f %f %f %f - %f",
                      i, xv[i], yv[i], zv[i], thlen[i], delta);
        }
    }

    tchlen = thlen[nlenx - 1];  /* total cumulative horizontal length */
    if (tchlen < 1.0) tchlen = 1.0;

    /* recompute sampling and extension so it becomes uniform */
    nsam = x_nint(tchlen / smpl);
    if (nsam < 2) nsam = 2;
    usmpl = tchlen / nsam;  /* updated sampling distance*/

    if (debug > 2) xtg_speak(sbn, 3, "Sampling orig vs upd: %f %f", smpl,
                              usmpl);
    /* look at start/end vectors and derive angles (exception if ~vertical)
     * but need some samples
     */

    if (tchlen > 1.0) {
        hxxd = usmpl / 2.0;

        x0 = 0; x1 = 0; x2 = 0; x3 = 0;
        y0 = 0; y1 = 0; y2 = 0; y3 = 0;
        x0 = xv[0]; y0 = yv[0];
        for (i = 0;  i < nlenx; i++) {
            if (thlen[i] > hxxd) {
                x1 = xv[i]; y1 = yv[i];
                break;
            }
        }

        /* the other end */
        x2 = xv[nlenx - 1]; y2 = yv[nlenx -1];
        for (i = (nlenx - 1);  i >= 0; i--) {
            dist = thlen[nlenx - 1] - thlen[i];
            if (dist > hxxd) {
                x3 = xv[i]; y3 = yv[i];
                break;
            }
        }

        if (x1 == 0 || x3 == 0) {
            x1 = x2; y1 = y2;
            x3 = x0; y3 = y0;
        }
        x_vector_info2(x1, x0, y1, y0, &vlen, &angr_start,
                       &angd, 1, debug);
        x_vector_info2(x3, x2, y3, y2, &vlen, &angr_end,
                   &angd, 1, debug);
    }
    else{
        angr_start = PI; angr_end = 0.0;  /* west - east slice */
    }

    if (debug > 2) xtg_speak(sbn, 3, "Angle 1 and 2 (rad): %f %f", angr_start,
                              angr_end);

    /* compute re-estimated extension lengths */

    nsam = x_nint(hext / usmpl);
    uhext = nsam * usmpl;

    if (debug > 2) xtg_speak(sbn, 3, "Orig vs updated ext %f %f", hext,
                              uhext);

    /* now make a new points at start and end; extrapolate using angles */

    x_vector_extrapol2(xv[0], yv[0], zv[0], &xs, &ys, &zs, uhext,
                       angr_start, debug);
    x_vector_extrapol2(xv[nlenx - 1], yv[nlenx - 1], zv[nlenx - 1], &xe, &ye,
                       &ze, uhext, angr_end, debug);

    /* and finally make a merged trajectory */
    for (ic = 0; ic < nlenx; ic++) {
        if (ic == 0) {
            txv[0] = xs; tyv[0] = ys; tzv[0] = zv[ic];
        }
        txv[ic + 1] = xv[ic]; tyv[ic + 1] = yv[ic]; tzv[ic + 1] = zv[ic];
        if (ic == (nlenx - 1)) {
            txv[nlenx + 1] = xe; tyv[nlenx + 1] = ye;
            tzv[nlenx + 1] = zv[ic];
        }
    }

    if (debug > 2) {
        /* debugging only work */
        ier = pol_geometrics(txv, nlenx + 2,
                             tyv, nlenx + 2,
                             tzv, nlenx + 2,
                             ttlenx, nlenx + 2,
                             tdtlenx, nlenx + 2,
                             thlenx, nlenx + 2,
                             tdhlenx, nlenx + 2,
                             debug);
        for (i = 0;  i < (nlenx + 2); i++) {
            delta = 0.0;
            if (i>0) delta = thlenx[i] - thlenx[i - 1];
            xtg_speak(sbn, 3, "Extended I X Y Z H - DH: %d %f %f %f %f - %f",
                      i, txv[i], tyv[i], tzv[i], thlenx[i], delta);
        }
    }

    /* ========================================================================
     * Part two, refine the vector, otherwise resampling may fail.
     * ========================================================================
     */
    int nnnf = 0;
    nnnf = pol_refine(nlenx + 2, naddr, txv, tyv, tzv, usmpl/2.0,
                      1, debug);

    /* ========================================================================
     * Part three, resample along the t*v vectors
     * ========================================================================
     */
    nct = 0;
    x0 = txv[0]; y0 = tyv[0]; z0 = tzv[0];
    if (option == 1) z0 = 0.0;
    xov[nct] = x0; yov[nct] = y0; zov[nct] = z0;
    nct++;
    for (i = 1; i < nnnf; i++) {
        x1 = txv[i-1];
        y1 = tyv[i-1];
        z1 = tzv[i-1];
        if (option == 1) z1 = 0.0;

        x2 = txv[i];
        y2 = tyv[i];
        z2 = tzv[i];
        if (option == 1) z2 = 0.0;

        xtg_speak(sbn, 3, "XX x0 x1 x2 y0 y1 y2 %f %f %f   %f %f %f",
                  x0, x1, x2, y0, y1, y2);

        length1 = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2));  /* only XY len */
        length2 = sqrt(pow(x2 - x0, 2) + pow(y2 - y0, 2));  /* only XY len */

        if (length1 <= usmpl && length2 > usmpl) {

            if ((length2 - length1) < 1.E-20) {
                dscaler = 1;
            }
            else{
                dscaler = (usmpl - length1) / (length2 - length1);
            }
            ier = x_vector_linint(x1, y1, z1, x2, y2, z2, dscaler,
                                   &xr, &yr, &zr, debug);

            if (ier < 0) {
                xtg_error(sbn, "Something went wrong in %s IER=%d", sbn, ier);
            }

            delta = sqrt(pow(xr - x0, 2) + pow(yr - y0, 2));

            xov[nct] = xr;
            yov[nct] = yr;
            if (option == 1) zr = 0.0;
            zov[nct] = zr;

            nct++;

            if (nct >= nbufx) return -8;

            /* redefine start point */
            x0 = xr; y0 = yr; z0 = zr;

        }
    }

    *nolen = nct;

    xtg_speak(sbn, 2, "Updated NOLEN is %d", *nolen);

    tmptt = calloc(nbufh, sizeof(double));
    tmpdt = calloc(nbufh, sizeof(double));
    tmpdh = calloc(nbufh, sizeof(double));

    /* find the new hlen vector */
    ier = pol_geometrics(xov, *nolen,
                         yov, *nolen,
                         zov, *nolen,
                         tmptt, *nolen,
                         tmpdt, *nolen,
                         hlen, *nolen,
                         tmpdh, *nolen,
                         debug);

    for (i = 0;  i < nct; i++) hlen[i] -= uhext;

    if (debug > 2) {
        /* debugging only work */
        for (i = 0;  i < nct; i++) {
            delta = 0.0;
            if (i>0) delta = hlen[i] - hlen[i - 1];
            xtg_speak(sbn, 3, "Final I X Y Z H - DH: %d %f %f %f %f - %f",
                      i, xov[i], yov[i], zov[i], hlen[i], delta);
        }
    }

    if (ier != 0) {
        xtg_error(sbn, "Something went wrong in gemetrics for %s IER = %d",
                  sbn, ier);
    }

    free(thlen);
    free(tmptt);
    free(tmpdt);
    free(tmpdh);
    free(thlenx);
    free(txv);
    free(tyv);
    free(tzv);

    return EXIT_SUCCESS;
}
