/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_cube_attr_intv.c
 *
 * DESCRIPTION:
 *    Provide averages for cubes between 2 surfaces. All possible attributes are
 *    computed in C to speed up calcuations. All is based on surfaces that
 *    matches cube geometry exact in 2D (ncol, nrow, xori, ...)
 *
 *    New from May 2020, replaces surf_slice_cube_window*
 *
 * ARGUMENTS:
 *    ncol, nrow...  i     cube dimensions and relevant increments
 *    czori, czinc   i     Cube zori and zinc
 *    cubevalsv      i     Cube array with lengths for swig
 *    surfsv*        i     Surface 1 2 array swith lengths
 *    maskv*         i     Mask 1 2 array swith lengths
 *    sliczinc       i     slice increments
 *    ndiv*          i     Number of divisions
 *    sresult       i/o    Results array, a stack of attribute surfaces
 *    optnearest     i     If 1 use nearest node, else do interpolation aka trilinear
 *    optmask        i     If 1 then masked cells
 *    optprogress    i     If show progress print
 *    masktreshold   i     Intervals thinner than maskthreshold will be masked (undef)
 *    optsum         i     If 1, at least one sum attribute is needed (optimise speed)
 *
 * RETURNS:
 *    Function: 0: upon success
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    See XTGeo lisence
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

#define NATTR 14

static void
pillar_attrs(double *pillar, int np, double *pattr)
{
    /* private routine that computes all possible attributes along on pillar/column*/

    int n;
    for (n = 0; n < NATTR; n++)
        pattr[n] = UNDEF;

    if (np < 1)
        return;

    int k;
    double val;

    double max = VERYLARGENEGATIVE;
    double min = VERYLARGEPOSITIVE;
    double sum = 0.0;
    double rms = 0.0;
    double maxpos = VERYLARGENEGATIVE;
    double maxneg = VERYLARGEPOSITIVE;
    double maxabs = VERYLARGENEGATIVE;
    double sumpos = 0.0;
    double sumneg = 0.0;
    double sumabs = 0.0;

    int nppos = 0;
    int npneg = 0;

    for (k = 0; k < np; k++) {

        val = pillar[k];

        if (val >= max)
            max = val;
        if (val < min)
            min = val;
        if (val >= 0.0 && val >= maxpos)
            maxpos = val;
        if (val < 0.0 && val < maxneg)
            maxneg = val;
        if (fabs(val) > maxabs)
            maxabs = fabs(val);

        sum = sum + val;
        rms = rms + val * val;

        if (val >= 0.0) {
            sumpos = sumpos + val;
            nppos++;
        }
        if (val < 0.0) {
            sumneg = sumneg + val;
            npneg++;
        }
        sumabs = sumabs + fabs(val);
    }

    double mean = sum / (double)np;

    double meanabs = sumabs / (double)np;

    if (maxpos <= VERYLARGENEGATIVE)
        maxpos = UNDEF;
    if (maxneg >= VERYLARGEPOSITIVE)
        maxneg = UNDEF;

    double meanpos = UNDEF;
    double meanneg = UNDEF;
    if (nppos > 0) {
        meanpos = sumpos / (double)nppos;
    }
    if (npneg > 0) {
        meanneg = sumneg / (double)npneg;
    }

    rms = sqrt(rms / (double)np);

    sum = 0.0;
    for (k = 0; k < np; k++) {
        val = pillar[k];
        sum = sum + pow(val - mean, 2.0);
    }

    double var = sum / (double)np;  // sample variance

    pattr[0] = min;
    pattr[1] = max;
    pattr[2] = mean;
    pattr[3] = var;
    pattr[4] = rms;
    pattr[5] = maxpos;
    pattr[6] = maxneg;
    pattr[7] = maxabs;
    pattr[8] = meanabs;
    pattr[9] = meanpos;
    pattr[10] = meanneg;
}

static void
pillar_attrs_sums(double *pillar, int np, double *pattr)
{
    /* SUM attributes are only menaingful on a discrete level*/

    int n;
    for (n = 11; n <= 13; n++)
        pattr[n] = UNDEF;

    if (np < 1)
        return;

    int k;

    double sumpos = 0.0;
    double sumneg = 0.0;
    double sumabs = 0.0;

    int nppos = 0;
    int npneg = 0;

    for (k = 0; k < np; k++) {

        double val = pillar[k];

        if (val >= 0.0) {
            sumpos = sumpos + val;
            nppos++;
        }
        if (val < 0.0) {
            sumneg = sumneg + val;
            npneg++;
        }
        sumabs = sumabs + fabs(val);
    }

    if (nppos <= 0) {
        sumpos = UNDEF;
    }

    if (npneg <= 0) {
        sumneg = UNDEF;
    }

    pattr[11] = sumpos;
    pattr[12] = sumneg;
    pattr[13] = sumabs;
}

static void
compute_attributes(double **stack, long nsurf, int ndiv, double *sres, int cflag)
{
    /* private */

    double *pattr;

    pattr = calloc(NATTR, sizeof(double));

    int i;
    for (i = 0; i < nsurf; i++) {

        // look along each pillar:
        int k;
        double *pillar = calloc(ndiv, sizeof(double));
        int np = 0;
        for (k = 0; k < ndiv; k++) {

            double val = stack[i][k];
            if (val < UNDEF_LIMIT) {
                pillar[np++] = val;
            }
        }
        int nat;
        if (cflag == 0) {
            pillar_attrs(pillar, np, pattr);
            for (nat = 0; nat <= 10; nat++) {
                long ic = i + nsurf * nat;
                sres[ic] = pattr[nat];
            }
        } else {
            pillar_attrs_sums(pillar, np, pattr);
            for (nat = 11; nat <= 13; nat++) {
                long ic = i + nsurf * nat;
                sres[ic] = pattr[nat];
            }
        }

        free(pillar);
    }
    free(pattr);
}

int
surf_cube_attr_intv(int ncol,
                    int nrow,
                    int nlay,
                    double czori,
                    double czinc,
                    float *cubevalsv,
                    long ncube,
                    double *surfsv1,
                    long nsurf1,
                    double *surfsv2,
                    long nsurf2,
                    mbool *maskv1,
                    long nmask1,
                    mbool *maskv2,
                    long nmask2,
                    double slicezinc,
                    int ndiv,
                    int ndivdisc,
                    double *sresult,
                    long nresult,
                    int optnearest,
                    int optmask,
                    int optprogress,
                    double maskthreshold,
                    int optsum)

{

    logger_info(LI, FI, FU, "Enter %s", FU);

    double **stack = x_allocate_2d_double(nsurf1, ndiv + 1);
    mbool **rmask = x_allocate_2d_mbool(nsurf1, ndiv + 1);

    if (optprogress)
        printf("progress: initialising for attributes...\n");

    logger_info(LI, FI, FU, "Initialise");

    int i, ic;
    for (i = 0; i < nsurf1; i++) {
        for (ic = 0; ic <= ndiv; ic++) {
            if (maskv1[i] == 0 && maskv2[i] == 0) {
                stack[i][ic] = surfsv1[i] + ic * slicezinc;
                rmask[i][ic] = 0;
            } else {
                stack[i][ic] = UNDEF;
                rmask[i][ic] = 1;
            }

            if (surfsv2[i] < (surfsv1[i] + maskthreshold)) {
                stack[i][ic] = UNDEF;
                rmask[i][ic] = 1;
            }
        }
    }

    logger_info(LI, FI, FU, "Surf slice...");

    // get surfs from cube and stack them
    surf_stack_slice_cube(ncol, nrow, nlay, ndiv + 1, czori, czinc, cubevalsv, stack,
                          rmask, optnearest, optmask);

    logger_info(LI, FI, FU, "Init... dode", FU);

    if (optprogress)
        printf("progress: compute mean, variance, etc attributes...\n");

    logger_info(LI, FI, FU, "Attributes...");
    compute_attributes(stack, nsurf1, ndiv + 1, sresult, 0);

    x_free_2d_double(stack);
    x_free_2d_mbool(rmask);

    logger_info(LI, FI, FU, "Done");

    if (optsum == 0)
        return EXIT_SUCCESS;  // don't compute sum attribute unless they are asked for

    /*
     * Special treatment of sum attributes, as they are not trivial to deduce
     * if cells are interpolated; hence use only a discrete scheme here:
     */

    double **dstack = x_allocate_2d_double(nsurf1, ndivdisc + 1);
    mbool **drmask = x_allocate_2d_mbool(nsurf1, ndivdisc + 1);

    if (optprogress)
        printf("progress: initialising for sum attributes...\n");

    for (i = 0; i < nsurf1; i++) {
        for (ic = 0; ic <= ndivdisc; ic++) {
            if (maskv1[i] == 0 && maskv2[i] == 0) {
                dstack[i][ic] = surfsv1[i] + ic * czinc;
                drmask[i][ic] = 0;
            } else {
                dstack[i][ic] = UNDEF;
                drmask[i][ic] = 1;
            }

            if (surfsv2[i] < (surfsv1[i] + maskthreshold)) {
                dstack[i][ic] = UNDEF;
                drmask[i][ic] = 1;
            }
        }
    }
    // get surfs from cube and stack them
    surf_stack_slice_cube(ncol, nrow, nlay, ndivdisc + 1, czori, czinc, cubevalsv,
                          dstack, drmask, optnearest, optmask);

    if (optprogress)
        printf("progress: compute sum attributes...\n");

    compute_attributes(dstack, nsurf1, ndivdisc + 1, sresult, 1);

    x_free_2d_double(dstack);
    x_free_2d_mbool(drmask);

    return EXIT_SUCCESS;
}
