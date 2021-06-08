/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_convert_hybrid.c
 *
 * DESCRIPTION:
 *    Convert a 3D grid to a hybrid grid; i.e. horisontal layers between two levels
 *
 *    There are two algorithns here for historical reasons, where no 2 can work
 *    inside a region.
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     Dimensions
 *    p_*_v          i     Geometry arrays (w numpy dimensions)
 *    nzhyb          i     Number of NZ for hybrid
 *    p_*hyb_v       o     New geometry arrays
 *    toplevel       i     Where to start hybrid (depth)
 *    botlevel       i     Where to stop hybrid (depth)
 *    ndiv           i     Divisison in hybridpart
 *    p_region_v     i     region array w/ numpy dimensions
 *    region         i     Actual region to update
 *
 * RETURNS:
 *    Void, update array pointer
 *
 * TODO/ISSUES/BUGS:
 *    - Code rewriting needed at some point, collect all in one common routine
 *    - Rewrite to xtgformat=2
 *    - Allow different settings for different regions
 *
 * LICENCE:
 *    LGPL v3. Copyright Equinor ASA.
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

/* local */
static void
_grd3d_convert_hybrid1(int nx,
                       int ny,
                       int nz,
                       double *coordsv,
                       double *zcornsv,
                       int *actnumsv,
                       int nzhyb,
                       double *p_zcornhyb_v,
                       int *p_actnumhyb_v,
                       double toplevel,
                       double botlevel,
                       int ndiv)

{
    /* locals */
    int i, j, k, n, ic, ibp, ibh, inp = 0, inh = 0, khyb, iflagt, iflagb;
    double z1, dz, zsum, ztop, zbot, zhyb, zsumh;

    /* thickness in horizontal section */
    dz = (botlevel - toplevel) / ndiv;

    for (j = 1; j <= ny; j++) {

        for (i = 1; i <= nx; i++) {
            iflagt = 1;
            iflagb = 1;
            ztop = UNDEF;
            zbot = -1 * UNDEF;
            /* first do the top-down truncation, collecting all at toplevel */
            for (k = 1; k <= nz + 1; k++) {
                ibp = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                if (ibp < 0 || ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside grid "
                                    "in _grd3d_convert_hybrid1");
                    return;
                }
                /* do for all corners */
                zsum = 0.0;
                for (ic = 1; ic <= 4; ic++) {
                    z1 = zcornsv[4 * ibp + 1 * ic - 1];
                    if (z1 > toplevel) {
                        p_zcornhyb_v[4 * ibh + 1 * ic - 1] = toplevel;
                    } else {
                        p_zcornhyb_v[4 * ibh + 1 * ic - 1] = z1;
                    }
                    /* store avg top depth; will be used to truncate later if active*/
                    zsum = zsum + z1;
                }
                /* now store top depth in input grid */
                if (k <= nz) {
                    if (actnumsv[ibp] == 1 && iflagt == 1) {
                        ztop = zsum / 4.0;
                        iflagt = 0;
                    }
                    p_actnumhyb_v[ibh] = actnumsv[ibp];
                }
            }

            /* now doing it the other way (from bottom) */
            khyb = nzhyb + 1;
            for (k = nz + 1; k >= 1; k--) {
                ibp = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                ibh = x_ijk2ib(i, j, khyb, nx, ny, nzhyb + 1, 0);
                if (ibp < 0 || ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in _grd3d_convert_hybrid1");
                    return;
                }
                /* in terms of active cells index, layer k _bottom_ shall refer to cell
                 * k-1 */
                if (k > 1) {
                    inp = x_ijk2ib(i, j, k - 1, nx, ny, nz + 1, 0);
                    inh = x_ijk2ib(i, j, khyb - 1, nx, ny, nzhyb + 1, 0);
                    if (inp < 0 || inh < 0) {
                        throw_exception("Loop through grid resulted in index outside "
                                        "boundary in grd3d_convert_hybrid");
                        return;
                    }
                }

                /* do for all corners */
                zsum = 0.0;
                for (ic = 1; ic <= 4; ic++) {
                    z1 = zcornsv[4 * ibp + 1 * ic - 1];
                    if (z1 < botlevel) {
                        p_zcornhyb_v[4 * ibh + 1 * ic - 1] = botlevel;
                    } else {
                        p_zcornhyb_v[4 * ibh + 1 * ic - 1] = z1;
                    }
                    /* store avg bot depth; will be used to truncate later if active*/
                    zsum = zsum + z1;
                }
                /* now bot depth from input grid */
                if (k > 1) {
                    if (actnumsv[inp] == 1 && iflagb == 1) {
                        zbot = zsum / 4.0;
                        iflagb = 0;
                    }
                    p_actnumhyb_v[inh] = actnumsv[inp];
                }
                khyb--;
            }

            /* now filling the intermediate */
            n = 0;
            for (k = nz + 1; k <= nz + 1 + ndiv - 1; k++) {
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                if (ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid");
                    return;
                }
                /* do for all corners */
                if (k > nz + 1) {
                    n++;
                    for (ic = 1; ic <= 4; ic++) {
                        p_zcornhyb_v[4 * ibh + 1 * ic - 1] = toplevel + n * dz;
                    }
                }
                p_actnumhyb_v[ibh] = 1;
            }

            /* truncate - ensure same volume, first from top, eval by cell centre */
            for (k = 1; k <= nzhyb; k++) {
                zsumh = 0.0;
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                inh = x_ijk2ib(i, j, k + 1, nx, ny, nzhyb + 1, 0);
                if (ibh < 0 || inh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid");
                    return;
                }
                /* do for all corners */
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * ibh + 1 * ic - 1];
                }
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * inh + 1 * ic - 1];
                }
                zhyb = 0.125 * zsumh; /* cell center */

                if (p_actnumhyb_v[ibh] == 1 && zhyb < ztop) {
                    p_actnumhyb_v[ibh] = 0;
                }
            }

            /* truncate - ensure same volume, now from bot, eval by cell centre */
            for (k = nzhyb + 1; k > 1; k--) {
                zsumh = 0.0;
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                inh = x_ijk2ib(i, j, k - 1, nx, ny, nzhyb + 1, 0);
                if (ibh < 0 || inh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid");
                    return;
                }
                /* do for all corners */
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * ibh + 1 * ic - 1];
                }
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * inh + 1 * ic - 1];
                }
                zhyb = 0.125 * zsumh;

                if (p_actnumhyb_v[inh] == 1 && zhyb > zbot) {
                    p_actnumhyb_v[inh] = 0;
                }
            }
        }
    }
}

/* local */
static void
_grd3d_convert_hybrid2(int nx,
                       int ny,
                       int nz,
                       double *coordsv,
                       double *zcornsv,
                       int *actnumsv,
                       int nzhyb,
                       double *p_zcornhyb_v,
                       int *p_actnumhyb_v,
                       double toplevel,
                       double botlevel,
                       int ndiv,
                       int *p_region_v,
                       int region)

{
    /* locals */
    int i, j, k, n, ic, ibp, ibh, inp = 0, inh = 0, khyb;
    int iflagt, iflagb, iflagr, actual_region = 0;
    double usetoplevel1, usetoplevel2, usetoplevel3, usetoplevel4;
    double usebotlevel1, usebotlevel2, usebotlevel3, usebotlevel4;
    double z1, z2, z3, z4, dz, zsum, ztop, zbot, zhyb, zsumh, usedz;

    usetoplevel1 = 0.0;
    usetoplevel2 = 0.0;
    usetoplevel3 = 0.0;
    usetoplevel4 = 0.0;

    /* thickness in horizontal section */
    dz = (botlevel - toplevel) / ndiv;

    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {

            iflagt = 1;
            iflagb = 1;
            ztop = UNDEF;
            zbot = -1 * UNDEF;

            /* first need to scan K column to see if any hybrid level within the column;
             * if hybrid region is found, the toplevel is used; otherwise the actual
             * cells is used: however as going down, it will end with tha last bottom
             * layer
             */

            iflagr = 0;
            usedz = dz;

            for (k = 1; k <= nz + 1; k++) {

                ibp = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                if (ibp < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                if (k <= nz) {
                    long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);
                    if (ic < 0) {
                        throw_exception("Loop through grid resulted in index outside "
                                        "boundary in grd3d_convert_hybrid2");
                        return;
                    }
                    actual_region = p_region_v[ic]; /* region is C order! */
                }

                /* this will end with last bottom layer unless hybrid region is
                 * found...*/
                if (actual_region != region && iflagr == 0) {
                    usetoplevel1 = zcornsv[4 * ibp + 1 * 1 - 1];
                    usetoplevel2 = zcornsv[4 * ibp + 1 * 2 - 1];
                    usetoplevel3 = zcornsv[4 * ibp + 1 * 3 - 1];
                    usetoplevel4 = zcornsv[4 * ibp + 1 * 4 - 1];
                }

                if (actual_region == region) {
                    iflagr = 1;
                    usetoplevel1 = toplevel;
                    usetoplevel2 = toplevel;
                    usetoplevel3 = toplevel;
                    usetoplevel4 = toplevel;
                }
            }

            /* first do the top-down truncation, collecting all at toplevel (which is
             * either the hybrid top OR the last layer)
             */

            for (k = 1; k <= nz + 1; k++) {
                ibp = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                if (ibp < 0 || ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                /* do for all corners */
                zsum = 0.0;

                /* CORNER 1 */
                z1 = zcornsv[4 * ibp + 1 * 1 - 1];
                if (z1 > usetoplevel1) {
                    p_zcornhyb_v[4 * ibh + 1 * 1 - 1] = usetoplevel1;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 1 - 1] = z1;
                }
                /* store avg top depth; will be used to truncate later if active*/
                zsum = zsum + z1;

                /* CORNER 2 */
                z2 = zcornsv[4 * ibp + 1 * 2 - 1];
                if (z2 > usetoplevel2) {
                    p_zcornhyb_v[4 * ibh + 1 * 2 - 1] = usetoplevel2;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 2 - 1] = z2;
                }
                /* store avg top depth; will be used to truncate later if active*/
                zsum = zsum + z2;

                /* CORNER 3 */
                z3 = zcornsv[4 * ibp + 1 * 3 - 1];
                if (z3 > usetoplevel3) {
                    p_zcornhyb_v[4 * ibh + 1 * 3 - 1] = usetoplevel3;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 3 - 1] = z3;
                }
                /* store avg top depth; will be used to truncate later if active*/
                zsum = zsum + z3;

                /* CORNER 4 */
                z4 = zcornsv[4 * ibp + 1 * 4 - 1];
                if (z4 > usetoplevel4) {
                    p_zcornhyb_v[4 * ibh + 1 * 4 - 1] = usetoplevel4;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 4 - 1] = z4;
                }
                /* store avg top depth; will be used to truncate later if active*/
                zsum = zsum + z4;

                /* now store top depth in input grid; it will be needed for later usage
                 */
                if (k <= nz) {
                    if (actnumsv[ibp] == 1 && iflagt == 1) {
                        ztop = zsum / 4.0;
                        iflagt = 0;
                    }

                    /* inherit the ACTNUM */
                    p_actnumhyb_v[ibh] = actnumsv[ibp];
                }
            }

            /* now doing it the other way (from bottom), but here the botlevel shall be
             * either hybdir bottom OR the base layer, which shall be similar to the one
             * used when looping top down (see above)
             */

            zsum = 0;
            khyb = nzhyb + 1;
            for (k = nz + 1; k >= 1; k--) {

                if (iflagr == 0) {
                    usebotlevel1 = usetoplevel1;
                    usebotlevel2 = usetoplevel2;
                    usebotlevel3 = usetoplevel3;
                    usebotlevel4 = usetoplevel4;
                } else {
                    usebotlevel1 = botlevel;
                    usebotlevel2 = botlevel;
                    usebotlevel3 = botlevel;
                    usebotlevel4 = botlevel;
                }

                ibp = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                ibh = x_ijk2ib(i, j, khyb, nx, ny, nzhyb + 1, 0);
                if (ibp < 0 || ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                /* in terms of active cells index, layer k _bottom_ shall refer to cell
                 * k-1 */
                if (k > 1) {
                    inp = x_ijk2ib(i, j, k - 1, nx, ny, nz + 1, 0);
                    inh = x_ijk2ib(i, j, khyb - 1, nx, ny, nzhyb + 1, 0);
                    if (inp < 0 || inh < 0) {
                        throw_exception("Loop through grid resulted in index outside "
                                        "boundary in grd3d_convert_hybrid2");
                        return;
                    }
                }

                /* CORNER 1 */
                z1 = zcornsv[4 * ibp + 1 * 1 - 1];
                if (z1 < usebotlevel1) {
                    p_zcornhyb_v[4 * ibh + 1 * 1 - 1] = usebotlevel1;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 1 - 1] = z1;
                }
                /* store avg bot depth; will be used to truncate later if active*/
                zsum = zsum + z1;

                /* CORNER 2 */
                z2 = zcornsv[4 * ibp + 1 * 2 - 1];
                if (z2 < usebotlevel2) {
                    p_zcornhyb_v[4 * ibh + 1 * 2 - 1] = usebotlevel2;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 2 - 1] = z2;
                }
                /* store avg bot depth; will be used to truncate later if active*/
                zsum = zsum + z2;

                /* CORNER 3 */
                z3 = zcornsv[4 * ibp + 1 * 3 - 1];
                if (z3 < usebotlevel3) {
                    p_zcornhyb_v[4 * ibh + 1 * 3 - 1] = usebotlevel3;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 3 - 1] = z3;
                }
                /* store avg bot depth; will be used to truncate later if active*/
                zsum = zsum + z3;

                /* CORNER 4 */
                z4 = zcornsv[4 * ibp + 1 * 4 - 1];
                if (z4 < usebotlevel4) {
                    p_zcornhyb_v[4 * ibh + 1 * 4 - 1] = usebotlevel4;
                } else {
                    p_zcornhyb_v[4 * ibh + 1 * 4 - 1] = z4;
                }
                /* store avg bot depth; will be used to truncate later if active*/
                zsum = zsum + z4;

                /* now bot depth from input grid */
                if (k > 1) {
                    if (actnumsv[inp] == 1 && iflagb == 1) {
                        zbot = zsum / 4.0;
                        iflagb = 0;
                    }
                    p_actnumhyb_v[inh] = actnumsv[inp];
                }
                khyb--;
            }

            /* now filling the intermediate */
            n = 0;
            if (iflagr == 0) {
                usedz = 0.0;
            }

            for (k = nz + 1; k <= nz + 1 + ndiv - 1; k++) {
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                if (ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                /* do for all corners */
                if (k > nz + 1) {
                    n++;

                    p_zcornhyb_v[4 * ibh + 1 * 1 - 1] = usetoplevel1 + n * usedz;
                    p_zcornhyb_v[4 * ibh + 1 * 2 - 1] = usetoplevel2 + n * usedz;
                    p_zcornhyb_v[4 * ibh + 1 * 3 - 1] = usetoplevel3 + n * usedz;
                    p_zcornhyb_v[4 * ibh + 1 * 4 - 1] = usetoplevel4 + n * usedz;
                }
                p_actnumhyb_v[ibh] = 1;
            }

            /* truncate - ensure same volume, first from top, eval by cell centre */
            for (k = 1; k <= nzhyb; k++) {
                zsumh = 0.0;
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                inh = x_ijk2ib(i, j, k + 1, nx, ny, nzhyb + 1, 0);
                if (inh < 0 || ibh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                /* do for all corners */
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * ibh + 1 * ic - 1];
                }
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * inh + 1 * ic - 1];
                }
                zhyb = 0.125 * zsumh; /* cell center */

                if (p_actnumhyb_v[ibh] == 1 && zhyb < ztop) {
                    p_actnumhyb_v[ibh] = 0;
                }
            }

            /* truncate - ensure same volume, now from bot, eval by cell centre */
            for (k = nzhyb + 1; k > 1; k--) {
                zsumh = 0.0;
                ibh = x_ijk2ib(i, j, k, nx, ny, nzhyb + 1, 0);
                inh = x_ijk2ib(i, j, k - 1, nx, ny, nzhyb + 1, 0);
                if (ibh < 0 || inh < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_convert_hybrid2");
                    return;
                }
                /* do for all corners */
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * ibh + 1 * ic - 1];
                }
                for (ic = 1; ic <= 4; ic++) {
                    zsumh = zsumh + p_zcornhyb_v[4 * inh + 1 * ic - 1];
                }
                zhyb = 0.125 * zsumh;

                if (p_actnumhyb_v[inh] == 1 && zhyb > zbot) {
                    p_actnumhyb_v[inh] = 0;
                }
            }
        }
    }
}

void
grd3d_convert_hybrid(int nx,
                     int ny,
                     int nz,

                     double *coordsv,
                     long ncoordin,
                     double *zcornsv,
                     long nzcornin,
                     int *actnumsv,
                     long nactin,

                     int nzhyb,

                     double *p_zcornhyb_v,
                     long nzcornhybin,
                     int *p_actnumhyb_v,
                     long nacthybin,

                     double toplevel,
                     double botlevel,
                     int ndiv,

                     int *p_region_v,
                     long nreg,

                     int region)
{

    if (region <= 0) {
        _grd3d_convert_hybrid1(nx, ny, nz, coordsv, zcornsv, actnumsv, nzhyb,
                               p_zcornhyb_v, p_actnumhyb_v, toplevel, botlevel, ndiv);
    } else {

        _grd3d_convert_hybrid2(nx, ny, nz, coordsv, zcornsv, actnumsv, nzhyb,
                               p_zcornhyb_v, p_actnumhyb_v, toplevel, botlevel, ndiv,
                               p_region_v, region);
    }
}
