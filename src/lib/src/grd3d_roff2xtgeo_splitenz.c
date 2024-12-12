/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_zcorn.c
 *
 * DESCRIPTION:
 *    Convert from ROFF arrays to XTGeo arrays: ZCORN
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     Dimension (nx=ncol, ny=nrow, nz=nlay)
 *    zoffset          i     Offsets in XYZ spesified in ROFF
 *    zscale           i     Scaling in XYZ spesified in ROFF
 *    splitenz         i     Split node vector
 *    zdata            i     Input zdata array ROFF fmt
 *    zcornsv          o     Output zcorn array XTGEO fmt
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              -1: Unsupported split type, supported types is 1 and 4
 *              -2: Incorrect size of splitenz
 *              -3: Incorrect size of zdata
 *              -4: Incorrect size of zcorn
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "logger.h"

int
grd3d_roff2xtgeo_splitenz(int nz,
                          float zoffset,
                          float zscale,
                          char *splitenz,
                          long nsplitenz,
                          float *zdata,
                          long nzdata,
                          float *zcornsv,
                          long nzcorn)

{

    // We Read one corner line (pillar) from zdata at a time (one for each
    // i,j), transform it according to zoffset and zscale, and put it in to
    // zcornsv in reverse order.

    // As i and j order and size is the same for both zcornsv and zdata, we can
    // ignore it here and place one pillar at the time regardless of how many
    // pillars there are.

    long num_row = 4 * nz;
    if (nzcorn % num_row != 0) {
        return -4;
    }
    if (nsplitenz != nzcorn / 4) {
        return -2;
    }
    float *pillar = malloc(sizeof(float) * num_row);
    size_t it_zdata = 0, it_splitenz = 0, it_zcorn = 0;
    while (it_zcorn < nzcorn) {
        for (size_t it_pillar = 0; it_pillar < num_row;) {
            char split = splitenz[it_splitenz++];
            if (split == 1) {
                // There is one value for this corner which
                // we must duplicate 4 times in zcornsv
                if (it_zdata >= nzdata) {
                    free(pillar);
                    return -3;
                }
                float val = (zdata[it_zdata++] + zoffset) * zscale;
                for (int n = 0; n < 4; n++) {
                    pillar[it_pillar++] = val;
                }
            } else if (split == 4) {
                // There are four value for this corner which
                // we must duplicate 4 times in zcornsv
                if (it_zdata + 3 >= nzdata) {
                    free(pillar);
                    return -3;
                }
                // As we place the pillar in reverse order into zcornsv,
                // we must put zdata in reverse order into pillar to
                // preserve n,s,w,e directions.
                pillar[it_pillar + 3] = (zdata[it_zdata++] + zoffset) * zscale;
                pillar[it_pillar + 2] = (zdata[it_zdata++] + zoffset) * zscale;
                pillar[it_pillar + 1] = (zdata[it_zdata++] + zoffset) * zscale;
                pillar[it_pillar] = (zdata[it_zdata++] + zoffset) * zscale;
                it_pillar += 4;
            } else {
                free(pillar);
                return -1;
            }
        }
        // Put the pillar into zcornsv in reverse order
        for (size_t it_pillar = num_row; it_pillar >= 1;) {
            zcornsv[it_zcorn++] = pillar[--it_pillar];
        }
    }

    if (it_splitenz != nsplitenz) {
        free(pillar);
        return -2;
    }
    if (it_zdata != nzdata) {
        free(pillar);
        return -3;
    }
    if (it_zcorn != nzcorn) {
        free(pillar);
        return -4;
    }
    free(pillar);
    return EXIT_SUCCESS;
}
