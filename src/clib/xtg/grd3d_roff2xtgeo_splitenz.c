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
 *    coordsv          o     Output zcorn array XTGEO fmt
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

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_roff2xtgeo_splitenz(
                       int nz,
                       float zoffset,
                       float zscale,
                       char *splitenz,
                       long nsplitenz,
                       float *zdata,
                       long nzdata,
                       float *zcornsv,
                       long nzcorn)

{
    long num_row = 4 * nz;
    float pillar[num_row];
    if(nzcorn % num_row != 0){
        return -4;
    }
    if(nsplitenz != nzcorn / 4){
        return -2;
    }
    size_t it_zdata = 0, it_splitenz = 0, it_zcorn = 0;
    while(it_zcorn < nzcorn){
        for(size_t it_pillar = 0; it_pillar < num_row;){
            char split = splitenz[it_splitenz++];
            if(split == 1){
                if(it_zdata >= nzdata){
                    return -3;
                }
                float val = (zdata[it_zdata++] + zoffset) * zscale;
                for(int n = 0; n < 4; n++){
                    pillar[it_pillar++] = val;
                }
            } else if(split == 4){
                    if(it_zdata+3 >= nzdata){
                        return -3;
                    }
                    pillar[it_pillar+3] = (zdata[it_zdata++] + zoffset) * zscale;
                    pillar[it_pillar+2] = (zdata[it_zdata++] + zoffset) * zscale;
                    pillar[it_pillar+1] = (zdata[it_zdata++] + zoffset) * zscale;
                    pillar[it_pillar]   = (zdata[it_zdata++] + zoffset) * zscale;
                    it_pillar += 4;
            } else {
                return -1;
            }
        }
        for(size_t it_pillar = num_row; it_pillar >= 1;){
            zcornsv[it_zcorn++] = pillar[--it_pillar];
        }
    }

    if(it_splitenz != nsplitenz){
        return -2;
    }
    if(it_zdata != nzdata){
        return -3;
    }
    if(it_zcorn != nzcorn){
        return -4;
    }

    return EXIT_SUCCESS;
}
