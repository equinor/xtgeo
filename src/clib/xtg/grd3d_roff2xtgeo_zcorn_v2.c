/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_zcorn_v2.c
 *
 * DESCRIPTION:
 *    Alternative version Convert from ROFF arrays to XTGeo arrays: ZCORN
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     NCOL, NROW, NLAY dimens
 *    *offset          i     Offsets in XYZ spesified in ROFF
 *    *scale           i     Scaling in XYZ spesified in ROFF
 *    p_splitenz_v     i     Split node vector with dimensions
 *    p_zdata_v        i     Input zdata array ROFF fmt with dimensions
 *    coordsv          o     Output zcorn array XTGEO fmt with dimensions
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
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

static long
_ibwhere(long icin, int where, int nx, int ny, int nz)
{
    // where: 1: SW           NW=3  | NE=4      Note cell corner numbers differs
    //        2: SE          -------|------        3 |----------| 4
    //        3: NW           SW=1  | SE=2           |          |
    //        4  NE                                1 |----------| 2
    //                        Corner crosses

    // since the roxar grid is at nodes, increase with one in all directions
    int nnx = nx + 1;
    int nny = ny + 1;
    int nnz = nz + 1;

    int inode = 1;
    int jnode = 1;
    int knode = 1;
    x_ic2ijk(icin, &inode, &jnode, &knode, nnx, nny, nnz, 0);

    // the knode is listed from base in ROFF
    knode = nnz - knode + 1;

    // now find the corresponding I J K cell
    int ix, jy;
    int kz = knode - 1;

    if (knode == nnz) {
        kz = knode;
    }

    if (where == 1) {
        ix = inode - 1;
        jy = jnode - 1;
    } else if (where == 2) {
        ix = inode;
        jy = jnode - 1;
    } else if (where == 3) {
        ix = inode - 1;
        jy = jnode;
    } else if (where == 4) {
        ix = inode;
        jy = jnode;
    }

    if (ix < 1 || ix > nx || jy < 1 || jy > ny || kz < 1 || kz > nnz)
        return -1;

    if (knode == nnz) {
        return x_ijk2ib(ix, jy, kz, nx, ny, nnz, 0);
    }
    return x_ijk2ib(ix, jy, kz, nx, ny, nz, 0);
}

void
grd3d_roff2xtgeo_zcorn_v2(int nx,
                          int ny,
                          int nz,
                          float xoffset,
                          float yoffset,
                          float zoffset,
                          float xscale,
                          float yscale,
                          float zscale,
                          int *p_splitenz_v,
                          long n1,
                          float *p_zdata_v,
                          long n2,
                          double *zcornsv,
                          long nzcorn)

{

    long ic;
    float cross[4];

    logger_info(LI, FI, FU, "Enter %s", FU);

    long nc = 0;
    for (ic = 0; ic < n1; ic++) {
        int cr;
        for (cr = 0; cr < 4; cr++) {
            cross[cr] = (p_zdata_v[nc] + zoffset) * zscale;
            if (p_splitenz_v[ic] == 4) {
                nc++;
            }
        }

        for (cr = 0; cr < 4; cr++) {
            long ib = _ibwhere(ic, cr, nx, ny, nz);
            if (ib >= 0) {
                // printf("IB at %ld\n", ib);
                zcornsv[4 * ib + 1 * (4 - cr + 1) - 1] = cross[cr];
            }
        }
        nc += 1;
    }
    logger_info(LI, FI, FU, "Exit from %s", FU);
}