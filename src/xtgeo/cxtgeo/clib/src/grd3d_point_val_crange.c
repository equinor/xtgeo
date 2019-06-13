/*
****************************************************************************************
 *
 * Get a property value from a range of I, J
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_point_val_crange
 *
 * DESCRIPTION:
 *    From a given point in XYZ, find the cell value. This routine assumes that
 *    I, J range is limited to speed up. It also uses a proposed I, J as start point.
 *    This routine is applied when sampling a fence from a 3D grid parameter
 *
 * ARGUMENTS:
 *
 *     x,y,z           i      input points
 *     nx..nz          i      grid dimensions
 *     p_coord_v       i      grid coords
 *     p_zcorn_v       i      grid zcorn
 *     p_actnum_v      i      grid active cell indicator
 *     p_prop_v        i      property to work with
 *     i1, i2...k2     i      cell index ranges in I J K
 *     value           o      value to return (UNDEF if not found)
 *     ib             i/o     Start point cell index, updated to actual cell index
 *     option          i      To be
 *     debug           i      debug/verbose flag

 * RETURNS:
 *     void
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"


/*
 ***************************************************************************************
 * Private function
 ***************************************************************************************
 */

long _find_ib(x, y, imin, imax, jmin, jmax, kmin, kmax, nx, ny, nz,
              p_coord_v, p_zcorn_v, debug)
{
    /* cell not found -9; otherwise >= 0 but active may be 0 if inactive cell*/

    int i, j, k;
    long ib = -1;

    for (k = kmin; k <= kmax; k++) {
        for (j = jmin; j <= jmax; j++) {
            for (i = imin; i <= imax; i++) {

                if (debug>2) {
                    xtg_speak(s,3,"Cell IJK: %d %d %d",i,j,k);
                }
                ib = x_ijk2ib(i,j,k,nx,ny,nz,0);
                /* get the corner for the cell */
                grd3d_corners(i,j,k,nx,ny,nz,p_coor_v,p_zcorn_v,
			      corners,debug);
                inside=x_chk_point_in_cell(x,y,z,corners,1,debug);

                if (inside > 0) {
                    xtg_speak(s,2,"Found at IJK: %d %d %d",i,j,k);
                    value = p_val_v[ib];
                    *ibs = ib;
                    return ib;
                }

            }
        }
    }
    return -9;
}

/*
 ***************************************************************************************
 * Public function
 ***************************************************************************************
 */

void grd3d_point_val_crange(
                            int ibstart,
                            int kzonly,
                            double x,
                            double y,
                            double z,
                            int nx,
                            int ny,
                            int nz,
                            double *p_coor_v,
                            double *p_zcorn_v,
                            int *p_actnum_v,
                            double *p_val_v,
                            double *value,
                            int imin,
                            int imax,
                            int jmin,
                            int jmax,
                            int kmin,
                            int kmax,
                            long *ibs,
                            int option,
                            int debug
                            )

{
    /* locals */
    int i, j, k, inside;
    long ibstart, ib;
    int istart, jstart, kstart;
    double corners[24];
    char  sbn[24] = "grd3d_point_val_crange";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering %s", sbn);

    xtg_speak(s,2,"IBSTART %d", ibstart);

    ibstart = *ibs;

    if (ibstart < 0) {
	ibstart = x_ijk2ib(imin, jmin, kmin, nx, ny, nz, 0);
    }

    x_ib2ijk(ibstart, &istart, &jstart, &kstart, nx, ny, nz, 0);

    *value = UNDEF_FLOAT;

    /* first try, loop over a limited cells in range based on start*/
    k1 = kstart - 1; if (k1 < kmin) k1 = kmin;
    k2 = kstart + 1; if (k2 > kmax) k2 = kmax;
    j1 = jstart - 1; if (j1 < jmin) j1 = jmin;
    j2 = jstart + 1; if (j2 > jmax) j2 = jmax;
    i1 = istart - 1; if (i1 < imin) i1 = imin;
    i2 = istart + 1; if (i2 > imax) i2 = imax;

    ib = _find_ib(x, y, i1, i2, j1, j2, k1, k2, nx, ny, nz,
                  p_coord_v, p_zcorn_v, debug);

    /* second try; fall back loop over fuller range */
    if (ib == -9) {
        ib = _find_ib(x, y, imin, imax, jmin, jmax, kmin, kmax, nx, ny, nz,
                      p_coord_v, p_zcorn_v, debug);
    }

    if (ib > 0) {
        *ibs = ib;
        if (p_actnum_v[ib] == 1) {
            *value = p_val_v[ib];
        }
        return EXIT_SUCCESS;
    }

    return -1;
}
