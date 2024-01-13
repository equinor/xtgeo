/*
 ***************************************************************************************
 *
 * NAME:
 *    cube_ijk_from_xyz.c
 *
 * DESCRIPTION:
 *    Find the "best" (see flag) IJK index given one X Y Z
 *    point (e.g. a map value)
 *
 * ARGUMENTS:
 *    i,j,k          o     Index to be returned (int pointers)
 *    rx,ry,rz       o     Relative to origo 0 for P, returned
 *    x,y,z          i     X Y Z location for point P
 *    xori,xinc etc  i     XYZ origo and step of cube
 *    nx,ny,nz       i     max I J K of cube
 *    rot_deg        i     Rotation (degrees), positive anti clock
 *    yflip          i     If the Y axis is flipped then -1, else 1
 *    flag           i     Options flag:
 *                   i     0: cell mode; treat cube node as a cell center,
 *                            return location of cell when P is closest
 *                   i     1: Node mode; find lower left corner IJK
 *                            i.e. P is within i,j i+1,j j+1,i, i+1,j+1
 *                         >= 10; skip I J calc (keep current, use static value)
 *
 * RETURNS:
 *    Function: -1: point is outside cube
 *               0: OK, point is in cube
 *    Result I J K are updated
 *
 *
 * TODO/ISSUES/BUGS:
 *    What if Cube has YFLIP -1? Seems to be solved now, but more testing may
 *    be needed
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

int
cube_ijk_from_xyz(int *i,
                  int *j,
                  int *k,
                  double *rx,
                  double *ry,
                  double *rz,
                  double x,
                  double y,
                  double z,
                  double xori,
                  double xinc,
                  double yori,
                  double yinc,
                  double zori,
                  double zinc,
                  int nx,
                  int ny,
                  int nz,
                  double rot_deg,
                  int yflip,
                  int flag)
{
    /* locals */
    static int ii = 0, jj = 0, ier = 0;
    int kk;
    double pz, usex, usey, usez;
    static double rrx = 0.0, rry = 0.0;

    usex = x;
    usey = y;
    usez = z;

    if (flag < 10) {
        ier = sucu_ij_from_xy(&ii, &jj, &rrx, &rry, usex, usey, xori, xinc, yori, yinc,
                              nx, ny, yflip, rot_deg, flag);
    }

    *i = ii;
    *j = jj;

    *rx = rrx;
    *ry = rry;

    if (z < zori || z > zori + (nz - 1) * zinc) {
        /* point is above or below cube (node wise thinking) */
        return -1;
    }

    if (ier == -1) {
        return -1;
    }

    if (ier != 0)
        return ier;

    pz = usez - zori;

    if (flag == 0 || flag == 10) {
        kk = (int)((pz + 0.5 * zinc) / zinc) + 1;
        if (kk < 1 || kk > nz) {
            return -1;
        }
        *k = kk;
    } else {
        kk = (int)(pz / zinc) + 1;
        if (kk < 1 || kk >= nz) {
            return -1;
        }
        *k = kk;
    }

    *rz = pz;

    return EXIT_SUCCESS;
}
