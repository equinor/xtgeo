/*
 ***************************************************************************************
 *
 * NAME:
 *    x_chk_point_in_octagedron.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given X Y Z vectors, determine if a point is inside or on the boundary of a
 *    octahedron.
 *
 * ARGUMENTS:
 *    x, y, z             i     point
 *    coor                i     The coordinates as 24 lenth array
 *
 * RETURNS:
 *    -1 if outside, 0 on boundary and 1 if inside
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include <math.h>

#include "libxtg.h"
#include "libxtg_.h"

#define CBIG 1E32

int _check_envelope(double x, double y, double z, double *coor)
{

    /*
     * Check first if point is outside envelope which is a cube based on most extreme
     * values
     */

    double vminx = CBIG;
    double vmaxx = -CBIG;
    double vminy = CBIG;
    double vmaxy = -CBIG;
    double vminz = CBIG;
    double vmaxz = -CBIG;

    int i;
    for (i = 1; i <= 8; i++) {
	if (vminx > coor[3 * i - 3]) vminx = coor[3 * i - 3];
	if (vmaxx < coor[3 * i - 3]) vmaxx = coor[3 * i - 3];
	if (vminy > coor[3 * i - 2]) vminy = coor[3 * i - 2];
	if (vmaxy < coor[3 * i - 2]) vmaxy = coor[3 * i - 2];
	if (vminz > coor[3 * i - 1]) vminz = coor[3 * i - 1];
	if (vmaxz < coor[3 * i - 1]) vmaxz = coor[3 * i - 1];
    }

    if (x < vminx) return -1;
    if (x > vmaxx) return -1;
    if (y < vminy) return -1;
    if (y > vmaxy) return -1;
    if (z < vminz) return -1;
    if (z > vmaxz) return -1;

    return 0;
}


int _inside_plane(int ic0, int ic1, int ic2, double x, double y, double z,
                  double *coor, int flip) {

    /*
     * A plane consists of 3 corners, and the normal vector result is positive
     * upwards when a normal right handed system. So be careful when thinking base;
     * then turn head upside down!
     * If flip, we have a left handed system insted. Corners counts from 1 to 8
     */

    double nvec[4], pt[9];
    int ic;

    ic = ic0;  /*  cell corners, 1..8 */
    pt[0] = coor[3 * ic - 3]; pt[1] = coor[3 * ic - 2], pt[2] = coor[3 * ic - 1];

    ic = ic1;
    pt[3] = coor[3 * ic - 3]; pt[4] = coor[3 * ic - 2], pt[5] = coor[3 * ic - 1];

    ic = ic2;
    pt[6] = coor[3 * ic - 3]; pt[7] = coor[3 * ic - 2], pt[8] = coor[3 * ic - 1];

    int ier = x_plane_normalvector(pt, nvec, 0, 0);

    if (ier != 0) {
        /*  could not make normal vector... */
        return 0;
    }

    double prod = (nvec[0] * x + nvec[1] * y + nvec[2] * z + nvec[3]) * flip;

    if (prod < 0.0) return 1;  /*  1 for "INSIDE" */
    if (prod > 0.0) return -1;  /* for OUTSIDE */

    return 0;
}

int x_chk_point_in_octahedron (
    double x,
    double y,
    double z,
    double *coor,
    int flip
    )
{

    /* double pp[3], pm[3], tri[4][3]; */
    /* int   istat[13], ier; */
    /* double cbig=1.0e14, vminx, vmaxx, vminy, vmaxy, vminz, vmaxz; */
    /* int   i, ic, isum; */

    /*
     * Initialize
     */

    if (_check_envelope(x, y, z, coor) == -1) return -1;

    /*
     * Each side face of a cell can be regarded as 2 plane triangles. However the way
     * one divides it matters. The idea here is to compute the normal vector to
     * each plane in such a way that it is pointing outwards. Hence, only of a point
     * in on the negative side of all trangles, it is inside (Hence inside is given
     * value 1). Since triangle division matters and this is non-unique we try all
     * and that is why there are 4 normal vectors per side
     */
    int score = 0, i, ist[24];

    /* top and base */
    ist[0] = _inside_plane(1, 2, 3, x, y, z, coor, flip);
    ist[1] = _inside_plane(4, 3, 2, x, y, z, coor, flip);
    ist[2] = _inside_plane(2, 4, 1, x, y, z, coor, flip);
    ist[3] = _inside_plane(3, 1, 4, x, y, z, coor, flip);
    score = 0; for (i = 0; i < 4; i++) score += ist[i];
    /* for (i = 0; i < 4; i++) printf("IST index %d is %d\n",i , ist[i]); */
    if (score < 0) return -1;

    ist[4] = _inside_plane(5, 7, 6, x, y, z, coor, flip);
    ist[5] = _inside_plane(8, 6, 7, x, y, z, coor, flip);
    ist[6] = _inside_plane(6, 5, 8, x, y, z, coor, flip);
    ist[7] = _inside_plane(7, 8, 5, x, y, z, coor, flip);
    score = 0; for (i = 4; i < 8; i++) score += ist[i];
    /* for (i = 4; i < 8; i++) printf("IST index %d is %d\n",i , ist[i]); */
    if (score < 0) return -1;

    /* front  and back */
    ist[8] = _inside_plane(1, 5, 2, x, y, z, coor, flip);
    ist[9] = _inside_plane(6, 2, 5, x, y, z, coor, flip);
    ist[10] = _inside_plane(5, 6, 1, x, y, z, coor, flip);
    ist[11] = _inside_plane(2, 1, 6, x, y, z, coor, flip);
    score = 0; for (i = 8; i < 12; i++) score += ist[i];
    if (score < 0) return -1;

    ist[12] = _inside_plane(3, 4, 7, x, y, z, coor, flip);
    ist[13] = _inside_plane(8, 7, 4, x, y, z, coor, flip);
    ist[14] = _inside_plane(4, 8, 3, x, y, z, coor, flip);
    ist[15] = _inside_plane(7, 3, 8, x, y, z, coor, flip);
    score = 0; for (i = 12; i < 16; i++) score += ist[i];
    if (score < 0) return -1;

    /* left and right*/
    ist[16] = _inside_plane(1, 3, 5, x, y, z, coor, flip);
    ist[17] = _inside_plane(7, 5, 3, x, y, z, coor, flip);
    ist[18] = _inside_plane(5, 1, 7, x, y, z, coor, flip);
    ist[19] = _inside_plane(3, 7, 1, x, y, z, coor, flip);
    score = 0; for (i = 16; i < 20; i++) score += ist[i];
    if (score < 0) return -1;

    ist[20] = _inside_plane(6, 8, 2, x, y, z, coor, flip);
    ist[21] = _inside_plane(4, 2, 8, x, y, z, coor, flip);
    ist[22] = _inside_plane(2, 6, 4, x, y, z, coor, flip);
    ist[23] = _inside_plane(8, 4, 6, x, y, z, coor, flip);
    score = 0; for (i = 20; i < 24; i++) score += ist[i];
    if (score < 0) return -1;

    /* cumulattive score */
    score = 0;
    for (i = 0; i < 24; i++){
        score += ist[i];
        /* printf("SCORE %d  -->  %d\n", i, ist[i]); */
    }
    return score;
}
