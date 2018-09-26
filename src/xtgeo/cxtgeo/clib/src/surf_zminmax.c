#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    surf_zminmax.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    get min and max Z value
 *
 * ARGUMENTS:
 *    nx         i      NX (columns)
 *    ny         i      NY (rows)
 *    p_map_v    i      pointer to X values (must be allocated in caller)
 *    zmin       o      Z value minimum result
 *    zmax        i     Z value maximum result
 *    debug      i      Debug flag
 *
 * RETURNS
 *    EXIT success unless all maps nodes are UNDEF
 *
 * LICENCE:
 *    CF XTGeo license
 ******************************************************************************
 */

int surf_zminmax(
                 int nx,
                 int ny,
                 double *p_map_v,
                 double *zmin,
                 double *zmax,
                 int debug
                 )
{
    /* locals */
    char sbn[24] = "surf_zminmax";

    xtgverbose(debug);
    xtg_speak(sbn, 2, "Running %s", sbn);

    long mxy = nx * ny;
    long ic;
    int found = 0;


    double zzmin = VERYLARGEPOSITIVE;
    double zzmax = VERYLARGENEGATIVE;

    for (ic = 0; ic < mxy; ic++) {
        if (p_map_v[ic] < UNDEF_LIMIT) {
            found = 1;
            if (p_map_v[ic] > zzmax) zzmax = p_map_v[ic];
            if (p_map_v[ic] < zzmin) zzmin = p_map_v[ic];
        }
    }

    xtg_speak(sbn, 2, "Map min max %f %f", zzmin, zzmax);

    if (found == 0) {*zmin = UNDEF; *zmax = UNDEF; return -2;}

    *zmin = zzmin;
    *zmax = zzmax;

    return EXIT_SUCCESS;
}
