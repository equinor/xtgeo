/*
 *******************************************************************************
 *
 * Import Irap ascii map (with rotation)
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    surf_import_irap_ascii.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Import a map on Irap ascii format.
 *
 * ARGUMENTS:
 *    filename       i     File name, character string
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    option         i     0: read only dimensions (for memory alloc), 1 all
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    Issue: The surf_* routines in XTGeo will include rotation, and origins
 *           (not xmin etc ) and steps are used to define the map extent.
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

int surf_import_irap_ascii (
                            char   *file,
                            int    mode,
                            int    *nx,
                            int    *ny,
                            long   *ndef,
                            double *xori,
                            double *yori,
                            double *xinc,
                            double *yinc,
                            double *rot,
                            double *p_map_v,
                            long   nmap,
                            int    option,
                            int    debug
                            )
{

    /* locals*/
    int idum, ib, ic, i, j, k, iok;
    long ncount;
    FILE *fd;
    char s[24] = "surf_import_irap_ascii";

    float rdum, value;
    double dval;

    xtgverbose(debug);

    /* read header */
    xtg_speak(s, 2, "Entering routine");


    fd = fopen(file,"r");
    if (fd == NULL) {
	xtg_speak(s,2,"Opening Irap ascii file FAILED!");
    }
    else{
	xtg_speak(s,2,"Opening Irap ascii file...OK!");
    }


    ncount = 0;

    /* read header */
    xtg_speak(s,2,"Reading header!");

    iok = fscanf(fd, "%d %d %lf %lf %lf %f %lf %f %d %lf %f %f %d %d %d %d %d "
                 "%d %d", &idum, ny, xinc, yinc,
                 xori, &rdum, yori, &rdum,
                 nx, rot, &rdum, &rdum,
                 &idum, &idum, &idum, &idum, &idum, &idum, &idum);

    if (iok < 19) xtg_error(s, "Something went wrong with Irap ASCII import. "
                            "Report as BUG");

    if (*rot < 0.0) *rot = *rot + 360.0;

    if (mode == 0) {
        return EXIT_SUCCESS;
    }

    /* read values */
    for (ib = 0; ib < nmap; ib++) {
	iok = fscanf(fd, "%f", &value);

	if (value == UNDEF_MAP_IRAP) {
	    dval = UNDEF_MAP;
	}
        else{
	    dval = value;
            ncount ++;
	}

        /* convert to C order */
        x_ib2ijk(ib, &i, &j, &k, *nx, *ny, 1, 0);
        ic = x_ijk2ic(i, j, 1, *nx, *ny, 1, 0);

        p_map_v[ic] = dval;

    }

    *ndef = ncount;

    return EXIT_SUCCESS;
}
