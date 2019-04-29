/*
 ******************************************************************************
 *
 * Export Storm binary map format (no rotation)
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    surf_export_storm_binary.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export a map to Storm binary. Note the map must be unrotated!
 *
 * ARGUMENTS:
 *    filename       i     File name, character string
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    mxy            i     Length of map array (for Swig numpy conv.)
 *    option         i     Options flag for later usage
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
 ******************************************************************************
 */

int surf_export_storm_bin(
                          char *filename,
                          int mx,
                          int my,
                          double xori,
                          double yori,
                          double xinc,
                          double yinc,
                          double *p_map_v,
                          long mxy,
                          double zmin,
                          double zmax,
                          int option,
                          int debug
                          )
{

    /* local declarations */
    int i;
    double xmax, ymax;
    double dbl_value;
    int swap = 0;

    char    s[24] = "surf_export_storm_bin";

    FILE    *fc;

    xtgverbose(debug);
    xtg_speak(s,1,"Write Storm binary map file ...", s);

    swap=x_swap_check();

    xmax = xori + (mx - 1) * xinc;
    ymax = yori + (my - 1) * yinc;

    fc = fopen(filename,"wb");

    if (fc == NULL) {
        xtg_warn(s, 0, "Some thing is wrong with requested filename <%s>",
                 filename);
        xtg_error(s, "Could be: Non existing folder, wrong permissions ? ..."
                  " anyway: STOP!", s);
        return -9;
    }

    fprintf(fc,"STORMGRID_BINARY\n\n");
    fprintf(fc,"%d %d %lf %lf\n%lf %lf %lf %lf\n",
	    mx, my, xinc, yinc, xori, xmax, yori, ymax);

    for (i = 0; i < mxy; i++) {
	dbl_value = p_map_v[i];

	if (dbl_value > UNDEF_MAP_LIMIT) {
	    dbl_value = UNDEF_MAP_STORM;
	}

	/* byte swapping if needed */
	if (swap == 1) SWAP_DOUBLE(dbl_value);

	if (fwrite(&dbl_value, 8, 1, fc) != 1) {
            xtg_error(s, "Error writing to Storm format. Bug?");
            return -2;
        }
    }

    fclose(fc);

    return EXIT_SUCCESS;
}
