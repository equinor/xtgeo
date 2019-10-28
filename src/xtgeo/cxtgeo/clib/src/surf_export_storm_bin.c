/*
****************************************************************************************
 *
 * Export Storm binary map format (no rotation)
 *
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

/*
****************************************************************************************
 *
 * NAME:
 *    surf_export_storm_binary.c
 *
 * DESCRIPTION:
 *    Export a map to Storm binary. Note the map must be unrotated!
 *
 * ARGUMENTS:
 *    fc             i     File handle
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    mxy            i     Length of map array (for Swig numpy conv.)
 *    option         i     Options flag for later usage
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
 ***************************************************************************************
 */

int surf_export_storm_bin(
                          FILE *fc,
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
                          int option
                          )
{

    /* local declarations */
    int i;
    double xmax, ymax;
    double dbl_value;
    int swap = 0;

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Write Storm binary map file ... (%s)", __FUNCTION__);

    swap=x_swap_check();

    xmax = xori + (mx - 1) * xinc;
    ymax = yori + (my - 1) * yinc;

    if (fc == NULL) return -1;

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
            logger_error(__LINE__, "Error writing to Storm format. Bug?");
            return -1;
        }
    }


    return EXIT_SUCCESS;
}
