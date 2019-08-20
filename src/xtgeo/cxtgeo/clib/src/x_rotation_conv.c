/*
 ******************************************************************************
 *
 * Convert angles (rotation) different modes
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    x_rotation_convert.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Transforms different rotation schemes. Ie. to transform from radians
 *    azimuth to degrees anticlock from x:
 *    deg = z_rotation_conv(0.343, 3, 0, 0, 1)
 *
 * ARGUMENTS:
 *    ain            i     Input angle
 *    ainmode        i     Input angle mode:
 *                               0 degrees, anti-clock from X
 *                               1 radians, anti-clock from X
 *                               2 degrees, clock from Y (azimuth)
 *                               3 radians, clock from Y (azimuth)
 *    mode           i     Mode of returning angle (same codes as ainmode)
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Transformed angle.
 *
 * TODO/ISSUES/BUGS:
 *    UNFINISHED! All combinations of conversion are not finished yet
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */
#include "libxtg.h"
#include "libxtg_.h"


double x_rotation_conv (
			double  ain,
			int     ainmode,
			int     mode,
			int     option,
			int     debug
		    )
{
    /* locals */
    char     s[24] = "x_rotation_conv";
    double   result = 0.0;


    xtgverbose(debug);
    xtg_speak(s,3,"Entering routine");

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if (ainmode == 0 || ainmode == 2) {
	if (ain<-360 || ain>360) {
	    xtg_error(s,"Input angle (degrees) out of boundary: %f",ain);
	    return -9;
	}
    }

    if (ainmode == 1 || ainmode == 3) {
	if (ain<-2*PI || ain>2*PI) {
	    xtg_error(s,"Input angle (radians) out of boundary: %f",ain);
	    return -9;
	}
    }

    /*
     * ------------------------------------------------------------------------
     * Use degrees when comptuting
     * ------------------------------------------------------------------------
     */
    if (ainmode == 1 || ainmode == 3) ain = ain * 180.0 / PI;

    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */

    /* from angle to azimuth */
    if (ainmode <= 1 && mode >= 2) {
	result = -ain + 90;
	if (result > 360) result=result-360;
        if (ainmode == 1) result = result * PI / 180.0;
    }

    /* convert degrees azimuth angle to degrees normal angle */
    if (ainmode >= 2 && mode <= 1) {
	result = 450 - ain;
	if (result > 360) result=result-360;
        if (ainmode == 3) result = result * PI / 180.0;
    }

    /* convert radians azimuth angle to degrees normal angle */
    if (ainmode==3 && mode==0) {
        ain = ain * 180 / PI;
	result = 450 - ain;
	if (result > 360) result=result-360;
    }


    return result;
}
