/*
****************************************************************************************
 *
 * Export Irap binary map (with rotation)
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
 *    surf_export_irap_bin.c
 *
 * DESCRIPTION:
 *    Export a map on Irap binary format.
 *
 * ARGUMENTS:
 *    fc             i     File handle
 *    mx             i     Map dimension X (I)
 *    my             i     Map dimension Y (J)
 *    xori           i     X origin coordinate
 *    yori           i     Y origin coordinate
 *    xinc           i     X increment
 *    yinc           i     Y increment
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    p_surf_v       i     1D pointer to map/surface values pointer array
 *    ntot           i     Number of nodes (for allocation)
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
 ***************************************************************************************
 */

void _writeint(FILE *fc, int ival, int swap)
{
    if (swap) SWAP_INT(ival);

    if (fwrite(&ival, sizeof(int), 1, fc) != 1) {
        logger_critical(__LINE__, "Cannot write int to file! <%s>", __FUNCTION__);
    }
}

void _writefloat(FILE *fc, float fval, int swap)
{
    if (swap) SWAP_FLOAT(fval);

    if (fwrite(&fval, sizeof(float), 1, fc) != 1) {
        logger_critical(__LINE__, "Cannot write float to file! <%s>", __FUNCTION__);
    }
}


int surf_export_irap_bin(
			 FILE   *fc,
			 int    mx,
			 int    my,
			 double xori,
			 double yori,
			 double xinc,
			 double yinc,
			 double rot,
			 double *p_map_v,
                         long   ntot,
			 int    option
			 )
{

    /* local declarations */
    int     swap,  i;
    float   xmax, ymax;

    /* code: */

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Write IRAP binary map file...");

    if (ntot != mx * my) logger_critical(__LINE__, "Bug in %", __FUNCTION__);

    /* check endianess */
    swap=0;
    if (x_swap_check()==1) swap=1;

    /*
     * Do some computation first, find (pseudo) xmin, ymin, xmax, ymax
     * ---------------------------------------------------------------------------------
     */

    xmax = xori + xinc*(mx-1);
    ymax = yori + yinc*(my-1);

    /*
     * WRITE HEADER
     * ---------------------------------------------------------------------------------
     * Reverse engineering says that the binary header is
     * <32> ID MY XORI XMAX YORI YMAX XINC YINC <32>
     * <16> MX ROT X0ORI Y0ORI<16>
     * <28> 0 0 0 0 0 0 0 <28>
     * ---------------------------------------------------------------------------------
     */

    if (fc == NULL) return EXIT_FAILURE;

    /* first line in header */
    _writeint(fc, 32, swap);
    _writeint(fc, -996, swap);
    _writeint(fc, my, swap);
    _writefloat(fc, xori, swap);
    _writefloat(fc, xmax, swap);
    _writefloat(fc, yori, swap);
    _writefloat(fc, ymax, swap);
    _writefloat(fc, xinc, swap);
    _writefloat(fc, yinc, swap);
    _writeint(fc, 32, swap);

    /* second line in header */
    _writeint(fc, 16, swap);
    _writeint(fc, mx, swap);
    _writefloat(fc, rot, swap);
    _writefloat(fc, xori, swap);
    _writefloat(fc, yori, swap);
    _writeint(fc, 16, swap);

    /* third line in header */
    _writeint(fc, 28, swap);
    for (i = 0; i < 7; i++) _writeint(fc, 0, swap);
    _writeint(fc, 28, swap);

    /* record length */
    int nrec = mx * sizeof(float);

    long ib = 0;
    int j;
    for (j=1;j<=my;j++) {

        _writeint(fc, nrec, swap);

        for (i=1;i<=mx;i++) {
            _writefloat(fc, (float)p_map_v[ib], swap);
            ib++;
        }

        _writeint(fc, nrec, swap);
    }

    return EXIT_SUCCESS;

}
