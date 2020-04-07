/*
****************************************************************************************
*
* Export Irap binary map (with rotation) test version
*
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

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

void
_writeint2(FILE *fc, int ival, int swap)
{
    if (swap)
        SWAP_INT(ival);

    if (fwrite(&ival, sizeof(int), 1, fc) != 1) {
        logger_critical(LI, FI, FU, "Cannot write int to file! <%s>", FU);
    }
}

void
_writefloat2(FILE *fc, float fval, int swap)
{
    if (swap)
        SWAP_FLOAT(fval);

    if (fwrite(&fval, sizeof(float), 1, fc) != 1) {
        logger_critical(LI, FI, FU, "Cannot write float to file! <%s>", FU);
    }
}

int
surf_export_irap_bin_test(FILE *fc,
                          int mx,
                          int my,
                          double xori,
                          double yori,
                          double xinc,
                          double yinc,
                          double rot,
                          double *rsurfv,
                          long nsurf,
                          mbool *maskv,
                          long nmask)
{

    /* local declarations */
    int swap, i;
    float xmax, ymax;

    /* code: */

    logger_info(LI, FI, FU, "Write IRAP binary map file...");

    if (nsurf != mx * my)
        logger_critical(LI, FI, FU, "Bug in %", FU);

    /* check endianess */
    swap = 0;
    if (x_swap_check() == 1)
        swap = 1;

    /*
     * Do some computation first, find (pseudo) xmin, ymin, xmax, ymax
     * ---------------------------------------------------------------------------------
     */

    xmax = xori + xinc * (mx - 1);
    ymax = yori + yinc * (my - 1);

    /*
     * WRITE HEADER
     * ---------------------------------------------------------------------------------
     * Reverse engineering says that the binary header is
     * <32> ID MY XORI XMAX YORI YMAX XINC YINC <32>
     * <16> MX ROT X0ORI Y0ORI<16>
     * <28> 0 0 0 0 0 0 0 <28>
     * ---------------------------------------------------------------------------------
     */

    if (fc == NULL)
        return EXIT_FAILURE;

    /* first line in header */
    _writeint2(fc, 32, swap);
    _writeint2(fc, -996, swap);
    _writeint2(fc, my, swap);
    _writefloat2(fc, xori, swap);
    _writefloat2(fc, xmax, swap);
    _writefloat2(fc, yori, swap);
    _writefloat2(fc, ymax, swap);
    _writefloat2(fc, xinc, swap);
    _writefloat2(fc, yinc, swap);
    _writeint2(fc, 32, swap);

    /* second line in header */
    _writeint2(fc, 16, swap);
    _writeint2(fc, mx, swap);
    _writefloat2(fc, rot, swap);
    _writefloat2(fc, xori, swap);
    _writefloat2(fc, yori, swap);
    _writeint2(fc, 16, swap);

    /* third line in header */
    _writeint2(fc, 28, swap);
    for (i = 0; i < 7; i++)
        _writeint2(fc, 0, swap);
    _writeint2(fc, 28, swap);

    /* record length */
    int nrec = mx * sizeof(float);

    int j;
    for (j = 1; j <= my; j++) {

        _writeint2(fc, nrec, swap);

        for (i = 1; i <= mx; i++) {
            long ib = x_ijk2ic(i, j, 1, mx, my, 1, 0);

            if (maskv[ib] == 0) {
                // printf("VALUE %f\n", rsurfv[ib]);
                _writefloat2(fc, (float)rsurfv[ib], swap);
            } else {
                _writefloat2(fc, UNDEF_MAP_IRAPB, swap);
            }
        }

        _writeint2(fc, nrec, swap);
    }

    return EXIT_SUCCESS;
}
