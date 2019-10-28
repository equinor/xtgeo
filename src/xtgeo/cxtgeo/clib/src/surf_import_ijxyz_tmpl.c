/*
****************************************************************************************
 *
 * Import map value on seismic points format (OW specific perhaps?) on the
 * premise that the map topology (through INLINE and XLINE is known); hence
 * map values are based on INLINE/XLINE match.
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
 *    surf_import_ijxyz_tmpl.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Import a map on DSG I J XYZ format. This format is a coordinate value
 *    per cell. Undefined cells are simply not mentioned. The routine is
 *    a bit complicated, and numerous "options" may occur. Example:
 *      1347   2020    532398.6557089972     6750702.972141342     11712.976
 *      1340   2049    532758.1090287864     6750842.3494704       7387.8145
 *      1340   2048    532748.6891376793     6750834.13270089      7485.6885
 *      1347   2029    532483.4347289609     6750776.923066932     7710.251
 *      1347   2028    532474.0148378538     6750768.706297422     7850.605
 *      1347   2027    532464.5949467467     6750760.489527912     7640.044
 *      1347   2026    532455.1750556396     6750752.272758402     7700.624
 *      1347   2025    532445.7551645326     6750744.055988892     8482.951
 *
 *    In some cases headersand footers are present. Assume footer when
 *    line is like this, and I J may be float, not INT
 *      @File_Version: 4
 *      @Coordinate_Type_is: 3
 *      @Export_Type_is: 1
 *      @Number_of_Projects 1
 *      @Project_Type_Name: , 3,TROLL_WEST,
 *      @Project_Unit_is: meters , ST_ED50_UTM31N_P23031_T1133 , PROJECTED_..
 *      #File_Version____________-> 4
 *      #Project_Name____________-> TROLL_WEST
 *      #Horizon_remark_size_____-> 0
 *
 *      #End_of_Horizon_ASCII_Header_
 *
 *    See there is no system in fastest vs slowest ILINE vs XLINE!
 *
 * ARGUMENTS:
 *    filename       i      File name, character string
 *    ilines         i      Vector of inline numbers, as template
 *    xlines         i      Vector of xline numbers, as template
 *    p_surf_v       o      1D pointer to map/surface values pointer array
 *    option         i      For later use
 *    debug          i      Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *    -1: inline/xline in file is outside input ilines, xlines
 *
 * TODO/ISSUES/BUGS:
 *    Code could benefit from a refactorization
 *
 * LICENCE:
 *    See XTGeo license
 ***************************************************************************************
 */


int surf_import_ijxyz_tmpl(
                           FILE *fd,
                           int *ilines,
                           long ncol,
                           int *xlines,
                           long nrow,
                           double *p_map_v,
                           long nmap,
                           int option
                           )
{

    /* locals*/

    int iok, iline, xline, nncol, nnrow, found, ili, xli;
    long ind, ic;
    float rdum, zval, filine, fxline;
    char lbuffer[132];

    nncol = (int)ncol;
    nnrow = (int)nrow;

    for (ind = 0; ind < nnrow*nncol; ind++) p_map_v[ind] = UNDEF;

    while(fgets(lbuffer, 132, (FILE*) fd)) {
        if (strncmp(lbuffer, "\n", 1) == 0) continue;
        lbuffer[strcspn(lbuffer, "\n")] = 0;

        if (strncmp(lbuffer, "#", 1) == 0) continue;
        if (strncmp(lbuffer, "@", 1) == 0) continue;
        if (strncmp(lbuffer, "E", 1) == 0) continue;

        iok = sscanf(lbuffer, "%f %f %f %f %f", &filine, &fxline,
                     &rdum, &rdum, &zval);

        iline = (int)(filine + 0.01);
        xline = (int)(fxline + 0.01);

        /* some sanity tests first */
        if (iline < ilines[0] || iline > ilines[nncol - 1] ||
            xline < xlines[0] || xline > xlines[nnrow - 1]) {
            logger_error(__LINE__, "ILINE or XLINE in file outside template ranges");
            return -1;
        }

        found = 0;
        for (ili = 0; ili < nncol; ili++) {
            for (xli = 0; xli < nnrow; xli++) {
                if (iline == ilines[ili] && xline == xlines[xli]) {
                    ic = x_ijk2ic(ili + 1, xli + 1, 1, nncol, nnrow, 1, 0);
                    p_map_v[ic] = zval;
                    found = 1;
                    break;
                }
            }
            if (found == 1) break;
        }

    }

    return EXIT_SUCCESS;
}
