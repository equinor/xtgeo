/*
****************************************************************************************
 *
 * Import maps on seismic points format (OW specific perhaps?)
 *
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>

#define MAXIX 1000000


/*
****************************************************************************************
 *
 * NAME:
 *    surf_import_ijxyz.c
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
 *     @File_Version: 4
       @Coordinate_Type_is: 3
       @Export_Type_is: 1
       @Number_of_Projects 1
       @Project_Type_Name: , 3,TROLL_WEST,
       @Project_Unit_is: meters , ST_ED50_UTMXXX , PROJECTED_COORDINATE_SYSTEM
       #File_Version____________-> 4
       #Project_Name____________-> TROLL_WEST
       #Horizon_remark_size_____-> 0

       #End_of_Horizon_ASCII_Header_


 *
 *    See there is no system in fastest vs slowest ILINE vs XLINE!
 *
 * ARGUMENTS:
 *    filename       i      File name, character string
 *    mode           i      If 0, then scan only for dimensions
 *    mx            i/o     Map dimension X (I)
 *    my            i/o     Map dimension Y (J)
 *    xori           o      X origin coordinate
 *    yori           o      Y origin coordinate
 *    xinc           o      X increment
 *    yinc           o      Y increment
 *    rot            o      Rotation (degrees, from X axis, anti-clock)
 *    ilines         o      Vector of inline numbers
 *    xlines         o      Vector of xline numbers
 *    p_surf_v       o      1D pointer to map/surface values pointer array
 *    yflip          o      YFLIP indicator (1 for normal, -1 for reverse)
 *    option         i      For later use
 *    debug          i      Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    Code could benefit from a refactorization
 *
 * LICENCE:
 *    See XTGeo license
 ***************************************************************************************
 */


/*
****************************************************************************************
 * Local routine to scan for NX, NY
 ***************************************************************************************
 */

void _scan_dimensions(FILE *fd, int *nx, int *ny)
{
    int inum, nrow, ncol, iline, xline, iok;
    int ilinemin, ilinemax, xlinemin, xlinemax;
    int ispacing, xspacing, ispace, mispace, mxspace;
    int *itmp, *xtmp;
    float rdum, filine, fxline;
    char lbuffer[132] = "";

    itmp = calloc(MAXIX, 4);
    xtmp = calloc(MAXIX, 4);

    nrow = 0;
    ncol = 0;

    for (inum = 0; inum < MAXIX; inum++) itmp[inum] = 0;
    for (inum = 0; inum < MAXIX; inum++) xtmp[inum] = 0;

    ilinemin = 999999999;
    ilinemax = -99999999;
    xlinemin = 999999999;
    xlinemax = -99999999;
    iok = 1;
    while(fgets(lbuffer, 132, (FILE*) fd)) {
        if (strncmp(lbuffer, "\n", 1) == 0) continue;
        lbuffer[strcspn(lbuffer, "\n")] = 0;
        if (strncmp(lbuffer, "#", 1) == 0) continue;
        if (strncmp(lbuffer, "@", 1) == 0) continue;
        if (strncmp(lbuffer, "E", 1) == 0) continue;

        iok = sscanf(lbuffer, "%f %f %f %f %f", &filine, &fxline,
                     &rdum, &rdum, &rdum);

        iline = (int)(filine + 0.01);
        xline = (int)(fxline + 0.01);

        if (iok > 5) logger_error(__LINE__, "Wrong file format for map file?");

        if (iline < ilinemin) ilinemin = iline;
        if (iline > ilinemax) ilinemax = iline;
        if (xline < xlinemin) xlinemin = xline;
        if (xline > xlinemax) xlinemax = xline;
        itmp[iline] = 1;
        xtmp[xline] = 1;
    }

    mispace = (ilinemax - ilinemin)/4;
    mxspace = (xlinemax - xlinemin)/4;

    /* find minimum spacing in INLINES */
    ispacing = 0;
    for (ispace = 1; ispace < mispace; ispace ++) {
        for (inum = ilinemin; inum < ilinemax - mispace; inum++) {
            if (itmp[inum] == 1 && itmp[inum + ispace] == 1) {
                ispacing = ispace;
                break;
            }
        }
        if (ispacing > 0) break;
    }

    /* find minimum spacing in XLINES */
    xspacing = 0;
    for (ispace = 1; ispace < mxspace; ispace ++) {
        for (inum = xlinemin; inum < xlinemax - mxspace; inum++) {
            if (xtmp[inum] == 1 && xtmp[inum + ispace] == 1) {
                xspacing = ispace;
                break;
            }
        }
        if (xspacing > 0) break;
    }

    *nx = (ilinemax - ilinemin)/ispacing + 1;
    *ny = (xlinemax - xlinemin)/xspacing + 1;

    free(itmp);
    free(xtmp);

}


/*
****************************************************************************************
 * Local routine to collect all values form file as 1D buffers
 ***************************************************************************************
 */

long _collect_values(FILE *fd, int *ilinesb, int *xlinesb, double *xbuffer,
                     double *ybuffer, double *zbuffer,
                     int *ilmin, int *ilmax, int *xlmin, int *xlmax)
{
    int ilinemin, ilinemax, xlinemin, xlinemax, iok;
    int iline, xline;
    float filine, fxline;
    long ixnum;
    double xval, yval, zval;
    char lbuffer[132];


    char sbn[24] = "_collect_values";

    ilinemin = 999999999;
    ilinemax = -99999999;
    xlinemin = 999999999;
    xlinemax = -99999999;


    ixnum = 0;
    iok = 1;
    while(fgets(lbuffer, 132, (FILE*) fd)) {
        if (strncmp(lbuffer, "\n", 1) == 0) continue;
        lbuffer[strcspn(lbuffer, "\n")] = 0;
        if (strncmp(lbuffer, "#", 1) == 0) continue;
        if (strncmp(lbuffer, "@", 1) == 0) continue;
        if (strncmp(lbuffer, "E", 1) == 0) continue;

        iok = sscanf(lbuffer, "%f %f %lf %lf %lf", &filine, &fxline,
                     &xval, &yval, &zval);

        iline = (int)(filine + 0.01);
        xline = (int)(fxline + 0.01);

        ilinesb[ixnum] = iline;
        xlinesb[ixnum] = xline;
        xbuffer[ixnum] = xval;
        ybuffer[ixnum] = yval;
        zbuffer[ixnum] = zval;

        if (iline < ilinemin) ilinemin = iline;
        if (iline > ilinemax) ilinemax = iline;
        if (xline < xlinemin) xlinemin = xline;
        if (xline > xlinemax) xlinemax = xline;

        ixnum++;
    }

    *ilmin = ilinemin;
    *ilmax = ilinemax;
    *xlmin = xlinemin;
    *xlmax = xlinemax;

    xtg_speak(sbn, 2, "Range ILINES: %d - %d", ilinemin, ilinemax);
    xtg_speak(sbn, 2, "Range XLINES: %d - %d", xlinemin, xlinemax);

    return ixnum;
}


/*
****************************************************************************************
 * Compute map vectors
 */

int _compute_map_vectors(int ilinemin, int ilinemax, int ncol,
                         int xlinemin, int xlinemax, int nrow,
                         long ixnummax,
                         int *ilinesb, int *xlinesb,
                         double *xbuffer, double *ybuffer, double *zbuffer,
                         double *xcoord, double *ycoord,
                         /* result vectors: */
                         int *ilines, int *xlines, double *p_map_v
                         )
{
    int i, itrue, jtrue;
    long ic, ixnum;
    int ilinestep, xlinestep;


    ilinestep = (ilinemax - ilinemin) / (ncol - 1);
    xlinestep = (xlinemax - xlinemin) / (nrow - 1);

    /* set the *ilines and *xlines results vectors */
    for (i = 0; i < ncol; i++) ilines[i] = ilinemin + i * ilinestep;
    for (i = 0; i < nrow; i++) xlines[i] = xlinemin + i * xlinestep;


    /* setting Z values; UNDEF initially */
    for (ic = 0; ic < ncol * nrow; ic++) p_map_v[ic] = UNDEF;

    for (ixnum = 0; ixnum < ixnummax; ixnum++) {
        itrue = (ilinesb[ixnum] / ilinestep) - ilinemin / ilinestep + 1;  /* base = 1 */
        jtrue = (xlinesb[ixnum] / xlinestep) - xlinemin / xlinestep + 1;  /* base = 1 */

        /* get the C order index */
        ic = x_ijk2ic(itrue, jtrue, 1, ncol, nrow, 1, 0);

        p_map_v[ic] = zbuffer[ixnum];
        xcoord[ic] = xbuffer[ixnum];
        ycoord[ic] = ybuffer[ixnum];
    }
    return EXIT_SUCCESS;
}

/*
****************************************************************************************
 * Compute map properties in XTGeo format, need to deduce from "incomplete
 * data". This version is rather basic, it looks only on one step. It may be
 * more precise to look at long vectors, but that is more complicated to
 * get correct.
 ***************************************************************************************
 */

int _compute_map_props(int ncol, int nrow, double *xcoord, double *ycoord,
                       double *p_map_v,
                       double *xori, double *yori, double *xinc, double *yinc,
                       double *rot, int *yflip)
{

    int okstatus = 0, icol, jrow;
    long icn0, icn1, icn2, ic0 = 0, jc0 = 1;
    double xc0 = 0.0, xc1 = 0.0, xc2 = 0.0, yc0 = 0.0, yc1 = 0.0, yc2 = 0.0;
    double a1rad, a2rad, roty, sinusrad;

    okstatus = 0;

    for (icol = 1; icol < ncol; icol++) {
        for (jrow = 1; jrow < nrow; jrow++) {

            icn0 = x_ijk2ic(icol, jrow, 1, ncol, nrow, 1, 0);
            icn1 = x_ijk2ic(icol + 1, jrow, 1, ncol, nrow, 1, 0);
            icn2 = x_ijk2ic(icol, jrow + 1, 1, ncol, nrow, 1, 0);

            if (p_map_v[icn0] < UNDEF_LIMIT && p_map_v[icn1] < UNDEF_LIMIT &&
                p_map_v[icn2] < UNDEF_LIMIT) {

                xc0 = xcoord[icn0];
                xc1 = xcoord[icn1];
                xc2 = xcoord[icn2];

                yc0 = ycoord[icn0];
                yc1 = ycoord[icn1];
                yc2 = ycoord[icn2];

                ic0 = icol;
                jc0 = jrow;

                okstatus = 1;
                break;
            }
            if (okstatus == 1) break;
        }
        if (okstatus == 1) break;
    }

    if (okstatus == 0) {
        logger_error(__LINE__, "Could not find info to deduce map properties");
        return -9;
    }


    x_vector_info2(xc0, xc1, yc0, yc1, xinc, &a1rad, rot, 1, 0);
    x_vector_info2(xc0, xc2, yc0, yc2, yinc, &a2rad, &roty, 1, 0);

    /* compute yflip: sin (y-x) = sin(y)*cos(x) - sin(x)*cos(y) */
    *yflip = 1;
    sinusrad = sin(a2rad) * cos(a1rad) - sin(a1rad) * cos (a2rad);
    if (sinusrad < 0) *yflip = -1;

    /* find origin */
    surf_xyori_from_ij(ic0, jc0, xc0, yc0, xori, *xinc, yori, *yinc,
                       ncol, nrow, *yflip, *rot, 0, 0);


    return EXIT_SUCCESS;
}



int surf_import_ijxyz (
                       FILE *fd,
                       int mode,
                       int *nx,
                       int *ny,
                       long *ndef,
                       double *xori,
                       double *yori,
                       double *xinc,
                       double *yinc,
                       double *rot,
                       int *ilines,
                       long ncol,
                       int *xlines,
                       long nrow,
                       double *p_map_v,
                       long nmap,
                       int *yflip,
                       int option
                       )
{

    /* locals*/
    int iok = 0;
    int ilinemin, ilinemax, xlinemin, xlinemax;

    double *xbuffer, *ybuffer, *zbuffer, *xcoord, *ycoord;
    int *ilinesb, *xlinesb;
    int nncol, nnrow;

    logger_init(__FILE__, __FUNCTION__);

    /* read header */
    logger_info(__LINE__, "Entering routine %s", __FUNCTION__);

    fseek(fd, 0, SEEK_SET);

    /* =================================================================================
     * scan mode; to determine dimensions */
    if (mode == 0) {
        _scan_dimensions(fd, nx, ny);

        return EXIT_SUCCESS;
    }

    /* =================================================================================
     * Read mode; now dimensions shall be known */

    nncol = (int)ncol;
    nnrow = (int)nrow;

    *nx = nncol;
    *ny = nnrow;

    ilinesb = calloc(ncol * nrow + 10, sizeof(int));
    xlinesb = calloc(ncol * nrow + 10, sizeof(int));
    xbuffer = calloc(ncol * nrow + 10, sizeof(double));
    ybuffer = calloc(ncol * nrow + 10, sizeof(double));
    zbuffer = calloc(ncol * nrow + 10, sizeof(double));
    xcoord = calloc(ncol * nrow + 10, sizeof(double));
    ycoord = calloc(ncol * nrow + 10, sizeof(double));


    *ndef = _collect_values(fd, ilinesb, xlinesb, xbuffer, ybuffer, zbuffer,
                            &ilinemin, &ilinemax, &xlinemin, &xlinemax);

    iok = _compute_map_vectors(ilinemin, ilinemax, nncol,
                               xlinemin, xlinemax, nnrow,
                               *ndef,
                               ilinesb, xlinesb,
                               xbuffer, ybuffer, zbuffer,
                               xcoord, ycoord,
                               ilines, xlines,
                               p_map_v);

    iok = _compute_map_props(nncol, nnrow, xcoord, ycoord, p_map_v,
                             xori, yori, xinc, yinc, rot, yflip);

    if (iok != 0) logger_error(__LINE__, "Error, cannot compute map props");

    free(ilinesb);
    free(xlinesb);
    free(xbuffer);
    free(ybuffer);
    free(zbuffer);
    free(xcoord);
    free(ycoord);

    return EXIT_SUCCESS;

}
