/*
 ******************************************************************************
 *
 * Import Irap binary map (with rotation)
 *
 ******************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    surf_import_irap_bin.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Imports a surface map on Irap binary format.
 *
 * ARGUMENTS:
 *    fhandle        i     Filehandle (steered from caller)
 *    mode           i     0 = scan mode to find mx, my; 1 = normal mode
 *    p_mx           o     Map dimension X (I) pointer
 *    p_my           o     Map dimension Y (J) pointer
 *    p_ndef         o     Number of defined nodes, pointer
 *    p_xori         o     X origin coordinate pointer
 *    p_yori         o     Y origin coordinate pointer
 *    p_xinc         o     X increment pointer
 *    p_yinc         o     Y increment pointer
 *    p_rot          o     Rotation (rad?) pointer
 *    p_surf_v       o     1D pointer to map/surface values pointer array
 *    nsurf          i     No. of map nodes (for allocation from Python/SWIG)
 *    option         i     Options flag for later usage
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *    Result pointers are updated
 *
 * TODO/ISSUES/BUGS/NOTES:
 *    Issue: The surf_* routines in XTGeo will include rotation, and origins
 *           (not xmin etc ) and steps are used to define the map extent.
 *    Notes: Irap format rotation is X axis anticlockwise rotation angle (deg),
 *           and this is the same convention as XTGeo. But secure that
 *           angle is within 0:360 deg.
 *
 *           Result array is converted to C order (this file format is F order).
 *
 * LICENCE:
 *    See XTGeo licence
 ******************************************************************************
 */
/* Local functions: */

#define ERROR -999999

int _intread(FILE *fc, int swap, int trg, char info[50]) {
    /* read an INT item in the IRAP binary header */
    int ier, myint;

    ier = fread(&myint, sizeof(int), 1, fc);
    if (ier != 1) {
        logger_critical(__LINE__, "Error in reading INT in Irap binary header");
        return ERROR;
    }
    if (swap) SWAP_INT(myint);

    /* test againts target if target > 0 */
    if (trg > 0) {
        if (myint != trg) {
            logger_critical(__LINE__, "Error in reading INT in Irap binary header");
            return ERROR;
        }
    }
    return myint;
}


double _floatread(FILE *fc, int swap, float trg, char info[50]) {
    /* read a FLOAT item in the IRAP binary header, return as DOUBLE */
    int ier;
    float myfloat;

    ier = fread(&myfloat, sizeof(float), 1, fc);
    if (ier != 1) {
        logger_critical(__LINE__, "Error in reading FLOAT in Irap binary header");
        return ERROR;
    }

    if (swap) SWAP_FLOAT(myfloat);

    /* test againts target if target > 0 */
    if (trg > 0.0) {
        if (myfloat != trg)  {
            logger_critical(__LINE__, "Error in reading FLOAT in Irap binary header");
            return ERROR;
        }
    }

    return (double)myfloat;
}


int surf_import_irap_bin(
			 FILE *fc,
			 int mode,
			 int *p_mx,
			 int *p_my,
			 long *p_ndef,
			 double *p_xori,
			 double *p_yori,
			 double *p_xinc,
			 double *p_yinc,
			 double *p_rot,
			 double *p_map_v,
			 long nsurf,
			 int option
			 )
{

    /* local declarations */
    int swap, ier, myint, nvv, i, j, k, mstart, mstop, nx, ny, idum;
    int idflag, ireclen;
    long nn, ntmp, ib, ic, mx, my;
    float myfloat;

    double xori, yori, xmax, ymax, xinc, yinc, rot, x0ori, y0ori, dval;

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Read IRAP binary map file: %s", __FUNCTION__);

    if (mode==0) logger_info(__LINE__, "Scan mode!");
    if (mode==1) logger_info(__LINE__, "Values mode!");

    logger_info(__LINE__, "FSEEK...! in %d", fileno(fc));
    fseek(fc, 0, SEEK_SET);
    logger_info(__LINE__, "FSEEK done!");

    /*
     * READ HEADER
     * This is Fortran format, hence there will be an integer
     * in both ends, defining the record length in bytes. The Irap
     * ASCII header looks like this:
     * ------------------------------------------------------------------------
     *  -996    53     25.000000     25.000000
     * 464308.406250   465733.406250  7337233.500000  7338533.500000
     * 58      -70.000008   464308.406250  7337233.500000
     * 0     0     0     0     0     0     0
     * ---- i.e. ---------------------------------
     * IDFLAG   NY   XINC  YINC
     * XORI  XMAX  YORI  YMAX
     * NX  ROTATION  X0ORI  Y0ORI
     * 0     0     0     0     0     0     0
     * ------------------------------------------------------------------------
     * However!!, reverse engineering says that the BINARY header is
     * <32> IDFLAG NY XORI XMAX YORI YMAX XINC YINC <32>
     * <16> NX ROT X0ORI Y0ORI<16>
     * <28> 0 0 0 0 0 0 0 <28>
     * ---data---
     * Note, XMAX and YMAX are based on unroted distances and are
     * not used directly? =>
     * XINC = (XMAX-XORI)/(NX-1) etc
     * X0ORI/Y0ORI seems to be rotation origin? Set them equal to XORI/YORI
     * ------------------------------------------------------------------------
     * Last record: Not sure what these mean, treat them as dummy
     */

    /* check endianess */
    swap = 0;
    if (x_swap_check() == 1) swap=1;

    /* record 1 */
    ireclen = _intread(fc, swap, 32, "Record start (1)");
    idflag = _intread(fc, swap, 0, "ID flag for Irap map");
    ny = _intread(fc, swap, 0.0, "NY");
    xori = _floatread(fc, swap, 0.0, "XORI");
    xmax = _floatread(fc, swap, 0.0, "XMAX (not used by RMS)");
    yori = _floatread(fc, swap, 0.0, "YORI");
    ymax = _floatread(fc, swap, 0.0, "YMAX (not used by RMS)");
    xinc = _floatread(fc, swap, 0.0, "XINC");
    yinc = _floatread(fc, swap, 0.0, "YINC");
    ireclen = _intread(fc, swap, 32, "Record end (1)");

    /* record 2 */
    ireclen = _intread(fc, swap, 16, "Record start (2)");
    nx = _intread(fc, swap, 0, "NX");
    rot = _floatread(fc, swap, 0.0, "Rotation");
    x0ori = _floatread(fc, swap, 0.0, "Rotation origin X (not used)");
    y0ori = _floatread(fc, swap, 0.0, "Rotation origin Y (not used)");
    ireclen = _intread(fc, swap, 16, "Record end (2)");

    /* record 3 */
    ireclen = _intread(fc, swap, 28, "Record start (3)");
    for (i = 0; i < 7; i++) {
        idum = _intread(fc, swap, 0, "INT FLAG (not used...)");
    }
    ireclen = _intread(fc, swap, 28, "Record end (3)");

    *p_mx = nx;
    *p_my = ny;
    *p_xori = xori;
    *p_yori = yori;
    *p_xinc = xinc;
    *p_yinc = yinc;

    if (rot < 0.0) rot = rot + 360.0;
    *p_rot = rot;

    /* if scan mode only: */
    if (mode==0) {
        logger_info(__LINE__, "Scan mode done");
	return EXIT_SUCCESS;
    }

    mx = (long)nx;
    my = (long)ny;

    logger_info(__LINE__, "NX and NY %d %d", nx, ny);

    /*
     * READ DATA
     * These are floats bounded by Fortran records, reading I (x) dir fastest
     */

    ier = 1;
    nn = 0;
    ntmp = 0;
    logger_info(__LINE__, "Read Irap map values...");

    while (ier == 1) {
	/* start of block integer */
	ier = fread(&myint, sizeof(int), 1, fc); if (swap) SWAP_INT(myint);
	if (ier == 1 && myint > 0) {
	    mstart = myint;

	    /* read data */
	    nvv = mstart / sizeof(float);
	    for (ib = 0; ib < nvv; ib++) {
		ier = fread(&myfloat, sizeof(int), 1, fc);
		if (swap) SWAP_FLOAT(myfloat);

		if (myfloat > UNDEF_MAP_IRAPB_LIMIT) {
		    dval = UNDEF_MAP;
		}
		else{
		    dval = myfloat;
		    ntmp++;
		}

                /* convert to C order */
                x_ib2ijk(nn, &i, &j, &k, mx, my, 1, 0);
                ic = x_ijk2ic(i, j, 1, mx, my, 1, 0);

                p_map_v[ic] = dval;

		nn++;
	    }

	    /* end of block integer */
	    if (fread(&myint, sizeof(int), 1, fc) != 1) {
                logger_error(__LINE__, "Error in reading end of block integer");
                return EXIT_FAILURE;
            }

            if (swap) SWAP_INT(myint);
	    mstop = myint;
	    if (mstart != mstop) {
                logger_error(__LINE__, "Error en reading irap file (mstart %d mstop %d)",
                             mstart, mstop);
                return EXIT_FAILURE;
	    }
	}
	else{
	    break;
	}
    }

    *p_ndef = ntmp;
    if (nn != mx * my) {
	logger_error(__LINE__, "Error, number of map nodes read not equal to MX*MY");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
