/*
 ******************************************************************************
 *
 * Import Irap binary map (with rotation)
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/* local functions */
int _intread(FILE *fc, int swap, int trg, char info[50], int debug);
double _floatread(FILE *fc, int swap, float trg, char info[50], int debug);

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
 *    filename       i     File name, character string
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
 *    debug          i     Debug level
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
 *           Result array is converted to C order (this format is F order).
 *
 * LICENCE:
 *    See XTGeo licence
 ******************************************************************************
 */
int surf_import_irap_bin(
			 char *filename,
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
			 int option,
			 int debug
			 )
{

    /* local declarations */
    int swap, ier, myint, nvv, i, j, k, mstart, mstop, nx, ny, idum;
    int idflag, ireclen;
    long nn, ntmp, ib, ic, mx, my;
    float myfloat;

    double xori, yori, xmax, ymax, xinc, yinc, rot, x0ori, y0ori, dval;

    char s[24] = "surf_import_irap_bin";

    FILE *fc;

    xtgverbose(debug);
    xtg_speak(s, 1,"Read IRAP binary map file: %s", filename);

    xtg_speak(s, 2, "Entering %s",s);

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

    fc = fopen(filename, "r");

    /* check endianess */
    swap = 0;
    if (x_swap_check() == 1) swap=1;

    /* record 1 */
    ireclen = _intread(fc, swap, 32, "Record start (1)", debug);
    idflag = _intread(fc, swap, 0, "ID flag for Irap map", debug);
    ny = _intread(fc, swap, 0.0, "NY", debug);
    xori = _floatread(fc, swap, 0.0, "XORI", debug);
    xmax = _floatread(fc, swap, 0.0, "XMAX (not used by RMS)", debug);
    yori = _floatread(fc, swap, 0.0, "YORI", debug);
    ymax = _floatread(fc, swap, 0.0, "YMAX (not used by RMS)", debug);
    xinc = _floatread(fc, swap, 0.0, "XINC", debug);
    yinc = _floatread(fc, swap, 0.0, "YINC", debug);
    ireclen = _intread(fc, swap, 32, "Record end (1)", debug);

    /* record 2 */
    ireclen = _intread(fc, swap, 16, "Record start (2)", debug);
    nx = _intread(fc, swap, 0, "NX", debug);
    rot = _floatread(fc, swap, 0.0, "Rotation", debug);
    x0ori = _floatread(fc, swap, 0.0, "Rotation origin X (not used)", debug);
    y0ori = _floatread(fc, swap, 0.0, "Rotation origin Y (not used)", debug);
    ireclen = _intread(fc, swap, 16, "Record end (2)", debug);

    /* record 3 */
    ireclen = _intread(fc, swap, 28, "Record start (3)", debug);
    for (i = 0; i < 7; i++) {
        idum = _intread(fc, swap, 0, "INT FLAG (not used...)", debug);
    }
    ireclen = _intread(fc, swap, 28, "Record end (3)", debug);

    *p_mx = nx;
    *p_my = ny;

    /* if scan mode only: */
    if (mode==0) {
	fclose(fc);
        xtg_speak(s, 2, "Scan mode!");
	return EXIT_SUCCESS;
    }

    mx = (long)nx;
    my = (long)ny;

    /*
     * READ DATA
     * These are floats bounded by Fortran records, reading I (x) dir fastest
     */

    ier = 1;
    nn = 0;
    ntmp = 0;
    xtg_speak(s, 3, "Read Irap map values...");

    while (ier == 1) {
	/* start of block integer */
	ier = fread(&myint, sizeof(int), 1, fc); if (swap) SWAP_INT(myint);
	if (ier == 1 && myint > 0) {
	    //xtg_speak(s,2,"RECORD DATA START is %d", myint);
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
	    ier = fread(&myint, sizeof(int), 1, fc); if (swap) SWAP_INT(myint);
	    //xtg_speak(s,2,"RECORD DATA STOP is %d", myint);
	    mstop = myint;
	    if (mstart != mstop) {
		xtg_error(s,"Error en reading irap file");
	    }
	}
	else{
	    break;
	}
    }

    *p_ndef = ntmp;
    if (nn != mx * my) {
	xtg_error(s, "Error, number of map nodes read not equal to MX*MY");
    }
    else{
	xtg_speak(s, 2, "Number of map nodes read are: %d", nn);
	xtg_speak(s, 2, "Number of defind map nodes read are: %d", ntmp);
    }
    xtg_speak(s, 3, "Read data map ... OK");

    fclose(fc);

    *p_xori = xori;
    *p_yori = yori;
    *p_xinc = xinc;
    *p_yinc = yinc;

    if (rot < 0.0) rot = rot + 360.0;
    *p_rot = rot;


    return EXIT_SUCCESS;

}


/* Local functions: */

int _intread(FILE *fc, int swap, int trg, char info[50], int debug) {
    /* read an INT item in the IRAP binary header */
    int ier, myint;
    char s[24] = "_intread";
    xtgverbose(debug);

    ier = fread(&myint, sizeof(int), 1, fc);
    if (ier != 1) xtg_error(s, "Error in reading Irap binary map header (32)");

    if (swap) SWAP_INT(myint);

    /* test againts target if target > 0 */
    if (trg > 0) {
        if (myint != trg) {
            xtg_error(s, "Error in reading Irap binary map header (33)");
        }
    }
    xtg_speak(s, 2, "Reading: %s; value is %d", info, myint);

    return myint;
}


double _floatread(FILE *fc, int swap, float trg, char info[50], int debug) {
    /* read a FLOAT item in the IRAP binary header, return as DOUBLE */
    int ier;
    float myfloat;
    char s[24] = "_floatread";

    xtgverbose(debug);

    ier = fread(&myfloat, sizeof(float), 1, fc);
    if (ier != 1) xtg_error(s, "Error in reading Irap binary map header "
                            "(IER != 1) (%s)", info);

    if (swap) SWAP_FLOAT(myfloat);

    /* test againts target if target > 0 */
    if (trg > 0.0) {
        if (myfloat != trg) {
            xtg_error(s, "Error in reading Irap binary map header "
                      "(target val) (%s)", info);
        }
    }
    xtg_speak(s, 2, "Reading: %s; value is %f", info, myfloat);

    return (double)myfloat;
}
