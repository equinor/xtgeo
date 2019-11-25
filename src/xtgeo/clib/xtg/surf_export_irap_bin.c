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
    int     swap, ier, myint, nrec, i, j, ib;
    float   xmax, ymax, myfloat;


    /* code: */

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Write IRAP binary map file...");

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

    if (fc == NULL) return -1;

    /* first line in header */
    myint=32;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=-996;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=my;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myfloat=xori;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xmax;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yori;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=ymax;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xinc;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yinc;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myint=32;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);

    /* second line in header */
    myint=16;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=mx;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myfloat=rot;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xori;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yori;
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myint=16;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);

    /* third line in header */
    myint=28;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=0;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=28;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);


    /*
     * ---------------------------------------------------------------------------------
     * WRITE DATA
     * These are floats bounded by Fortran records, reading I (x) dir fastest
     * ---------------------------------------------------------------------------------
     */

    /* record length */
    nrec = mx*sizeof(float);

    ib=0;
    for (j=1;j<=my;j++) {

	myint = nrec;
	if (swap) SWAP_INT(myint);
	ier = fwrite(&myint,sizeof(int),1,fc);

	for (i=1;i<=mx;i++) {

            ib = x_ijk2ic(i, j, 1, mx, my, 1, 0); /* conv from C order */

	    myfloat=p_map_v[ib];
	    if (myfloat > UNDEF_MAP_LIMIT) myfloat = UNDEF_MAP_IRAPB;
	    if (swap) SWAP_FLOAT(myfloat);
	    ier=fwrite(&myfloat,sizeof(float),1,fc);
	    ib++;
	}

	myint = nrec;
	if (swap) SWAP_INT(myint);
	ier = fwrite(&myint,sizeof(int),1,fc);
    }

    return EXIT_SUCCESS;

}
