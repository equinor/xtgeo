/*
 *******************************************************************************
 *
 * Export Irap binary map (with rotation)
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    surf_export_irap_bin.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export a map on Irap binary format.
 *
 * ARGUMENTS:
 *    filename       i     File name, character string
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
 *******************************************************************************
 */
int surf_export_irap_bin(
			 char   *filename,
			 int    mx,
			 int    my,
			 double xori,
			 double yori,
			 double xinc,
			 double yinc,
			 double rot,
			 double *p_map_v,
                         long   ntot,
			 int    option,
			 int    debug
			 )
{

    /* local declarations */
    int     swap, ier, myint, nrec, i, j, ib;
    float   xmax, ymax, myfloat;

    char    s[24]="surf_export_irap_bin";

    FILE    *fc;

    /* code: */

    xtgverbose(debug);
    xtg_speak(s,1,"Write IRAP binary map file ...",s);

    xtg_speak(s,2,"Entering %s",s);

    /* check endianess */
    swap=0;
    if (x_swap_check()==1) swap=1;

    /*
     * Do some computation first, find (pseudo) xmin, ymin, xmax, ymax
     * -------------------------------------------------------------------------
     */
    xmax = xori + xinc*(mx-1);
    ymax = yori + yinc*(my-1);

    /*
     * WRITE HEADER
     * -------------------------------------------------------------------------
     * Reverse engineering says that the binary header is
     * <32> ID MY XORI XMAX YORI YMAX XINC YINC <32>
     * <16> MX ROT X0ORI Y0ORI<16>
     * <28> 0 0 0 0 0 0 0 <28>
     * -------------------------------------------------------------------------
     */

    fc = fopen(filename,"wb");

    if (fc == NULL) {
        xtg_warn(s, 0, "Some thing is wrong with requested filename <%s>",
                 filename);
        xtg_error(s, "Could be: Non existing folder, wrong permissions ? ..."
                  " anyway: STOP!", s);
    }

    /* first line in header */
    myint=32;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=-996;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=my;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myfloat=xori;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xmax;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yori;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=ymax;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xinc;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yinc;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myint=32;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);

    /* second line in header */
    myint=16;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myint=mx;
    xtg_speak(s,2,"Write %d",myint);
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,sizeof(int),1,fc);
    myfloat=rot;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=xori;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myfloat=yori;
    xtg_speak(s,2,"Write %f",myfloat);
    if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,sizeof(float),1,fc);
    myint=16;
    xtg_speak(s,2,"Write %d",myint);
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
     * -------------------------------------------------------------------------
     * WRITE DATA
     * These are floats bounded by Fortran records, reading I (x) dir fastest
     * -------------------------------------------------------------------------
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


    fclose(fc);
    return EXIT_SUCCESS;

}
