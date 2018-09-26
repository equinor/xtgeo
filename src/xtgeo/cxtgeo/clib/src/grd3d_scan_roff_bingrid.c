/*
 * ############################################################################
 * grd3d_scan_nxyz_roff_bin_grid.c
 * Scanning a Roff binary grid for nx, ny, nz
 * Author: JCR
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Scanning for size... (binary files)
 * ############################################################################
 */

void grd3d_scan_roff_bingrid (
			      int     *nx,
			      int     *ny,
			      int     *nz,
			      int     *nsubs,
			      char    *filename,
			      int     debug
			      )


{
    FILE *fc;
    char cname[100];
    int  idum, mynx=0, myny=0, mynz=0, num=0, mybyte, iendian;
    char sub[24]="grd3d_scan_roff_bingrid";

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering routine ...");
    /*
     *-------------------------------------------------------------------------
     * Check endiness
     *-------------------------------------------------------------------------
     */

    iendian=x_swap_check();
    if (iendian==1) {
	xtg_speak(sub,2,"Machine is little endian (linux intel, windows)");
	x_byteorder(1); /* assumed initially */
    }
    else{
	xtg_speak(sub,2,"Machine is big endian (many unix)");
	x_byteorder(0); /* assumed initially */
    }

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Opening binary ROFF file...");
    fc=fopen(filename,"rb");
    if (fc == NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");

    /*
     *=========================================================================
     * Loop file...
     *=========================================================================
     */

    _grd3d_roffbinstring(cname, fc);
    for (idum=1;idum<999;idum++) {
        _grd3d_roffbinstring(cname, fc);

        if (strcmp(cname, "tag") == 0) {

            _grd3d_roffbinstring(cname, fc);

            /*
             *-----------------------------------------------------------------
             * Getting 'filedata' values
             *-----------------------------------------------------------------
             */
            if (strcmp(cname, "filedata") == 0) {
                xtg_speak(sub,3,"Tag filedata was found");
                mybyte=_grd3d_getintvalue("byteswaptest",fc);
                xtg_speak(sub,2,"bytewaptest is %d", mybyte);
		if (mybyte != 1) {
		    if (iendian==1) x_byteorder(2);
		    if (iendian==0) x_byteorder(3);
		    SWAP_INT(mybyte);
		    xtg_speak(sub,1,"Roff file import need swapping");
		    xtg_speak(sub,2,"Bytewaptest is now %d", mybyte);
		    xtg_speak(sub,2,"Byte order flag is now %d", x_byteorder(-1));
		}
	    }
            /*
             *-----------------------------------------------------------------
             * Getting 'dimensions' values
             *-----------------------------------------------------------------
             */
            if (strcmp(cname, "dimensions") == 0) {
                xtg_speak(sub,3,"Tag dimensions was found");
                mynx=_grd3d_getintvalue("nX",fc);
                xtg_speak(sub,2,"nX is %d", mynx);
                myny=_grd3d_getintvalue("nY",fc);
                xtg_speak(sub,2,"nY is %d", myny);
                mynz=_grd3d_getintvalue("nZ",fc);
                xtg_speak(sub,2,"nZ is %d", mynz);
            }
            /*
             *-----------------------------------------------------------------
             * Getting 'subgrids' array
             *-----------------------------------------------------------------
             */
	    if (strcmp(cname, "subgrids") == 0) {
                xtg_speak(sub,3,"Tag subgrids was found");
                num=_grd3d_getintvalue("array",fc);
		if (num==-1) num=1;
                xtg_speak(sub,2,"Number of subgrids are are %d", num);
                break;
            }

        }

    }

    *nx=mynx;
    *ny=myny;
    *nz=mynz;
    *nsubs=num;

    fclose(fc);
}
