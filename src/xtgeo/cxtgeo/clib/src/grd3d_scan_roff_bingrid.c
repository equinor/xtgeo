/*
 * ############################################################################
 * grd3d_scan_nxyz_roff_bin_grid.c
 * Scanning a Roff binary grid for nx, ny, nz
 * Author: JCR
 * ############################################################################
 */

#include "logger.h"
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
			      char    *filename
			      )


{
    FILE *fc;
    char cname[100];
    int  idum, mynx=0, myny=0, mynz=0, num=0, mybyte, iendian;

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Entering routine %s", __FUNCTION__);

    /*
     *-------------------------------------------------------------------------
     * Check endiness
     *-------------------------------------------------------------------------
     */

    iendian=x_swap_check();
    if (iendian==1) {
	logger_info(__LINE__, "Machine is little endian (linux intel, windows)");
	x_byteorder(1); /* assumed initially */
    }
    else{
	logger_info(__LINE__, "Machine is big endian (many unix)");
	x_byteorder(0); /* assumed initially */
    }

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    fc=fopen(filename,"rb");
    if (fc == NULL) {
	logger_error(__LINE__, "Cannot open file!");
        exit(-1);
    }

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
                mybyte=_grd3d_getintvalue("byteswaptest",fc);
		if (mybyte != 1) {
		    if (iendian==1) x_byteorder(2);
		    if (iendian==0) x_byteorder(3);
		    SWAP_INT(mybyte);
		}
	    }
            /*
             *-----------------------------------------------------------------
             * Getting 'dimensions' values
             *-----------------------------------------------------------------
             */
            if (strcmp(cname, "dimensions") == 0) {
                mynx=_grd3d_getintvalue("nX",fc);
                myny=_grd3d_getintvalue("nY",fc);
                mynz=_grd3d_getintvalue("nZ",fc);
            }
            /*
             *-----------------------------------------------------------------
             * Getting 'subgrids' array
             *-----------------------------------------------------------------
             */
	    if (strcmp(cname, "subgrids") == 0) {
                num=_grd3d_getintvalue("array",fc);
		if (num==-1) num=1;
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
