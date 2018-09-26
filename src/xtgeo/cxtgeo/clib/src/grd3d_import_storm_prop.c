/*
 * ############################################################################
 * grd3d_import_storm_prop.c
 * Basic routines to handle import of Storm 3D grids
 * ############################################################################
 * $Id: grd3d_import_storm_prop.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_import_storm_prop.c,v $
 *
 * $Log: grd3d_import_storm_prop.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


/*
 * ****************************************************************************
 *                       GRD3D_IMPORT_STORM_PROP
 * ****************************************************************************
 * The header of the binary FACIES files may look like:
 * ----------------------------------------------------------------------------
 * storm_facies_binary
 * 8  zone008/facies/f01s03.par  -999
 *
 * 2
 * Tilje_4A_D  T_4A_DUMMY
 *
 * 2
 * 0 1
 *
 * 406446.000000  6650.000000  7126970.000000  8800.000000  zone008/structure/s
 * 03_t_4a_d.grd zone009/structure/s03_t_3a_d.grd 0.000000 0.000000
 * 33.093400  -37.000000
 *
 * 133  176  10
 * <binary integer numbers>
 * ----------------------------------------------------------------------------
 * The header of the petrophysical files may look like:
 * ----------------------------------------------------------------------------
 * storm_petro_binary
 * 7  zone007/petro/p03f01s03.par  -999.000000
 * phi
 * 406446.000000  6650.000000  7126970.000000  8800.000000  zone007/stru
 * cture/s03_t_4b_d.grd zone008/structure/s03_t_4a_d.grd 0.000000 0.000000
 * 62.178700  -37.000000
 *
 * 133  176  20
 * <binary double precision data...>
 * ----------------------------------------------------------------------------
 * m_action           1: Make full rectangular grid only
 *                    2: A.a., but make inactive cells according to maps
 *                    3: Erode top
 *                    4: Erode bottom
 *                    5: Erode both
 *                    6: Do "compaction"
 * nx, ny, nz         Grid dimension
 * nzyx               Assumed size of grid
 * num_active         Number of active cells
 * grd3d_v            Array holding cornerpoint grid
 * iactnum_v          ACTNUM array
 * iocn_v             IOCN array: ia=iocn(ib)
 * filename           file to read
 * debug              Debug (verbose) flag
 * ----------------------------------------------------------------------------
 */


int grd3d_import_storm_prop (
			      int    M_action,
			      int    nx,
			      int    ny,
			      int    nz,
			      int    inxyz,
			      int    *int_v,
			      double *double_v,
			      char   *filename,
			      int    debug
			      )
{

    int                 i, nnn, nzone, nfac1, nfac2, ndum, nnx, nny, nnz;
    int                 ib, nxyz, istype=1, numxxx, ier;

    float               xstart, xlen, ystart, ylen, zmax, rotation;
    char                stype[25], stormparfile[30], sdum[30], prop[30];
    char                s_top[132], s_bot[132], s_tero[132], s_bero[132];
    FILE                *fc;
    char                s[24] = "grd3d_imp.._storm_prop";
    /*
     * ========================================================================
     * INITIALLY, READ ASCII HEADER (CONTAINS ALL GRID-INFO!)
     * ========================================================================
     */

    xtgverbose(debug);

    xtg_speak(s, 1,"Opening %s",filename);
    fc=fopen(filename,"rb");
    xtg_speak(s, 2,"Finish opening %s",filename);

    ier = fscanf(fc,"%s", stype);
    if (ier != 1) xtg_error(s, "Error in reading in %s", s);

    /*
     * ------------------------------------------------------------------------
     * Facies file:
     * ------------------------------------------------------------------------
     */
    if (strncmp(stype,"storm_facies_binary",19) == 0) {
	istype=1;
	xtg_speak(s, 2,"Storm file is type FACIES");
	ier = fscanf(fc,"%d%s%d", &nzone, stormparfile, &nnn);
        if (ier != 3) xtg_error(s, "Error in reading in %s", s);
	xtg_speak(s, 2,"Reading: %d %s %d",nzone,stormparfile,nnn);
	ier = fscanf(fc,"%d", &nfac1);
        if (ier != 1) xtg_error(s, "Error in reading in %s", s);
	xtg_speak(s, 2,"Reading: NFAC1 = %d",nfac1);
	for (i=0;i<nfac1;i++) {
            ier = fscanf(fc,"%s", sdum);
            if (ier != 1) xtg_error(s, "Error in reading in %s", s);
            ier = fscanf(fc,"%d", &nfac2);
            if (ier != 1) xtg_error(s, "Error in reading in %s", s);
        }
        xtg_speak(s, 2,"Reading: NFAC2 = %d",nfac2);
	for (i=0;i<nfac2;i++) {
            ier = fscanf(fc,"%d", &ndum);
            if (ier != 1) xtg_error(s, "Error in reading in %s", s);
        }

	/* now I come the stuff I really need for this: */
	ier = fscanf(fc,"%f%f%f%f%s%s%s%s%f%f%d%d%d",
                     &xstart, &xlen, &ystart, &ylen, s_top, s_bot,
                     s_tero, s_bero, &zmax, &rotation,
                     &nnx, &nny, &nnz);
        if (ier != 13) xtg_error(s, "Error in reading in %s", s);


	xtg_speak(s, 2,"%f %f %f %f %s %s %s %s %f %f\n%d %d %d",
		xstart,xlen,ystart,ylen,s_top,s_bot,
		s_tero,s_bero,zmax,rotation,
		nnx, nny, nnz);

    }
    else if (strncmp(stype,"storm_petro_binary",18) == 0) {

	istype=2;
	xtg_speak(s, 2,"Storm file is type PETRO");

	ier = fscanf(fc,"%d%s%d", &nzone, stormparfile, &nnn);
        if (ier != 3) xtg_error(s, "Error in reading in %s", s);

	ier = fscanf(fc,"%s", prop);
        if (ier != 1) xtg_error(s, "Error in reading in %s", s);

	/* now I come the stuff I really need for this: */
	ier = fscanf(fc,"%f%f%f%f%s%s%s%s%f%f%d%d%d",
                     &xstart, &xlen, &ystart, &ylen, s_top, s_bot,
                     s_tero, s_bero, &zmax, &rotation,
                     &nnx, &nny, &nnz);
        if (ier != 13) xtg_error(s, "Error in reading in %s", s);

	xtg_speak(s, 2,"%f %f %f %f %s %s %s %s %f %f\n%d %d %d",
                  xstart,xlen,ystart,ylen,s_top,s_bot,
                  s_tero,s_bero,zmax,rotation,
                  nnx, nny, nnz);

    }
    else {
	xtg_error(s,"Unknown file type! STOP!");
	exit(1);
    }

    xtg_speak(s, 1,"Grid size: %d x %d x %d",nnx,nny,nnz);

    /*
     * ========================================================================
     * NEXT JOB IS TO READ THE ARRAY
     * ========================================================================
     */

    nxyz=nnx*nny*nnz;
    if (nxyz != inxyz) {
	xtg_warn(s,1,"<grd3d_import_storm_prop.c> NXYZ=%d != INXYZ=%d",nxyz,inxyz);
    }

    for (ib=0; ib<nxyz; ib++) {
	/* read a single value at the time */
	if (istype == 1) {
	    xtg_speak(s, 4,"Trying fread for INT pointer");
	    numxxx=fread (int_v,8,nxyz,fc);
	}
	else if (istype == 2) {
	    xtg_speak(s, 4,"Trying fread for DOUBLE pointer");
	    numxxx=fread (double_v,8,nxyz,fc);
	}
	xtg_speak(s, 1,"%d values read from Storm file...",numxxx);

    }



    /*
     * ========================================================================
     * NEXT JOB IS TO FIND AREALLY DEFINED CELLS. USE ONE MAP ONLY
     * ========================================================================
     */





    return 0;
}
