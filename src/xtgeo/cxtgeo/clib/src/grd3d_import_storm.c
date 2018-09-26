/*
 * ############################################################################
 * grd3d_import_storm.c
 * Basic routines to handle import of Storm 3D grids
 * ############################################################################
 * $Id: grd3d_import_storm.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_import_storm.c,v $
 *
 * $Log: grd3d_import_storm.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


/*
 * ****************************************************************************
 *                        GRD3D_IMPORT_STORM_BINARY
 * ****************************************************************************
 * The format is special. The grid is regular XY, and it is not defined as
 * a explicit grid, but by bounding surfaces. Therefore, we will need
 * to import the corrosponding surfaces to defines the grid
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
 * Notice that the last 0.000... for the grd files are the top erosion and/or
 * bottom erosion grids, respectively. (0.0000 if not present!)
 *
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

int grd3d_import_storm_binary (
			       int   m_action,
			       int   *nx,
			       int   *ny,
			       int   *nz,
			       int   nxyz,
			       int   *num_active,
			       double *grd3d_v,
			       int   *iactnum_v,
			       char  *filename,
			       int   debug
			       )
{

    int                 i,j,k, nnn, nzone, nfac1, nfac2, ndum, nnx, nny, nnz;
    int                 ib=0, ic, ixyz;

    float               xstart, xlen, ystart, ylen, zmax, rotation, angle;
    float               dx, dy, dz;
    char                stype[25], stormparfile[30], sdum[30], param[30];
    char                s_top[132], s_bot[132], s_tero[132], s_bero[132];
    FILE                *fc;
    char                s[24]="grd3d_import_storm";

    /*
     * ========================================================================
     * INITIALLY, READ ASCII HEADER (CONTAINS ALL GRID-INFO!)
     * ========================================================================
     */

    xtgverbose(debug);

    xtg_speak(s,1,"Opening %s",filename);
    fc=fopen(filename,"rb");
    xtg_speak(s,2,"Finish opening %s",filename);

    fscanf(fc,"%s", stype);

    /*
     * ------------------------------------------------------------------------
     * Facies file:
     * ------------------------------------------------------------------------
     */
    if (strncmp(stype,"storm_facies_binary",19) == 0) {
	xtg_speak(s,2,"Storm file is type FACIES");
	fscanf(fc,"%d%s%d", &nzone,stormparfile,&nnn);
	xtg_speak(s,2,"Reading: %d %s %d",nzone,stormparfile,nnn);
	fscanf(fc,"%d", &nfac1);
	xtg_speak(s,2,"Reading: NFAC1 = %d",nfac1);
	for (i=0;i<nfac1;i++) fscanf(fc,"%s", sdum);
	fscanf(fc,"%d", &nfac2);
	xtg_speak(s,2,"Reading: NFAC2 = %d",nfac2);
	for (i=0;i<nfac2;i++) fscanf(fc,"%d", &ndum);

	/* now I come the stuff I really need for this: */
	fscanf(fc,"%f%f%f%f%s%s%s%s%f%f%d%d%d",
	       &xstart,&xlen,&ystart,&ylen,s_top,s_bot,
	       s_tero,s_bero,&zmax,&rotation,
	       &nnx, &nny, &nnz);
	xtg_speak(s,2,"%f %f %f %f %s %s %s %s %f %f\n%d %d %d",
		     xstart,xlen,ystart,ylen,s_top,s_bot,
		     s_tero,s_bero,zmax,rotation,
		     nnx, nny, nnz);

    }
    else if (strncmp(stype,"storm_petro_binary",18) == 0) {

	xtg_speak(s,2,"Storm file is type PETRO");
	fscanf(fc,"%d%s%d", &nzone,stormparfile,&nnn);
	fscanf(fc,"%s", param);

	/* now I come the stuff I really need for this: */
	fscanf(fc,"%f%f%f%f%s%s%s%s%f%f%d%d%d",
	       &xstart,&xlen,&ystart,&ylen,s_top,s_bot,
	       s_tero,s_bero,&zmax,&rotation,
	       &nnx, &nny, &nnz);
	xtg_speak(s,2,"%f %f %f %f %s %s %s %s %f %f\n%d %d %d",
		xstart,xlen,ystart,ylen,s_top,s_bot,
		s_tero,s_bero,zmax,rotation,
		nnx, nny, nnz);

    }

    else {
	xtg_speak(s,3,"Unknown file type! STOP!");
	exit(1);
    }

    xtg_speak(s,1,"Grid size: %d x %d x %d",nnx,nny,nnz);

    /*
     * ========================================================================
     * NEXT JOB IS TO COMPUTE THE XYZ RECTANGULAR BOX
     * ========================================================================
     */

    angle=PI*rotation/180.0; /*radians */

    dx=xlen/nnx; dy=ylen/nny; dz=zmax/nnz;

    xtg_speak(s,3,"Making regular 3D grid...");

    for (k=1; k<=nnz; k++) {
	for (j=1; j<=nny; j++) {
	    for (i=1; i<=nnx; i++) {
		ib=x_ijk2ib(i,j,k,nnx,nny,nnz,0); /* sequential block enumeration */

		iactnum_v[ib]=1;
		ic=1; ixyz=1; /* X corner no. 1 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i-1)*dx*cos(angle) +
		    (j-1)*dy*cos(PIHALF+angle);

		ic=1; ixyz=2; /* Y corner no. 1 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j-1)*dy*sin(PIHALF+angle) +
		    (i-1)*dx*sin(angle);

		ic=1; ixyz=3; /* Z corner no. 1 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k-1)*dz;

		/* ------------------------------------------------------- */

		ic=2; ixyz=1; /* X corner no. 2 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i)*dx*cos(angle) +
		    (j-1)*dy*cos(PIHALF+angle);

		ic=2; ixyz=2; /* Y corner no. 2 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j-1)*dy*sin(PIHALF+angle) +
		    (i)*dx*sin(angle);

		ic=2; ixyz=3; /* Z corner no. 2 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k-1)*dz;

		/* ------------------------------------------------------- */

		ic=3; ixyz=1; /* X corner no. 3 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i-1)*dx*cos(angle) +
		    (j)*dy*cos(PIHALF+angle);

		ic=3; ixyz=2; /* Y corner no. 3 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j)*dy*sin(PIHALF+angle) +
		    (i-1)*dx*sin(angle);

		ic=3; ixyz=3; /* Z corner no. 3 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k-1)*dz;

		/* ------------------------------------------------------- */

		ic=4; ixyz=1; /* X corner no. 4 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i)*dx*cos(angle) +
		    (j)*dy*cos(PIHALF+angle);

		ic=4; ixyz=2; /* Y corner no. 4 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j)*dy*sin(PIHALF+angle) +
		    (i)*dx*sin(angle);

		ic=4; ixyz=3; /* Z corner no. 4 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k-1)*dz;

		/* ------------------------------------------------------- */

		ic=5; ixyz=1; /* X corner no. 5 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i-1)*dx*cos(angle) +
		    (j-1)*dy*cos(PIHALF+angle);

		ic=5; ixyz=2; /* Y corner no. 5 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j-1)*dy*sin(PIHALF+angle) +
		    (i-1)*dx*sin(angle);

		ic=5; ixyz=3; /* Z corner no. 5 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k)*dz;

		/* ------------------------------------------------------- */

		ic=6; ixyz=1; /* X corner no. 6 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i)*dx*cos(angle) +
		    (j-1)*dy*cos(PIHALF+angle);

		ic=6; ixyz=2; /* Y corner no. 6 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j-1)*dy*sin(PIHALF+angle) +
		    (i)*dx*sin(angle);

		ic=6; ixyz=3; /* Z corner no. 6 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k)*dz;

		/* ------------------------------------------------------- */

		ic=7; ixyz=1; /* X corner no. 7 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i-1)*dx*cos(angle) +
		    (j)*dy*cos(PIHALF+angle);

		ic=7; ixyz=2; /* Y corner no. 7 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j)*dy*sin(PIHALF+angle) +
		    (i-1)*dx*sin(angle);

		ic=7; ixyz=3; /* Z corner no. 7 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k)*dz;

		/* ------------------------------------------------------- */

		ic=8; ixyz=1; /* X corner no. 8 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    xstart+(i)*dx*cos(angle) +
		    (j)*dy*cos(PIHALF+angle);

		ic=8; ixyz=2; /* Y corner no. 8 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    ystart+(j)*dy*sin(PIHALF+angle) +
		    (i)*dx*sin(angle);

		ic=8; ixyz=3; /* Z corner no. 8 */
		grd3d_v[24*(ib-1)+3*(ic-1)+ixyz-1]=
		    (k)*dz;

	    }
	}
    }

    xtg_speak(s,3,"Making regular 3D grid... FINISHED");

    *nx=nnx;
    *ny=nny;
    *nz=nnz;
    *num_active=ib+1; /* here ... */

    xtg_speak(s,3,"Number of active cells is %d", *num_active);

    /*
     * ========================================================================
     * NEXT JOB IS TO FIND AREALLY DEFINED CELLS. USE ONE MAP ONLY
     * ========================================================================
     */





    return 0;
}
