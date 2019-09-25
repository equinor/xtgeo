/*
 ***************************************************************************************
 *
 * Convert from ROFF grid cornerpoint ZCORN spec to XTGeo
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_zcorn.c
 *
 * DESCRIPTION:
 *    Convert from ROFF arrays to XTGeo arrays
 *
 * ARGUMENTS:
 *    points_v       i     a [9] matrix with X Y Z of 3 points
 *    nvector        o     a [4] vector with A B C D
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int grd3d_roff2xtgeo_zcorn (
                            int nx,
                            int ny,
                            int nz,
                            float xoffset,
                            float yoffset,
                            float zoffset,
                            float xscale,
                            float yscale,
                            float zscale,
                            int *p_splitenz_v,
                            float *p_zdata_v,
                            double *p_zcorn_v,
                            int debug
                            )

{

    long ib, ic, nxyz;
    int i, j, k, l, ico, ipos, isplit;
    int *lookup_v, ncc[8];
    double z_sw_v[8], z_se_v[8], z_nw_v[8], z_ne_v[8], zz[8];

    char sbn[24] = "grd3d_roff2xtgeo_zcorn";
    xtgverbose(debug);

    xtg_speak(sbn, 2, "Transforming grid ROFF coords --> XTG representation ...");

    /*
     * ---------------------------------------------------------------------------------
     * ZCORN...
     * ---------------------------------------------------------------------------------
    */

    nxyz = (nx + 1) * (ny + 1) * (nz + 1);
    lookup_v = calloc(nxyz + 2, sizeof(int));

    lookup_v[0] = 0;
    for (i = 0; i < nxyz; i++) {
	lookup_v[i + 1] = lookup_v[i] + p_splitenz_v[i];
    }

    ib = 0;
    for (l = (nz - 1);l >= -1; l--) {
	for (j = 0; j < ny; j++) {
	    for (i = 0; i < nx; i++) {

		k = l;
		if (l == -1) k = 0;

		ncc[0] = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k;
		ncc[1] = (i + 1) * (ny + 1) * (nz + 1) + j * (nz + 1) + k;
		ncc[2] = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k;
		ncc[3] = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k;
		ncc[4] = i * (ny + 1) * (nz + 1) + j * (nz + 1) + (k + 1);
		ncc[5] = (i + 1) * (ny + 1) * (nz + 1) + j*(nz + 1) + k + 1;
		ncc[6] = i * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + k + 1;
		ncc[7] = (i + 1) * (ny + 1) * (nz + 1) + (j + 1) * (nz + 1) + (k + 1);

                for (ico = 0; ico < 8; ico++) {

                    iuse = ncc[ico];
                    ipos = lookup_v[iuse];
                    isplit = lookup_v[iuse + 1] - lookup_v[iuse];
                    if (isplit == 1) {
                        z = (zdata_v[ipos] + zoffset) * zscale;
                        z_sw_v[ico] = z;
                        z_se_v[ico] = z;
                        z_nw_v[ico] = z;
                        z_ne_v[ico] = z;
                    }
                    if (isplit == 4) {
                        z_sw_v[ico] = (zdata_v[ipos + 0] + zoffset) * zscale;
                        z_se_v[ico] = (zdata_v[ipos + 1] + zoffset) * zscale;
                        z_nw_v[ico] = (zdata_v[ipos + 2] + zoffset) * zscale;
                        z_ne_v[ico] = (zdata_v[ipos + 3] + zoffset) * zscale;
                    }
                }


		zz[1] = z_ne_v[4];
		zz[2] = z_nw_v[5];

		zz[3] = z_se_v[6];
		zz[4] = z_sw_v[7];

		zz[5] = z_ne_v[0];
		zz[6] = z_nw_v[1];

		zz[7] = z_se_v[2];
		zz[8] = z_sw_v[3];


		if (l>=0) {
		    for (ic=1;ic<=4;ic++) {
			p_zcorn_v[ib] = zz[ic];
			ib++;
		    }
		}

		if (l==-1) {
		    for (ic=5;ic<=8;ic++) {
			p_zcorn_v[ib] = zz[ic];
			ib++;
		    }
		}
            }
        }
    }

    free(lookup_v);

    return EXIT_SUCCESS;
}
