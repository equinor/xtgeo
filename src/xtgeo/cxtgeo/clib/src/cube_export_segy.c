/*
****************************************************************************************
 *
 * NAME:
 *    cube_export_segy.c
 *
 * DESCRIPTION:
 *    Export to SEGY format.
 *
 * ARGUMENTS:
 *    sfile          i     SEGY file
 *    nx..nz         i     Cube dimensions ; i/o dependent on flags
 *    p_cube_v       i     The 1D cube array (C order)
 *    xori..zinc     i     Cube metadata for geometries.
 *    rotation       i     Cube roatation: inline compared with UTMX, angle in
 *                         degrees, anti-clock, always in [0, 360>
 *    yflip          o     1 for normal, -1 if flipped in Y (think rotation=0)
 *    zflip          o     Currently not in use
 *    ilinesp        i     Array in INLINE index
 *    xlinesp        i     Array in XLINE index
 *    tracidp        i     Array of traceidcodes
 *    option         i     Unused
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    - update ARGUMENT list above
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include <limits.h>
#include "libxtg.h"
#include "libxtg_.h"


int _write_int_as_4bytes(FILE *fc, int nn) {
    int swap;
    swap = x_swap_check();

    if (swap == 1) SWAP_INT(nn);

    if (fwrite(&nn, 4, 1, fc) != 1) {
        return -1;
    }
    return 4;
}

int _write_int_as_2bytes(FILE *fc, int nn) {
    short n;
    int swap;

    swap = x_swap_check();

    n = nn;

    if (swap == 1) SWAP_SHORT(n);

    if (fwrite(&n, 2, 1, fc) != 1) {
        return -1;
    }
    return 2;
}

int _write_nbytes(FILE *fc, int nbytes) {
    char xx;
    int i;

    for (i = 0; i < nbytes; i++) {
        xx = '0';
        if (fwrite(&xx, 1, 1, fc) != 1) {
            return -1;
        }
    }
    return nbytes;

}


int cube_export_segy (
                      char *sfile,
                      int nx,
                      int ny,
                      int nz,
                      float *p_cube_v,
                      long ntotal,
                      double xori,
                      double xinc,
                      double yori,
                      double yinc,
                      double zori,
                      double zinc,
                      double rotation,
                      int yflip,
                      int zflip,
                      int *ilinesp,
                      int *xlinesp,
                      int *tracidp,
                      int option
                      )
{

    FILE *fc = NULL;
    int ic, i, j, k, nc, nn, ier, swap;
    int ixv, iyv;
    long nxy = 0;
    long ilc;
    unsigned char ubyte;
    float aval;

    double *xv, *yv;

    xv = calloc(nx*ny, sizeof(double));
    yv = calloc(nx*ny, sizeof(double));

    fc = fopen(sfile, "wb");

    swap = x_swap_check();

    ilc = nx * ny * nz;

    /*
     * ========================================================================
     * Textual File Header 3200 byte
     * ========================================================================
     */
    for (ic = 1; ic <= 40; ic++) {
        if (ic == 1) {
            fprintf(fc, "C%2d %-75s\n", ic, "OUTPUT FROM XTGEO");
        }
        else if (ic == 39) {
            fprintf(fc, "C%2d %-75s\n", ic, "SEG-Y REV1.0");
        }
        else if (ic == 40) {
            fprintf(fc, "C%2d %-75s\n", ic, "END TEXTUAL HEADER");
        }
        else{
            fprintf(fc, "C%2d %-75s\n", ic, " .............. ");
        }
    }

    /*
     * ========================================================================
     * Binary header 400 byte
     * ========================================================================
     */
    nc = 0;

    nc += _write_int_as_4bytes(fc, 9999);
    nc += _write_int_as_4bytes(fc, 1);
    nc += _write_int_as_4bytes(fc, 1);

    nc += _write_int_as_2bytes(fc, 1);
    nc += _write_int_as_2bytes(fc, 1);
    nc += _write_int_as_2bytes(fc, (int)zinc*1000); // sample interval
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, nz);             // N samples per trace
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 5);              // 5: 4 byte IEEE float
    nc += _write_int_as_2bytes(fc, 1);              // ensemble fold
    nc += _write_int_as_2bytes(fc, 4);              // trace sorting
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);

    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);

    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);              // Amplitude recovery

    nc += _write_int_as_2bytes(fc, 1);              // meters as meas.sys
    nc += _write_int_as_2bytes(fc, 0);
    nc += _write_int_as_2bytes(fc, 0);              // Virb polarity code

    nc += _write_nbytes(fc, 240);                   // unassigned in REV1

    /* SEGY version */
    ubyte=1; fwrite(&ubyte, 1, 1, fc);
    ubyte=0; fwrite(&ubyte, 1, 1, fc);
    nc += 2;
    nc += _write_int_as_2bytes(fc, 1);              // fixed length trace flag
    nc += _write_int_as_2bytes(fc, 0);

    nc += _write_nbytes(fc, 94);                    // unassigned in REV1

    /*
     * ========================================================================
     * Traces
     * ========================================================================
     */

    /* first get the X and Y coordinates */
    nxy = nx * ny;
    ier = surf_xy_as_values(xori, xinc, yori, yinc*yflip, nx, ny, rotation,
                            xv, nxy, yv, nxy, 1, XTGDEBUG);

    if (ier != 0) exit(-132);

    for (i = 1; i <= nx; i++) {  // inline
        for (j = 1; j <= ny; j++) {
            nc = 0;

            ilc = x_ijk2ic(i, j, 1, nx, ny, 1, 0);
            /*
             * ----------------------------------------------------------------
             * Binary trace header 240 byte, a total of 84 entries
             * ----------------------------------------------------------------
             */

            for (nn=1; nn <= 7; nn++) {
                nc += _write_int_as_4bytes(fc, 0);
            }

            nc += _write_int_as_2bytes(fc, tracidp[ilc]);  // 29 trace id code

            for (nn=9; nn <= 11; nn++) {
                nc += _write_int_as_2bytes(fc, 0);
            }

            for (nn=12; nn <= 19; nn++) {
                nc += _write_int_as_4bytes(fc, 0);
            }

            nc += _write_int_as_2bytes(fc, 0);         // 20
            nc += _write_int_as_2bytes(fc, -100);      // 21

            for (nn=22; nn <= 25; nn++) {
                nc += _write_int_as_4bytes(fc, 0);
            }
            for (nn=26; nn <= 35; nn++) {
                nc += _write_int_as_2bytes(fc, 0);
            }

            nc += _write_int_as_2bytes(fc, (int)zori);  // 36
            nc += _write_int_as_2bytes(fc, 0);          // 37
            nc += _write_int_as_2bytes(fc, 0);          // 38
            nc += _write_int_as_2bytes(fc, nz);         // 39
            nc += _write_int_as_2bytes(fc, (int)zinc*1000);  // 40 sample intv.

            for (nn=41; nn <= 71; nn++) {nc += _write_int_as_2bytes(fc, 0);}
            ixv = (int)(xv[ilc] * 100);
            iyv = (int)(yv[ilc] * 100);
            nc += _write_int_as_4bytes(fc, ixv);  // 72
            nc += _write_int_as_4bytes(fc, iyv);  // 73

            nc += _write_int_as_4bytes(fc, ilinesp[i - 1]);    // 74
            nc += _write_int_as_4bytes(fc, xlinesp[j - 1]);    // 75
            nc += _write_int_as_4bytes(fc, 0);      // 76
            for (nn=77; nn <= 78; nn++) {nc += _write_int_as_2bytes(fc, 0);}
            nc += _write_int_as_4bytes(fc, 0);       // 79
            for (nn=80; nn <= 85; nn++) {nc += _write_int_as_2bytes(fc, 0);}
            for (nn=86; nn <= 87; nn++) {nc += _write_int_as_4bytes(fc, 0);}
            for (nn=88; nn <= 93; nn++) {nc += _write_int_as_2bytes(fc, 0);}

            /*
             * ----------------------------------------------------------------
             * Binary values (use IEEE float)
             * ----------------------------------------------------------------
             */

            for (k=1; k<= nz; k++) {
                ilc = x_ijk2ic(i, j, k, nx, ny, nz, 0);
                aval = p_cube_v[ilc];

                if (swap == 1) SWAP_FLOAT(aval);
                if (fwrite(&aval, 4, 1, fc) != 1) {
                    return -9;
                }
            }
        }
    }

    fclose(fc);

    free(xv);
    free(yv);


    return EXIT_SUCCESS;
}
