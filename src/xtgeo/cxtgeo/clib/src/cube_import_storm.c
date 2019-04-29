/*
 ******************************************************************************
 *
 * Import a cube using STORM format
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    cube_import_storm.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *     Import a cube via the Storm petro binary format. The Storm format
 *     is column (FOrtran) ordered; hence a conversion is needed in the
 *     calling routine.
 *
 * ARGUMENTS:
 *    ncx...ncz      i     cube dimensions
 *    cxori...cxinc  i     cube origin + increment in xyz
 *    crotation      i     Cube rotation (deg, anticlock)
 *    p_cubeval_v    i     1D Array of cube values
 *    option         i     Options: 0 scan header, 1 do full import
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *    map pointers updated
 *
 * TODO/ISSUES/BUGS:
 *    - yflip handling?
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

int cube_import_storm (
                       int nx,
                       int ny,
                       int nz,
                       char *file,
                       int nlines,
                       float *p_cube_v,
                       long nxyz,
                       int option,
                       int debug
                       )
{

    FILE  *fc;
    char s[24]="cube_import_storm";
    int i, j, k, iok_close, swap;
    long ic;
    float fval;

    char *line = NULL;
    size_t len = 0;
    ssize_t read;

    xtgverbose(debug);

    swap = x_swap_check();

    /* The caller should do a check if file exist! */
    xtg_speak(s, 2, "Opening file %s at line %d", file, nlines);
    fc = fopen(file, "rb");

    /* skip header as this is parsed in Python/Perl */

    for (i = 1; i < nlines; i++) {
        read = getline(&line, &len, fc);  /* posix/gnu function */
        line[strcspn(line, "\n")] = 0;
        xtg_speak(s, 2, "Retrieved header line no %d of length %zu : %s",
                  i, read, line);
    }


    xtg_speak(s, 2, "NX NY NZ %d %d %d", nx, ny, nz);

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                /* read a single value at the time */

                if (fread (&fval, 4, 1, fc) != 1) {
                    return -4;
                }

                if (swap==1) SWAP_FLOAT(fval);

                ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);

                p_cube_v[ic] = fval;
            }
        }
    }

    iok_close=fclose(fc);

    if (iok_close != 0) {
        return(iok_close);
    }

    xtg_speak(s,1,"STORM import done, OK");

    return(EXIT_SUCCESS);
}
