
/*
 ******************************************************************************
 *
 * NAME:
 *    cube_import_storm.c
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
#include "libxtg.h"
#include "libxtg_.h"

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

/* The original code is public domain -- Will Hartung 4/9/09 */
/* Modifications, public domain as well, by Antti Haapala, 11/10/17
   - Switched to getc on 5/23/19 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>

// if typedef doesn't exist (msvc, blah)
typedef intptr_t ssize_t;

ssize_t _getline(char **lineptr, size_t *n, FILE *stream) {
    size_t pos;
    int c;

    if (lineptr == NULL || stream == NULL || n == NULL) {
        errno = EINVAL;
        return -1;
    }

    c = getc(stream);
    if (c == EOF) {
        return -1;
    }

    if (*lineptr == NULL) {
        *lineptr = malloc(128);
        if (*lineptr == NULL) {
            return -1;
        }
        *n = 128;
    }
    pos = 0;
    while(c != EOF) {
        if (pos + 1 >= *n) {
            size_t new_size = *n + (*n >> 2);
            if (new_size < 128) {
                new_size = 128;
            }
            char *new_ptr = realloc(*lineptr, new_size);
            if (new_ptr == NULL) {
                return -1;
            }
            *n = new_size;
            *lineptr = new_ptr;
        }

        ((unsigned char *)(*lineptr))[pos ++] = c;
        if (c == '\n') {
            break;
        }
        c = getc(stream);
    }

    (*lineptr)[pos] = '\0';
    return pos;
}

int cube_import_storm (
                       int nx,
                       int ny,
                       int nz,
                       char *file,
                       int nlines,
                       float *p_cube_v,
                       long nxyz,
                       int option
                       )
{

    FILE  *fc;
    int i, j, k, iok_close, swap;
    long ic;
    float fval;

    char *line = NULL;
    size_t len = 0;
    ssize_t read;


    swap = x_swap_check();

    /* The caller should do a check if file exist! */
    fc = fopen(file, "rb");

    /* skip header as this is parsed in Python/Perl */

    for (i = 1; i < nlines; i++) {
        read = _getline(&line, &len, fc);  /* original a posix/gnu function */
        line[strcspn(line, "\n")] = 0;
    }



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

    return(EXIT_SUCCESS);
}
