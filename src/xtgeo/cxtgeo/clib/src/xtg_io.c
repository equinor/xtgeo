/*
 * ############################################################################
 * xtg_io.c
 * Wrappers for file IO
 * ############################################################################
 */

#include <stdio.h>
#include "libxtg.h"

FILE *xtg_fopen(const char *filename, const char *mode)
{
    FILE *fhandle = NULL;

    fhandle = fopen(filename, mode);

    if (fhandle == NULL) {
        perror("Cannot open file");
    }

    return fhandle;
}

int xtg_fclose(FILE *fhandle)
{
    return fclose(fhandle);
}
