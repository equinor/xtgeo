/*
 * ############################################################################
 * xtg_io.c
 * Wrappers for file IO
 * ############################################################################
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include "libxtg.h"
#include "libxtg_.h"

FILE *xtg_fopen(const char *filename, const char *mode)
{
    FILE *fhandle = NULL;

    fhandle = fopen(filename, mode);

    if (fhandle == NULL) {
        perror("Cannot open file");
    }

    return fhandle;
}

FILE *xtg_fopen_bytestream(char *stream, long nstream, const char *mode)
{
    FILE *fhandle = NULL;
    void *buff;

    buff = (void*)stream;

    fhandle = fmemopen(buff, nstream, mode);

    if (fhandle == NULL) {
        perror("Cannot open file");
    }

    return fhandle;
}

int xtg_fclose(FILE *fhandle)
{
    return fclose(fhandle);
}
