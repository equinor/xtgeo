/*
 * ############################################################################
 * xtg_io.c
 * Wrappers for file IO
 * ############################################################################
 */
#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"
#define _GNU_SOURCE 1

FILE *xtg_fopen(const char *filename, const char *mode)
{
    FILE *fhandle = NULL;

    fhandle = fopen(filename, mode);

    if (fhandle == NULL) {
        perror("Cannot open file");
    }

    return fhandle;
}

#ifdef __linux__
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
#else
FILE *xtg_fopen_bytestream(char *stream, long nstream, const char *mode)
{
    FILE *fhandle = NULL;
    logger_critical(__LINE__, "Opening bytestrem is not implemented on this platform!");
    return fhandle;
}
#endif

int xtg_fclose(FILE *fhandle)
{
    return fclose(fhandle);
}
