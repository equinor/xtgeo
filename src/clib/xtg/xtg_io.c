/*
 * ############################################################################
 * xtg_io.c
 * Wrappers for file IO
 * ############################################################################
 */
#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#define _GNU_SOURCE 1

FILE *
xtg_fopen(const char *filename, const char *mode)
{
    FILE *fhandle = NULL;

    fhandle = fopen(filename, mode);

    if (fhandle == NULL) {
        perror("Cannot open file");
    }

    return fhandle;
}

#ifdef __linux__
FILE *
xtg_fopen_bytestream(char *stream, long nstream, const char *mode)
{
    FILE *fhandle = NULL;

    if (strncmp(mode, "w", 1) == 0) {
        size_t len;
        logger_info(LI, FI, FU, "Write to memory buffer");

        fhandle = open_memstream(&stream, &len);

        if (fhandle == NULL) {
            perror("Cannot open file memory stream for write");
        }
    } else {
        void *buff;
        logger_info(LI, FI, FU, "Read from memory buffer");

        buff = (void *)stream;

        fhandle = fmemopen(buff, nstream, mode);

        if (fhandle == NULL) {
            perror("Cannot open file memory stream for read");
        }
    }
    return fhandle;
}
#else
FILE *
xtg_fopen_bytestream(char *stream, long nstream, const char *mode)
{
    FILE *fhandle = NULL;
    logger_critical(LI, FI, FU, "Bytestream open is not implemented on this platform!");
    return fhandle;
}
#endif

int
xtg_fflush(FILE *fhandle)
{
    return fflush(fhandle);
}

long
xtg_ftell(FILE *fhandle)
{
    return ftell(fhandle);
}

int
xtg_fclose(FILE *fhandle)
{
    return fclose(fhandle);
}

int
xtg_get_fbuffer(FILE *fhandle, char *stream, long nstream)
{
    /* return the current buffer from filehandle*/
    long npos, nn;
    npos = ftell(fhandle); /* ftell given the current file offset */

    if (npos > nstream) {
        logger_critical(LI, FI, FU, "NPOS = %ld > NSTREAM = %ld", npos, nstream);
        return EXIT_FAILURE;
    }

    fseek(fhandle, 0, SEEK_SET);

    for (nn = 0; nn < npos; nn++) {
        char ch = fgetc(fhandle);
        if (feof(fhandle))
            break;
        stream[nn] = ch;
    }
    return EXIT_SUCCESS;
}
