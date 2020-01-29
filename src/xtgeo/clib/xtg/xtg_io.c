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

    logger_init(__FILE__, __FUNCTION__);

    if (strncmp(mode, "w", 1) == 0) {
        size_t len;
        logger_info(__LINE__, "Write to memory buffer");

        fhandle = open_memstream(&stream, &len);

        if (fhandle == NULL) {
            perror("Cannot open file memory stream for write");
        }
    }
    else{
        void *buff;
        logger_info(__LINE__, "Read from memory buffer");

        buff = (void*)stream;

        fhandle = fmemopen(buff, nstream, mode);

        if (fhandle == NULL) {
            perror("Cannot open file memory stream for read");
        }
    }
    return fhandle;
}
#else
FILE *xtg_fopen_bytestream(char *stream, long nstream, const char *mode)
{
    FILE *fhandle = NULL;
    logger_init(__FILE__, __FUNCTION__);
    logger_critical(__LINE__, "Bytestream open is not implemented on this platform!");
    return fhandle;
}
#endif


int xtg_fseek_start(FILE *fhandle)
{
    return fseek(fhandle, 0, SEEK_SET);
}

int xtg_fflush(FILE *fhandle)
{
    return fflush(fhandle);
}

long xtg_ftell(FILE *fhandle)
{
    return ftell(fhandle);
}

int xtg_fclose(FILE *fhandle)
{
    return fclose(fhandle);
}


int xtg_get_fbuffer(FILE *fhandle, char *stream, long nstream)
{
    /* return the current buffer from filehandle*/
    long npos, nn;
    npos = ftell(fhandle); /* ftell given the current file offset */

    if (npos > nstream) {
        logger_init(__FILE__, __FUNCTION__);
        logger_critical(__LINE__, "NPOS = %ld > NSTREAM = %ld", npos, nstream);
        return EXIT_FAILURE;
    }

    fseek(fhandle, 0, SEEK_SET);

    for (nn = 0; nn < npos; nn++) {
        char ch = fgetc(fhandle);
        if (feof(fhandle)) break;
        stream[nn] = ch;
    }
    return EXIT_SUCCESS;
}
