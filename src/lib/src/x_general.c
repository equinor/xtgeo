/*
 * ############################################################################
 * Some wrappers on common C function to avoid warning messages while compile:
 *     - fgets
 *
 *: JRIV
 * ############################################################################
 * ############################################################################
 */
#include <stddef.h>
#include <stdio.h>
#include "common.h"
#include "logger.h"

void
x_fread(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t ier;

    ier = fread(ptr, size, nmemb, stream);

    if (ier != nmemb) {
        logger_error(LI, FI, FU, "Problem in fread: IER=%d nmemb=%d", ier, nmemb);
    }
}
