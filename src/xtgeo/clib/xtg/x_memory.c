#include "logger.h"
#include "libxtg_.h"

/* Run free() on an arbitrary list of single pointers */

void x_free(int num, ...) {

    int i;
    va_list valist;

    va_start(valist, num);

    for (i = 0; i < num; i++){
        free(va_arg(valist, void*));
        logger_info(LI, FI, FU, "Freeing pointer %d of %d", i + 1, num);
    }

    va_end(valist);
}
