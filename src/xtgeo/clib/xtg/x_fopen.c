#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

/* improved fopen; checks more stuff */

FILE *
x_fopen(const char *filename, const char *mode)
{

    FILE *fc;

    fc = fopen(filename, mode);

    if (fc == NULL) {
        logger_warn(LI, FI, FU, "Some thing is wrong with requested filename <%s>",
                    filename);
        logger_critical(LI, FI, FU,
                        "Could be: Non existing folder, wrong permissions ? ..."
                        " anyway: STOP!",
                        FU);
    }

    return fc;
}
