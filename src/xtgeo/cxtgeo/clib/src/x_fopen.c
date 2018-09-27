#include "libxtg.h"
#include "libxtg_.h"


/* improved fopen; checks more stuff */

FILE *x_fopen(const char *filename, const char *mode, int debug) {

    FILE *fc;
    char s[24] = "x_fopen";
    xtgverbose(debug);


    fc = fopen(filename, mode);

    if (fc == NULL) {
        xtg_warn(s, 0, "Some thing is wrong with requested filename <%s>",
                 filename);
        xtg_error(s, "Could be: Non existing folder, wrong permissions ? ..."
                  " anyway: STOP!", s);
        exit(345);
    }
    return fc;
}
