/* test SCAN of Eclipse binary files */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>   // for atoi

/* Test scanning of Eclipse binary files; should be very fast */

int main () {
    char file[170];
    int debug, ier, ic;
    long maxkw = 2000;
    char *tags;
    int *rectypes;
    long *reclengths, *recstarts, nkey;
    char *token, *tofree;
    int msec = 0, nrec, swap;
    char s[24] = "test_ecl_scan";
    const char *senv = getenv("XTG_VERBOSE_LEVEL");
    FILE *fc;

    debug = 2;
    if (senv != NULL) debug = atoi(senv);

    tags = (char *) calloc(maxkw*100, sizeof(char));
    rectypes = (int *) calloc(maxkw, sizeof(int));
    reclengths = (long *) calloc(maxkw, sizeof(long));
    recstarts = (long *) calloc(maxkw, sizeof(long));

    xtgverbose(debug);
    xtg_verbose_file("NONE");

    plan(NO_PLAN);

    strcpy(file,"../../../../../../xtgeo-testdata/3dgrids/reek/reek_grd_w_props.roff");

    fc = fopen(file, "rb");
    clock_t before = clock();  /* time option */
    nrec = grd3d_scan_roffbinary(fc, &swap, tags, rectypes, reclengths,
                                 recstarts, maxkw, debug);

    clock_t difference = clock() - before;
    msec = difference * 1000 / CLOCKS_PER_SEC;

    xtg_speak(s, 1, "Scanning of <%d> keywords took %d milliseconds (%f secs)",
              nkey, msec, msec/1000.0);

    xtg_speak(s, 1, "NREC: <%d> SWAP is %d Tags: <%s>", nrec, swap, tags);

    if (debug >= 2) {
        tofree = tags;
        ic = 0;
        while ((token = strsep(&tags, "|")) != NULL)
            if (strlen(token) >= 1) {
                printf("Keyword %d: <%s> of type %d, length %ld, "
                       "position %ld\n",
                       ic++, token, rectypes[ic],
                       reclengths[ic], recstarts[ic]);
            }
        free(tofree);
    }

    ok (nrec == 30, "Number of records");
    ok (reclengths[29] == 35840, "Length of data array");

    done_testing();
}
