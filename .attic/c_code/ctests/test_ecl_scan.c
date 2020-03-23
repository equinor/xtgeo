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
    char *keywords;
    int *rectypes;
    long *reclengths, *recstarts, nkey;
    char *token, *tofree;
    int msec = 0;
    char s[24] = "test_ecl_scan";
    const char *senv = getenv("XTG_VERBOSE_LEVEL");
    FILE *fc;

    debug = 0;
    if (senv != NULL) debug = atoi(senv);

    keywords = (char *) calloc(maxkw*10, sizeof(char));
    rectypes = (int *) calloc(maxkw, sizeof(int));
    reclengths = (long *) calloc(maxkw, sizeof(long));
    recstarts = (long *) calloc(maxkw, sizeof(long));

    xtgverbose(debug);
    xtg_verbose_file("NONE");

    plan(NO_PLAN);

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/GULLFAKS_R003B-0.UNRST"); */

    /* fc = fopen(file, "rb"); */
    /* clock_t before = clock();  /\* time option *\/ */
    /* nkey = grd3d_scan_eclbinary(fc, keywords, rectypes, reclengths, */
    /*                             recstarts, maxkw, debug); */

    /* clock_t difference = clock() - before; */
    /* msec = difference * 1000 / CLOCKS_PER_SEC; */

    /* xtg_speak(s, 1, "Scanning of <%d> keywords took %d milliseconds (%f secs)", */
    /*           nkey, msec, msec/1000.0); */

    /* if (debug >= 2) { */
    /*     tofree = keywords; */
    /*     ic = 0; */
    /*     while ((token = strsep(&keywords, "|")) != NULL) */
    /*         printf("Keyword %d: <%s> of type %d, length %d, position %d\n", ic++, token, rectypes[ic], */
    /*                reclengths[ic], recstarts[ic]); */

    /*     free(tofree); */
    /* } */

    /* ok (nkey == 1155, "Number of keywords"); */

    done_testing();
}
