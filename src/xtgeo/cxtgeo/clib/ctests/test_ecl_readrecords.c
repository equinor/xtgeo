/* test SCAN of Eclipse binary files */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>   // for atoi

/* Reading some ECL records */

int rectest (int debug) {
    char file[170];
    int keytype, nx, ny, nz;
    long nkey, keylen, keystart;
    int *someint;
    float *somefloat;
    double *somedouble;
    char keyname[9];
    int msec = 0;
    char s[24] = "(rectest)";
    clock_t before, difference;
    FILE *fc;

    plan(NO_PLAN);

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/GULLFAKS_R003B-0.UNRST"); */

    /* /\* read first INTEHEAD, which is known fra scan */
    /*  * (see test_scan_ecl) to be location */
    /*  *\/ */

    /* strcpy(keyname, "INTEHEAD"); */
    /* keylen = 411; */
    /* keystart = 36; */
    /* keytype = 1; */

    /* somefloat = calloc(1, sizeof(float)); */
    /* someint = calloc(keylen, sizeof(int)); */


    /* before = clock();  /\* time option *\/ */


    /* fc = fopen(file, "rb"); */
    /* nkey = grd3d_read_eclrecord(fc, keystart, keytype, */
    /*                             someint, keylen, somefloat, 1, */
    /*                             somedouble, 1, debug); */

    /* nx = someint[8]; */
    /* ny = someint[9]; */
    /* nz = someint[10]; */

    /* ok (nx == 99, "NX (NCOL)"); */

    /* printf("%d\n", nx*ny*nz); */

    /* /\* read first PRESSURE, which is known fra scan */
    /*  * (see test_scan_ecl) to be location */
    /*  *\/ */

    /* strcpy(keyname, "PRESSURE"); */
    /* keylen = 359835; */
    /* keystart = 3609276; */
    /* keytype = 2; */

    /* somefloat = calloc(keylen, sizeof(float)); */
    /* someint = calloc(1, sizeof(int)); */


    /* nkey = grd3d_read_eclrecord(fc, keystart, keytype, */
    /*                             someint, 0, somefloat, keylen, */
    /*                             somedouble, 0, debug); */

    /* difference = clock() - before; */

    /* msec = difference * 1000 / CLOCKS_PER_SEC; */

    /* xtg_speak(s, 1, "Reading keywords took %d milliseconds (%f secs)  %d", */
    /*           nkey, msec, msec/1000.0, difference); */

    /* printf("PRESSURE no 1 is %f", somefloat[1]); */

    /* ok (fabs(somefloat[1] - 349.3872681) < 0.0001, "Pressure in cell no 1"); */

    /* fclose(fc); */

    done_testing();
}


int datetest (int debug)
{
    char file[170];
    char s[24] = "(datetest)";
    int *seqnums, *day, *mon, *year;
    int nsteps, i;
    long nn = 1000;
    FILE *fc;

    plan(NO_PLAN);

    /* seqnums = calloc(nn, sizeof(int)); */
    /* day = calloc(nn, sizeof(int)); */
    /* mon = calloc(nn, sizeof(int)); */
    /* year = calloc(nn, sizeof(int)); */


    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/GULLFAKS_R003B-0.UNRST"); */
    /* fc = fopen(file, "rb"); */


    /* nsteps = grd3d_ecl_tsteps (fc, seqnums, day, mon, year,nn, debug); */

    /* for (i=0; i<nsteps; i++) { */
    /*     xtg_speak(s, 1, "STEP %d: YEAR MON DAY:  %d  %d  %d", */
    /*               seqnums[i], year[i], mon[i], day[i]); */

    /* } */
    /* fclose(fc); */
    done_testing();
}


int main()
{
    char s[24] = "(datetest)";
    int debug;
    const char *senv = getenv("XTG_VERBOSE_LEVEL");
    debug = 0;
    if (senv != NULL) debug = atoi(senv);

    xtgverbose(debug);
    xtg_verbose_file("NONE");

    rectest(debug);
    datetest(debug);
}
