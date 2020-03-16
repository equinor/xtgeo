/* test the roff grid props routines */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    int    ib, ier, option=0;
    char   file[70], pname[33];
    int    nx, ny, nz, ptype, ncodes;
    double xori, yori, xinc, yinc, rot;
    int    mode, debug, ndef;
    double *p_double_v;
    char   *p_codenames_v;
    int    *p_int_v, *p_codevalues_v;

    debug = 3;

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    /*
     * ------------------------------------------------------------------------
     * Read an existing property in ROFF format
     * ------------------------------------------------------------------------
     */

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/gullfaks2_zone.roff"); */
    /* strcpy(pname,"Zone"); */

    /* p_codenames_v=calloc(999,sizeof(char)); */
    /* p_codevalues_v=calloc(888,sizeof(int)); */

    /* mode = 0; */
    /* printf("Scanning...\n"); */
    /* ier = grd3d_imp_prop_roffbin(file, mode, &ptype, &nx, &ny, &nz, &ncodes, */
    /*     			 pname, p_int_v, p_double_v, p_codenames_v, */
    /*     			 p_codevalues_v, option, debug); */

    /* mode = 1; */

    /* ok(ptype == 2, "Property type is 2"); */
    /* ok(ier == 0, "Property was found"); */


    /* if (ier != 0) printf("Parameter not found...\n"); */

    /* if (ptype==2) { */
    /*     p_int_v=calloc(nx*ny*nz,sizeof(int)); */
    /*     p_double_v=calloc(1, sizeof(double)); */
    /* } */
    /* else{ */
    /*     p_int_v=calloc(1, sizeof(int)); */
    /*     p_double_v=calloc(nx*ny*nz,sizeof(double)); */
    /* } */


    /* p_codenames_v=calloc(ncodes*33,sizeof(char)); */
    /* p_codevalues_v=calloc(ncodes,sizeof(int)); */


    /* printf("Reading...\n"); */
    /* ier = grd3d_imp_prop_roffbin(file, mode, &ptype, &nx, &ny, &nz, &ncodes, */
    /*     			 pname, p_int_v, p_double_v, p_codenames_v, */
    /*     			 p_codevalues_v, option, debug); */

    /* printf("Property no 277 (base 0) is: %d\n", p_int_v[277]); */
    /* printf("Property no 199 (base 0) is: %d\n", p_int_v[199]); */
    /* printf("Property no last (base 0) is: %d\n", p_int_v[nx*ny*nz-1]); */

    /* printf("IER: %d\n",ier); */

    /*
     * ------------------------------------------------------------------------
     * Read another existing property in ROFF format, file with mix of int
     * and floats (Emerald case)
     * ------------------------------------------------------------------------
     */

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff"); */
    /* strcpy(pname,"Column"); */

    /* mode = 0; */
    /* printf("Scanning... %s\n", file); */
    /* ier = grd3d_imp_prop_roffbin(file, mode, &ptype, &nx, &ny, &nz, &ncodes, */
    /*     			 pname, p_int_v, p_double_v, p_codenames_v, */
    /*     			 p_codevalues_v, option, debug); */

    /* mode = 1; */

    /* ok(ptype == 2, "Property type is 2"); */
    /* ok(ier == 0, "Property was found"); */


    /* if (ier != 0) printf("Parameter not found...\n"); */

    /* if (ptype==2) { */
    /*     p_int_v=calloc(nx*ny*nz,sizeof(int)); */
    /*     p_double_v=calloc(1, sizeof(double)); */
    /* } */
    /* else{ */
    /*     p_int_v=calloc(1, sizeof(int)); */
    /*     p_double_v=calloc(nx*ny*nz,sizeof(double)); */
    /* } */


    /* p_codenames_v=calloc(ncodes*33,sizeof(char)); */
    /* p_codevalues_v=calloc(ncodes,sizeof(int)); */


    /* printf("Reading...\n"); */
    /* ier = grd3d_imp_prop_roffbin(file, mode, &ptype, &nx, &ny, &nz, &ncodes, */
    /*     			 pname, p_int_v, p_double_v, p_codenames_v, */
    /*     			 p_codevalues_v, option, debug); */

    /* /\* printf("%d\n",p_int_v[199]); *\/ */
    /* ok(p_int_v[199]==60, "Column parameter Emerald"); */

    /* printf("IER: %d\n",ier); */





    done_testing();

}
