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
    int    mode, debug, ndef, nklist, ndates;
    double *p_double_v, *p_zcorn_v, *p_coord_v;
    int    *p_actnum_v;
    char   *p_codenames_v;
    int    *p_int_v, *p_codevalues_v;
    int    *day, *mon, *year, *nktype, *dsuccess, *norder, nklist3;

    double *dvec_v;


    char   *useprops;


    debug = 2;

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    /*
     * -------------------------------------------------------------------------
     * Read an existing grid in EGRID format
     * -------------------------------------------------------------------------
     */

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/ECLIPSE.EGRID"); */
    /* grd3d_scan_ecl_egrid_hd(2 , &nx, &ny, &nz, */
    /*     		    file, debug); */


    /* p_coord_v = calloc((nx+1)*(ny+1)*2*3, 8); */
    /* p_zcorn_v = calloc(nx*ny*(nz+1)*4, 8); */
    /* p_actnum_v = calloc(nx*ny*nz, 4); */

    /* grd3d_import_ecl_egrid (0, nx, ny, nz, */
    /*     		    &ndef, */
    /*     		    p_coord_v, p_zcorn_v, */
    /*     		    p_actnum_v, file, debug); */


    /* printf("%d\n",nx); */
    /* ok(nx==99, "NX"); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Read properties UNRST format */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/ECLIPSE.UNRST"); */
    /* grd3d_scan_ecl_init_hd(1 , &nx, &ny, &nz, */
    /* 			   file, debug); */

    /* ok(nx==99, "NX"); */

    /* useprops=calloc(300,sizeof(char)); */

    /* nklist=1; */
    /* strcpy(useprops, "SWAT    |PRESSURE|"); */

    /* ndates=1; */

    /* nklist3 = nklist+3; */

    /* dvec_v=calloc(ndates*nklist3*nx*ny*nz, 8); */
    /* nktype=calloc(ndates*nklist3, 4); */
    /* norder=calloc(ndates*nklist3, 4); */
    /* dsuccess=calloc(ndates, 4); */
    /* day=calloc(ndates, 4); */
    /* mon=calloc(ndates, 4); */
    /* year=calloc(ndates, 4); */

    /* day[0]=22; */
    /* mon[0]=1; */
    /* year[0]=1986; */

    /* grd3d_import_ecl_prop(5, */
    /* 			  nx*ny*nz, */
    /* 			  p_actnum_v, */
    /* 			  nklist, */
    /* 			  useprops, */
    /* 			  ndates, */
    /* 			  day, */
    /* 			  mon, */
    /* 			  year, */
    /* 			  file, */
    /* 			  dvec_v, */
    /* 			  nktype, */
    /* 			  norder, */
    /* 			  dsuccess, */
    /* 			  debug */
    /* 			  ); */

    //    ok(dsuccess[0]==1, "DSUCCESS no 0 shall be 1");


    done_testing();

}
