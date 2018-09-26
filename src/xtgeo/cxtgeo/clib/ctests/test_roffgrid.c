/* test the roff grid props routines */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    int    ib, ier, option=0, debug, nn, ncoord, nzcorn;
    char   file[133], pname[33];
    int    nx, ny, nz, nsubs, ndef;
    double *pcoord, *pzcorn;
    int    *pact, *psubs;

    debug = 3;

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    /*
     * -------------------------------------------------------------------------
     * Read an existing grid
     * -------------------------------------------------------------------------
     */

    /* strcpy(file,"../../../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff"); */



    /* grd3d_scan_roff_bingrid(&nx, &ny, &nz, &nsubs, */
    /*     		    file, debug); */

    /* ok(nx == 99, "NX"); */

    /* nn = nx*ny*nz; */
    /* ncoord = (nx+1)*(ny+1)*2*3; */
    /* nzcorn = nx*ny*(nz+1)*4; */

    /* pcoord=calloc(ncoord,8); */
    /* pzcorn=calloc(nzcorn,8); */
    /* pact=calloc(nn,4); */
    /* psubs=calloc(nsubs,4); */

    /* grd3d_import_roff_grid(&ndef, &nsubs, pcoord, */
    /*     		   pzcorn, pact, */
    /*     		   psubs, nsubs, file, */
    /*     		   debug); */

    done_testing();

}
