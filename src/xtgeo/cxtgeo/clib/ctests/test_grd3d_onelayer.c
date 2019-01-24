/* test map vs grd3d operations */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>

int main () {
    int ib, ier;
    char file[70];
    int  ndef, ndefsum, mx, my, debug=1;
    double xstep, ystep, xori, xmin, xmax, yori, ymin, ymax, zmin, zmax;
    double *zval_v, *zval2_v;
    int    numact, numsubs, nx, ny, nz, nactive, status;
    int    *p_actnum_v, *p_subgrd_v;
    int    *p_actnum2_v, *p_subgrd2_v;
    double *p_coord_v, *p_zcorn_v, *p_zcorn2_v;


    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    /*
     * -------------------------------------------------------------------------
     * Read an existing 3D grid
     */

    strcpy(file,"../../../../../../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff");

    nx=40;
    ny=64;
    nz=14;

    /* allocate */
    p_coord_v = calloc((nx+1)*(ny+1)*2*3,sizeof(double));
    p_zcorn_v = calloc(nx*ny*(nz+1)*4,sizeof(double));
    p_actnum_v = calloc(nx*ny*nz,sizeof(int));
    p_subgrd_v = calloc(1,sizeof(int));


    printf("Reading grid\n");
    grd3d_import_roff_grid(&numact, &numsubs, p_coord_v, p_zcorn_v,
			   p_actnum_v, p_subgrd_v, 1, file, debug);

    printf("Reading grid done\n");

    ok(p_actnum_v[33]==1, "ACTNUM of location 33");


    p_zcorn2_v = calloc(nx*ny*(1+1)*4,sizeof(double));
    p_actnum2_v = calloc(nx*ny*1,sizeof(int));
    p_subgrd2_v = calloc(1,sizeof(int));


    grd3d_reduce_onelayer(
			  nx,
			  ny,
			  nz,
			  p_zcorn_v,
			  p_zcorn2_v,
			  p_actnum_v,
			  p_actnum2_v,
			  &nactive,
			  0,
			  debug);

    ok(nactive==nx*ny, "Number of active cells reduced grid");

    ok(p_actnum2_v[33]==1, "ACTNUM2 of location 33");

    /* export */
    status = mkdir("./TMP", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    strcpy(file,"./TMP/test_grd3d_onelayer.roff");

    grd3d_export_roff_grid (
			    0,
			    nx,
			    ny,
			    1,
			    0,
			    0,
			    450000,
			    6700000,
			    0,
			    p_coord_v,
			    p_zcorn2_v,
			    p_actnum2_v,
			    p_subgrd2_v,
			    file,
			    debug
			    );

    strcpy(file,"./TMP/test_grd3d_onelayer.grdecl");

    grd3d_export_grdecl (
			 nx,
			 ny,
			 1,
			 p_coord_v,
			 p_zcorn2_v,
			 p_actnum2_v,
			 file,
                         1,
			 debug
			 );




    done_testing();

}
