/* test map vs grd3d operations */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

int main () {
    int ib, ier;
    char file[70];
    int  ndef, ndefsum, mx, my, debug=1;
    double xstep, ystep, xori, xmin, xmax, yori, ymin, ymax, zmin, zmax;
    double *zval_v, *zval2_v;
    int    numact, numsubs, nx, ny, nz;
    int    *p_actnum_v, *p_subgrd_v;
    double *p_coord_v, *p_zcorn_v;


    /* need to know (perl or python routine will scan header in file) */
    mx=523;
    my=601;

    zval_v=calloc(mx*my,sizeof(double));

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    /*
     * -------------------------------------------------------------------------
     * Read an existing map
     */

    strcpy(file,"../../../../../../xtgeo-testdata/surfaces/eme/1/emerald_z1735.grd");

    map_import_storm_binary(&ndef, &ndefsum, &mx, &my, &xstep, &ystep,
			    &xmin, &xmax, &ymin, &ymax, &zmin, &zmax,
			    zval_v, file,
			    &ier, debug);

    ok(xstep==25.0,"XSTEP map");
    ok(ystep==25.0,"YSTEP map");
    printf("%f\n", zmin);
    ok(fabs(zmin - 1734.999146) < 0.001, "ZMIN map");
    ok(mx==307,"MX map");
    ok(my==331,"MY map");

    /*
     * ------------------------------------------------------------------------
     * Create a new map with all UNDEF values
     */
    mx=1000;
    my=1200;
    xori=xmin-200;
    yori=ymin-200;
    xstep=10;
    ystep=10;

    /* allocate pointer */
    zval2_v=calloc(mx*my,sizeof(double));


    map_create(mx, my, xori, xstep, yori, ystep, 0.0,
	       zval2_v, 99.9, 0, debug);

    printf("ZVAL: %f\n",zval2_v[0]);
    ok((int)zval2_v[0]==99,"Value of created map (1)");


    /*
     * ------------------------------------------------------------------------
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

    ok((int)(p_zcorn_v[0]+0.5) == 1726, "ZCORN approx of location 0");


    /*
     * -------------------------------------------------------------------------
     * Sample a map from grid layer top
     */

    printf("Sample map\n");
    map_sample_grd3d_lay(nx, ny, nz, p_coord_v, p_zcorn_v, p_actnum_v,
			 1, mx, my, xori, xstep, yori, ystep,
			 zval2_v, 0, debug);
    printf("Sample map done\n");

    /* replace all UNDEF with 2000 */
    map_operation_value(5,mx,my,zval2_v,2000,UNDEF,UNDEF,debug);

    /*
     * -------------------------------------------------------------------------
     * export case
     */

    mkdir("TMP",0777);
    strcpy(file,"TMP/test_map_grd3d_01.grd");

    xmax=xori+xstep*(mx-1);
    ymax=yori+ystep*(my-1);

    map_export_storm_binary(mx, my, xstep, ystep, xori, xmax, yori, ymax,
			    zval2_v, file, debug);


    done_testing();

}
