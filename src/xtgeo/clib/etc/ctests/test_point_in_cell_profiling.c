#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    int ib, ier, inside, i;
    double nvector[4], line_v[6], point_v[3], c[24], myz;
    int    debug=1, nx, ny, nz, numact, numsubs;
    int    *actnumsv, *p_subgrd_v, ns;
    double *coordsv, *zcornsv;
    char file[70];
    double x, y, z, zadd;
    int  ibstart;


    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);

    strcpy(file,"../../../../../../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff");

    nx=40;
    ny=64;
    nz=14;

    /* allocate */
    coordsv = calloc((nx+1)*(ny+1)*2*3,sizeof(double));
    zcornsv = calloc(nx*ny*(nz+1)*4,sizeof(double));
    actnumsv = calloc(nx*ny*nz,sizeof(int));
    p_subgrd_v = calloc(1,sizeof(int));


    printf("Reading grid\n");
    grd3d_import_roff_grid(&numact, &numsubs, coordsv, zcornsv,
			   actnumsv, p_subgrd_v, 1, file, debug);

    printf("Reading grid done\n");

    ibstart=100;
    x=456640.7;
    y=5929000.00;
    z=1553.07;
    zadd=0.1;

    /* for (i=0;i<10000;i++) { */
    /*     z=z+zadd; */
    /*     ib=grd3d_point_in_cell( */
    /*     		       ibstart, */
    /*     		       0, */
    /*     		       x, */
    /*     		       y, */
    /*     		       z, */
    /*     		       nx, */
    /*     		       ny, */
    /*     		       nz, */
    /*     		       coordsv, */
    /*     		       zcornsv, */
    /*     		       actnumsv, */
    /*     		       5, */
    /*     		       1, */
    /*     		       &ns, */
    /*     		       0, */
    /*     		       debug */
    /*     		       ); */

    /*     if (ib>=0) { */
    /*         ibstart=ib; */
    /*     } */
    /*     if (ib<0) { */
    /*         zadd=-1*zadd; */
    /*     } */
    /*     if (i%100 == 0) { */
    /*         printf("Loop is %d and z is %f and ib is %d and ibstart is %d\n" */
    /*     	   "Nsearch is %d\n", */
    /*     	   i,z,ib, ibstart, ns); */
    /*     } */
    /* } */



    done_testing();

}
