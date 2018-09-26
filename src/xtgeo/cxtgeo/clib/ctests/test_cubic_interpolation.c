/*
 * test Cubic interpolation (trilinear) algorithm
 */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    int ib, ier, inside, i;
    double x_v[8], y_v[8], z_v[8];
    float p_v[8], value;
    double x=0, y=0, z=0;

    int debug = 3;
    xtgverbose(debug);
    xtg_verbose_file("NONE");

    plan(NO_PLAN);

    x_v[0] = 0;  x_v[1] = 100; x_v[2] = 0;    x_v[3] = 100;
    y_v[0] = 0;  y_v[1] = 0;   y_v[2] = 100;  y_v[3] = 100;
    z_v[0] = 0;  z_v[1] = 0;   z_v[2] = 0;    z_v[3] = 0;
    p_v[0] = 30; p_v[1] = 30;  p_v[2] = 30;   p_v[3] = 30;

    x_v[4] = 0;    x_v[5] = 100;  x_v[6] = 0;    x_v[7] = 100;
    y_v[4] = 0;    y_v[5] = 0;    y_v[6] = 100;  y_v[7] = 100;
    z_v[4] = 100; z_v[5] = 100;  z_v[6] = 100;  z_v[7] = 100;
    p_v[4] = 50;   p_v[5] = 50;   p_v[6] = 50;   p_v[7] = 50;

    /* now the point in the middle should give 40 ...*/

    x = 50; y = 50; z = 50;
    ier = x_interp_cube_nodes(x_v, y_v, z_v, p_v, x, y, z, &value, 1, debug);

    printf("Value is %f and ier is %d\n", value, ier);

    ok(fabs(value - 40) < 0.01, "Cube interpol 1");


    // try point outside:
    x = 150; y = 50; z = 75;
    ier = x_interp_cube_nodes(x_v, y_v, z_v, p_v, x, y, z, &value, 1, debug);

    printf("Value is %f and ier is %d\n", value, ier);
    ok(ier == -1, "Outside");

    // try point at border, a bit higher --> 45:
    x = 100; y = 50; z = 75;
    ier = x_interp_cube_nodes(x_v, y_v, z_v, p_v, x, y, z, &value, 1, debug);

    printf("Value is %f and ier is %d\n", value, ier);
    ok(fabs(value - 45) < 0.001, "Cube interpol 2");


    done_testing();

}
