#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int main () {
    int ib, ier, inside, i;
    double points_v[9]={0,1,-7, 3,1,-9, 0,-5,-8}; /* assign at compile time */
    double nvector[4], line_v[6], point_v[3], c[24], myz, res;
    double angle, nvec1[4], nvec2[4], diff;
    int    debug=1;

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);
    ok(3 == 3);

    /*
     * -------------------------------------------------------------------------
     * test ijk2ib
     */

    ib = x_ijk2ib(90,20,3,100,100,20,0);
    printf("IB is %d\n",ib);

    ok(ib==21989,"Test x_ijk2ib");

    /*
     * -------------------------------------------------------------------------
     * Angle between 3D vectors
     */

    nvec1[0] = 0.0; nvec1[1] = 1.0;  nvec1[2] = 4.0;  nvec1[3] = 0.0;
    nvec2[0] = 0.0; nvec2[1] = 1.0;  nvec2[2] = 4.0;  nvec2[3] = 0.0;
    angle = x_angle_vectors(nvec1, nvec2, debug);
    printf("Angle is %f\n", angle);
    ok(angle == 0.0, "Testing angle for paralell lines");

    nvec1[0] = 12.0; nvec1[1] = 13.0;  nvec1[2] = -5.0;  nvec1[3] = 0.0;
    nvec2[0] = 9.0; nvec2[1] = 11.0;  nvec2[2] = -4.0;  nvec2[3] = 0.0;
    angle = x_angle_vectors(nvec1, nvec2, debug);
    printf("Angle is %f\n", angle);
    diff = fabs(angle - 0.0574586656933585);
    ok(diff < 0.00000001, "Testing angle for some lines");

    /*
     * -------------------------------------------------------------------------
     * normal vector equation:
     * Example from http://mathinsight.org/forming_plane_examples
     * (0,1,-7) (3,1,-9) (0,-5,-8) shall give: -12x+3y-18z-129=0
     */

    ier=x_plane_normalvector(points_v, nvector, 0, debug);

    ok(ier==0,"Testing ier from x_plane_normalvector");
    ok(nvector[0]==-12, "A from nvector");
    ok(nvector[1]==3, "B from nvector");
    ok(nvector[2]==-18, "C from nvector");
    ok(nvector[3]==-129, "D from nvector");

    /*
     * -------------------------------------------------------------------------
     * Find an intersection between line and plane
     * http://ocw.mit.edu/courses/mathematics/18-02sc-multivariable-
     * calculus-fall-2010/
     * 1.-vectors-and-matrices/part-c-parametric-equations-for-curves/
     * session-16-intersection-of-a-line-and-a-plane/MIT18_02SC_we_9_comb.pdf
     * Plane is given by 2x+y-4z-4=0, and
     * a) line by (0,2,0) and (3,11,3) shall give (2,8,2)
     * b) line by (0,4,0) and (4,10,3) parallel
     */
    nvector[0]=2; nvector[1]=1; nvector[2]=-4; nvector[3]=-4;
    line_v[0]=0; line_v[1]=2; line_v[2]=0;
    line_v[3]=3; line_v[4]=11; line_v[5]=3;

    ier=x_isect_line_plane(nvector, line_v, point_v, 1, debug);

    ok(ier==0, "Point in plane test (IER)");
    ok(point_v[0]==2, "Point in plane test (X)");
    ok(point_v[1]==8, "Point in plane test (Y)");
    ok(point_v[2]==2, "Point in plane test (Z)");

    /* a paralell line */
    line_v[0]=1; line_v[1]=4; line_v[2]=0;
    line_v[3]=4; line_v[4]=10; line_v[5]=3;
    ier=x_isect_line_plane(nvector, line_v, point_v, 1, debug);

    ok(ier==1, "Line parallel to plane (IER=1)");



    /*
     * -------------------------------------------------------------------------
     * Sample Z from cell top or base
     * Cell 51, 61, 1 from Gullfaks case...
     */
    c[0]=455990;
    c[1]=6.78575e+06;
    c[2]=1885.12;
    c[3]=456083;
    c[4]=6.78575e+06;
    c[5]=1880.75;
    c[6]=455999;
    c[7]=6.78565e+06;
    c[8]=1885.55;
    c[9]=456091;
    c[10]=6.78565e+06;
    c[11]=1879.79;
    c[12]=456003;
    c[13]=6.78574e+06;
    c[14]=1892.58;
    c[15]=456097;
    c[16]=6.78574e+06;
    c[17]=1888.79;
    c[18]=456009;
    c[19]=6.78564e+06;
    c[20]=1892.34;
    c[21]=456104;
    c[22]=6.78564e+06;
    c[23]=1887.47;



    /* point ...*/
    myz = x_sample_z_from_xy_cell(c,456014,6785678,0,0,debug);
    ok((int)myz == 1884, "Sample from cell top");

    /*cell base */
    myz = x_sample_z_from_xy_cell(c,456014,6785678,1,0,debug);
    ok((int)myz == 1892, "Sample from cell base");



    /*
     * -------------------------------------------------------------------------
     * Check point in cell
     */

    for (i=0;i<100000; i++) {
	inside=x_chk_point_in_cell(456014,6785678,1885,c,1,debug);
    }

    ok(inside == 2, "Point in cell");


    /*
     * -------------------------------------------------------------------------
     * Angle conversion routine, NOT FINISHED!
     */
    res=x_rotation_conv(70,2,0,0,debug);
    //printf("%f",res);

    ok(res == 20, "AZI 70 to normal anticlock degrees");

    res=x_rotation_conv(-30,2,0,0,debug);
    //printf("%f",res);

    ok(res == 120, "AZI -30 to degrees");

    done_testing();

}
