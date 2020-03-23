/*
 * ############################################################################
 * Extrapolate between two 3D points
 * x1..z1 first point
 * x2..z2 second point (that will be moved)
 * dlen   the distance to move in 2d/3d; only 2D now!
 * JRIV
 * ############################################################################
 * ############################################################################
 */



#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


int x_vector_extrapol (
                       double x1,
                       double y1,
                       double z1,
                       double *x2,
                       double *y2,
                       double *z2,
                       double dlen,
                       double xang,
                       int   debug
                       )
{
    /* locals */
    char     sub[24]="x_vector_extrapol";
    float    x3, y3, z3, angle, xnew, ynew, newlen;

    xtgverbose(debug);
    xtg_speak(sub,2,"Entering routine");

    x3=*x2;
    y3=*y2;
    z3=*z2;

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if ( x1 == x3 && y1 == y3 && z1 == z3) {
        xtg_speak(sub,2,"Hmmm null length vector");
        return 0;
    }

    xtg_speak(sub,2,"LENGTH to extend is %10.2f",dlen);



    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */

    angle=atan2(y3-y1,x3-x1);

    angle=angle+xang;

    if (cos(angle) > 0.01 || cos(angle) < -0.01) {
        xnew=x3+dlen*cos(angle);
    }
    else{
        xnew=x3+dlen;
        ynew=y3;
    }

    if (sin(angle) > 0.01 || sin(angle) < -0.01) {
        ynew=y3+dlen*sin(angle);
    }
    else{
        ynew=y3+dlen;
        xnew=x3;
    }

    xtg_speak(sub,2,"XY was %10.2f %10.2f",x3,y3);
    xtg_speak(sub,2,"XY --> %10.2f %10.2f",xnew,ynew);

    newlen=sqrt(pow(x3-xnew,2)+pow(y3-ynew,2));
    xtg_speak(sub,2,"Added length is %10.2f",newlen);



    *x2=xnew;
    *y2=ynew;
    *z2=z3;

    return 1;
}
