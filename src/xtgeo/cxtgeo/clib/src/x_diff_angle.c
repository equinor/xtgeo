/*
 * ############################################################################
 * x_diff_angle.c
 *
 * Description:
 * Finds the smallest difference between two angles, taking the circle into
 * account. This routine thinks clockwise direction
 *
 * Arguments:
 * ang1           Requested angle
 * ang2           Actual angle
 * option         1 if degrees, otherwise radians
 * debug          Verbose flag
 *
 * Returns:
 * Angle difference (with sign): Requested - Actual
 *
 * Actual + Diff = Requested
 *
 * Examples (degrees): Requested=30, Actual=40, result shall be -10
 *                     Requested=360, Actual=340, result shall be 20
 *                     Requested=360, Actual=170, result shall be -170
 *
 * Author: J.C. Rivenaes JRIV@statoil.com
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

// OLD:
/* double x_diff_angle ( */
/* 		     double ang1, */
/* 		     double ang2, */
/* 		     int    option, */
/* 		     int    debug */
/* 		    ) */
/* { */
/*     /\* locals *\/ */
/*     char     s[24]="x_diff_angle"; */
/*     double   a1, a2, result; */

/*     xtgverbose(debug); */
/*     xtg_speak(s,3,"Entering routine %s",s); */

/*     if (option==1) { */
/* 	/\* angles and result shall be in degrees, not radians; convert *\/ */
/* 	a1=ang1*PI/180; */
/* 	a2=ang2*PI/180; */
/*     } */
/*     else{ */
/* 	a1=ang1; */
/* 	a2=ang2; */
/*     } */

/*     /\* */
/*      * ------------------------------------------------------------------------ */
/*      * Compute the result; case 1 angles are identical */
/*      * ------------------------------------------------------------------------ */
/*      *\/ */

/*     result=a1-a2; */

/*     if (result>PI) result=result-2*PI; */
/*     if (result<PI*-1) result=result+2*PI; */


/*     if (option==1){ */
/* 	result=result*180/PI; */
/* 	if (fabs(result)>180) { */
/* 	    xtg_error(s,"Something went wrong in %s, contact JRIV",s); */
/* 	} */
/*     } */
/*     else{ */
/* 	if (fabs(result)>PI) { */
/* 	    xtg_error(s,"Something went wrong in %s, contact JRIV",s); */
/* 	} */
/*     } */

/*     xtg_speak(s,3,"Difference between A1= %7.3f and A2= %7.3f is: %7.3f", */
/* 	      a1,a2,result); */


/*     return(result); */
/* } */


double x_diff_angle (
                     double ang1,
                     double ang2,
                     int option,
                     int debug
                     )
/* https://rosettacode.org/wiki/Angle_difference_between_two_bearings#C++ */
{
    double full=2*M_PI, half=M_PI;
    double diff;

    if (option == 1) {
        full = 360.0;
        half = 180.0;
    }

    diff = fmod(ang2 - ang1, full);
    if (diff < -half) diff += full;
    if (diff > half) diff -= full;

    return diff;
}
