/*
 * ############################################################################
 * x_nint.c
 * Getting the nearest integer of a double float, similar to Fortrans NINT
 * Author: J.C. Rivenaes
 * ############################################################################
 *
 */

#define _ISOC99_SOURCE
#include "libxtg_.h"
#include <math.h>

int x_nint(double value) {

    double near;

    near = nearbyint(value);

    return (int)near;
}
