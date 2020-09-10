/*
 * -------------------------------------------------------------------------------------
 * These headers are not not used raw but wrapped in cxtgeo.i
 * -------------------------------------------------------------------------------------
 */

#define _GNU_SOURCE 1

#pragma once

#include "xtg.h"
#include <stdint.h>
#include <stdio.h>

int
grd3cp3d_xtgformat1to2_geom_test(long ncol,
                                 long nrow,
                                 long nlay,
                                 double *coordsv1,
                                 double *zcornsv1,  // notice float type
                                 int *actnumsv1,
                                 double *coordsv2,
                                 float *zcornsv2,  // notice float type
                                 int *actnumsv2);
