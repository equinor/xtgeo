/*
 * ############################################################################
 * map_operation_map.c
 *
 * Description:
 * Do operations (add, subtract, etc) between map1 and map2, leaving the
 * the result in map 1. The calling routine must check that map1 and map2
 * has equal nx and ny!
 * iop=1  add
 * iop=2  subtract
 * iop=3  multiply
 * iop=4  divide
 * iop=5  set equal (copy)
 * iop=6  set below (force), MISSING: with gap factor
 * Bugs or potential problems:
 * - Test to check that dimensions are equal may be needed
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: map_export_storm_binary.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/map_export_storm_binary.c,v $
 *
 * $Log: $
 *
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

void map_operation_map (
			  int   nx,
			  int   ny,
			  double *p_map1_v,
			  double *p_map2_v,
			  int   iop,
			  double factor,
			  int   debug
			  )
{

    int i, nxy;
    char s[24];

    strcpy(s,"map_operation_map");

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <map_operation_map>...");

    if (iop==1) {
	xtg_speak(s,2,"Adding maps...");
    }
    else if (iop==2) {
	xtg_speak(s,2,"Subtracting maps...");
    }
    else if (iop==3) {
	xtg_speak(s,2,"Multiply maps...");
    }
    else if (iop==4) {
	xtg_speak(s,2,"Divide maps...");
    }
    else if (iop==5) {
	xtg_speak(s,2,"Copy values from map2 to map1...");
    }
    else if (iop==6) {
	xtg_speak(s,2,"Set map1 below map2...");
    }
    else if (iop==7) {
        xtg_speak(s,2,"Set map1 above map2...");
    }
    else if (iop==8) {
	xtg_speak(s,2,"Set map1 values to be equal to map2...");
    }
    else {
	xtg_error(s,"Illegal operation! STOP!");
    }

    nxy = nx * ny;

    for (i=0;i<nxy;i++) {
	if (p_map1_v[i] != UNDEF_MAP &&
	    p_map2_v[i] != UNDEF_MAP) {

	    if (iop==1) {
		p_map1_v[i]=p_map1_v[i] + p_map2_v[i];
	    }
	    if (iop==2) {
		p_map1_v[i]=p_map1_v[i] - p_map2_v[i];
	    }
	    if (iop==3) {
		p_map1_v[i]=p_map1_v[i] * p_map2_v[i];
	    }
	    if (iop==4) {
		/* avoid division on zero */
		if (fabs(p_map2_v[i]) < FLOATEPS) p_map2_v[i]=FLOATEPS;
		p_map1_v[i]=p_map1_v[i] / p_map2_v[i];
	    }
	    if (iop==5) {
		p_map1_v[i]=p_map2_v[i];
	    }

	    /* below */
	    if (iop==6) {
		if ((p_map1_v[i]-factor) <= p_map2_v[i]-FLOATEPS) {
		    p_map1_v[i]=p_map2_v[i]+factor;
		}
	    }

	    /* above */
	    if (iop==7) {
		if ((p_map1_v[i]+factor) >= p_map2_v[i]+FLOATEPS) {
		    p_map1_v[i]=p_map2_v[i]-factor;
		}
	    }
	    if (iop==8) {
	        if (p_map2_v[i] > factor - FLOATEPS && p_map2_v[i]
		    <= factor + FLOATEPS) {
		    p_map1_v[i]=p_map2_v[i];
	        }
	    }
	}

	if (p_map2_v[i] == UNDEF_MAP) {
	    p_map1_v[i] = p_map2_v[i];
	}




    }
    xtg_speak(s,2,"Exiting <map_operation_map>...");
}
