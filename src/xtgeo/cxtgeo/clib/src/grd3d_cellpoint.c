/*
 * ############################################################################
 * Calculating XYZ point for a cell, at edges, which is determined by a fault
 * code. The fault codes are:
 *
 * 0   = none, use cell midpoint
 * 1   = I-
 * 2   = I+
 * 3   = J-
 * 4   = J+
 * 5   = K-
 * 6   = K+
 *
 * 11   = I-J-
 * 12   = I-J+
 * 13   = I+J-
 * 14   = I+J+
 * 21   = I-K-
 * 22   = I-K+
 * 23   = I+K-
 * 24   = I+K+
 * 31   = J-K-
 * 32   = J-K+
 * 33   = J+K-
 * 34   = J+K+
 *
 * 41   = I-J-K-
 * 42   = I-J-K+
 * 43   = I-J+K-
 * 44   = I-J+K+
 * 45   = I+J-K-
 * 46   = I+J-K+
 * 47   = I+J+K-
 * 48   = I+J+K+
 
 * confer XTGeo/Zone/_Etc.pm
 *
 * CELL CORNERS is a 24 vector long (x y z x y z ....)
 *     ---> I
 *
 *    0 1 2  ------------------ 3 4 5     12 13 14  ------------------ 15 16  17         
 *           |                |		            |                |	      
 *           |     TOP        |		            |     BOT        |	      
 *           |                |		            |                |	      
 *    6 7 8  ----------------- 9 10 11	  18 19 20  -----------------  21 22 23
 *
 *     |
 *     v J
 
 * Author: JRIV
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_cellpoint (
		     int     i,
		     int     j,
		     int     k,
		     int     nx,
		     int     ny,
		     int     nz,
		     int     fcode,
		     double   *p_coord_v,
		     double   *p_zgrd3d_v,
		     double   *x,
		     double   *y,
		     double   *z,
		     int     debug
		     )


{
    double c[24];
    char   s[24]="grd3d_cellpoint";
    
    xtgverbose(debug);
    
    xtg_speak(s,4,"==== Entering routine ====");

    
    /* get all 24 corners */
    grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zgrd3d_v,c,debug);

    /* find the cellpoint depending on geometry given by fcode...*/
       
    if (fcode==1) {
	*x=0.25*(c[0]+c[6]+c[12]+c[18]);
	*y=0.25*(c[1]+c[7]+c[13]+c[19]);
	*z=0.25*(c[2]+c[8]+c[14]+c[20]);
    }
    else if (fcode==2) {
	*x=0.25*(c[3]+c[9]+c[15]+c[21]);
	*y=0.25*(c[4]+c[10]+c[16]+c[22]);
	*z=0.25*(c[5]+c[11]+c[17]+c[23]);
    }
    else if (fcode==3) {
	*x=0.25*(c[0]+c[3]+c[12]+c[15]);
	*y=0.25*(c[1]+c[4]+c[13]+c[16]);
	*z=0.25*(c[2]+c[5]+c[14]+c[17]);
    }
    else if (fcode==4) {
	*x=0.25*(c[6]+c[9]+c[18]+c[21]);
	*y=0.25*(c[7]+c[10]+c[19]+c[22]);
	*z=0.25*(c[8]+c[11]+c[20]+c[23]);
    }
    else if (fcode==5) {
	*x=0.25*(c[0]+c[3]+c[6]+c[9]);
	*y=0.25*(c[1]+c[4]+c[7]+c[10]);
	*z=0.25*(c[2]+c[5]+c[8]+c[11]);
    }
    else if (fcode==6) {
	*x=0.25*(c[12]+c[15]+c[18]+c[21]);
	*y=0.25*(c[13]+c[16]+c[19]+c[22]);
	*z=0.25*(c[14]+c[17]+c[20]+c[23]);
    }

    /* 11,12,13,14 is a bent corner face but take weighted average */ 

    else if (fcode==11) {
	*x=0.3*(c[0]+c[12])+0.1*(c[6]+c[18]+c[3]+c[15]);
	*y=0.3*(c[1]+c[13])+0.1*(c[7]+c[19]+c[4]+c[16]);
	*z=0.3*(c[2]+c[14])+0.1*(c[8]+c[20]+c[5]+c[17]);
    }
    else if (fcode==13) {
	
	*x=0.3*(c[3]+c[15])+0.1*(c[0]+c[12]+c[9]+c[21]);
	*y=0.3*(c[4]+c[16])+0.1*(c[1]+c[13]+c[10]+c[22]);
	*z=0.3*(c[5]+c[17])+0.1*(c[2]+c[14]+c[11]+c[23]);
    }
    else if (fcode==12) {
	*x=0.3*(c[6]+c[18])+0.1*(c[0]+c[12]+c[9]+c[21]);
	*y=0.3*(c[7]+c[19])+0.1*(c[1]+c[13]+c[10]+c[22]);
	*z=0.3*(c[8]+c[20])+0.1*(c[2]+c[14]+c[11]+c[23]);
    }
    else if (fcode==14) {
	*x=0.3*(c[9]+c[21])+0.1*(c[6]+c[18]+c[3]+c[15]);
	*y=0.3*(c[10]+c[22])+0.1*(c[7]+c[19]+c[4]+c[16]);
	*z=0.3*(c[11]+c[23])+0.1*(c[8]+c[20]+c[5]+c[17]);
    }


    else if (fcode==21) {
	*x=0.5*(c[0]+c[6]);
	*y=0.5*(c[1]+c[7]);
	*z=0.5*(c[2]+c[8]);
    }
    else if (fcode==22) {
	*x=0.5*(c[12]+c[18]);
	*y=0.5*(c[13]+c[19]);
	*z=0.5*(c[14]+c[20]);
    }
    else if (fcode==23) {
	*x=0.5*(c[3]+c[9]);
	*y=0.5*(c[4]+c[10]);
	*z=0.5*(c[5]+c[11]);
    }
    else if (fcode==24) {
	*x=0.5*(c[15]+c[21]);
	*y=0.5*(c[16]+c[22]);
	*z=0.5*(c[17]+c[23]);
    }


    else if (fcode==31) {
	*x=0.5*(c[0]+c[3]);
	*y=0.5*(c[1]+c[4]);
	*z=0.5*(c[2]+c[5]);
    }
    else if (fcode==32) {
	*x=0.5*(c[12]+c[15]);
	*y=0.5*(c[13]+c[16]);
	*z=0.5*(c[14]+c[17]);
    }
    else if (fcode==33) {
	*x=0.5*(c[6]+c[9]);
	*y=0.5*(c[7]+c[10]);
	*z=0.5*(c[8]+c[11]);
    }
    else if (fcode==34) {
	*x=0.5*(c[18]+c[21]);
	*y=0.5*(c[19]+c[22]);
	*z=0.5*(c[20]+c[23]);
    }

    else if (fcode==41) {
	*x=1.0*(c[0]);
	*y=1.0*(c[1]);
	*z=1.0*(c[2]);
    }

    else if (fcode==42) {
	*x=1.0*(c[12]);
	*y=1.0*(c[13]);
	*z=1.0*(c[14]);
    }

    else if (fcode==43) {
	*x=1.0*(c[6]);
	*y=1.0*(c[7]);
	*z=1.0*(c[8]);
    }

    else if (fcode==44) {
	*x=1.0*(c[18]);
	*y=1.0*(c[19]);
	*z=1.0*(c[20]);
    }

    else if (fcode==45) {
	*x=1.0*(c[3]);
	*y=1.0*(c[4]);
	*z=1.0*(c[5]);
    }

    else if (fcode==46) {
	*x=1.0*(c[15]);
	*y=1.0*(c[16]);
	*z=1.0*(c[17]);
    }


    else if (fcode==47) {
	*x=1.0*(c[9]);
	*y=1.0*(c[10]);
	*z=1.0*(c[11]);
    }

    else if (fcode==48) {
	*x=1.0*(c[21]);
	*y=1.0*(c[22]);
	*z=1.0*(c[23]);
    }

    else {
	*x=0.125*(c[0]+c[3]+c[6]+c[9]+c[12]+c[15]+c[18]+c[21]);
	*y=0.125*(c[1]+c[4]+c[7]+c[10]+c[13]+c[16]+c[19]+c[22]);
	*z=0.125*(c[2]+c[5]+c[8]+c[11]+c[14]+c[17]+c[20]+c[23]);
    }

	



    xtg_speak(s,4,"==== Exiting ====");

}


