/*
 * #################################################################################################
 * Name:      grd3d_cellfault_throws.c
 * Author:    JRIV@statoil.com
 * Created:   2015-10-13
 * Updates:   
 * #################################################################################################
 * Compute fault throw bewteen two neighbour cells
 *
 * Arguments:
 *     i j k            cell to evaluate
 *     nx..nz           grid dimensions
 *     k                common K
 *     p_coord_v        COORD array pointer
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ACTNUM array (pointer)
 *     throws           array of 8 entries [0..7], showing faults around a cell. 
 *                      Edge has value very_large
 *     option           if option=1, ignore actnum
 *     debug            debug/verbose flag
 *
 * Return status: 1 is OK
 * 
 * Here is the geometry/topology fo the fault array that is returned:
 *
 *  ^j      2|      3      | 4
 *  |   -----|-------------|------
 *  |        |             |
 *        1  |     i,j     | 5 (i+1,j)
 *           |             |
 *      -----|-------------|------
 *          0|      7      |6
 *  i-1,j-1  |             |             ---->i
 *
 * j
 * ^
 * |   3       4
 *     --------
 *     |      |  Assuming an unrotated cell:
 *     |      |  Layout for corners, here top. Bottom 5..8 in same order
 *     --------
 *    1        2   ---> i
 *
 * Array (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 x5 y5 z5 x6 y6 z6 x7 y7 z7 x8 y8 z8)
 *         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 * z:            2        5        8       11       14       17       20       23
 * 
 * Caveeats/issues:
 *     - Need test for edge cells 
 *     - This is a corner point evaluation, and stair cases are not dealt with. 
 *       more checks that cells are adjacent
 * #################################################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_cell_faultthrows(
			   int   i,
			   int   j,
			   int   k,
			   int   nx,
			   int   ny,
			   int   nz,
			   double *p_coord_v,
			   double *p_zcorn_v,
			   int   *p_actnum_v,
			   double throw[],
			   int   option,
			   int   debug
			   )
    
{
    /* locals */
    char      s[24]="grd3d_cell_faultthrows";
    double    corners[24], ncorners[24], tthrow, bthrow;
    int       im;
    
    
    /* get the corners of the host cell */
    
    grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,corners,debug);
    
    /*
     *----------------------------------------------------------------------------------------------
     * evaluate corner 0 (i-1, j-1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i-1,j-1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=corners[2]  - ncorners[11];
    bthrow=corners[14] - ncorners[23];
    
    throw[0]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Corner 0 i-1, j-1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 

    /*
     *----------------------------------------------------------------------------------------------
     * evaluate side 1 (i-1, j) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i-1,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=0.5*((corners[2] - ncorners[5])  +  (corners[8] -ncorners[11]));
    bthrow=0.5*((corners[14]-ncorners[17])  +  (corners[20]-ncorners[23]));

    throw[1]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Side 1 i-1, j: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "   %d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 


    /*
     *----------------------------------------------------------------------------------------------
     * evaluate corner 2 (i-1, j+1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i-1,j+1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=corners[8]  - ncorners[5];
    bthrow=corners[20] - ncorners[17];
    
    throw[2]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Corner 2 i-1, j+1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "   %d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	    xtg_speak(s,3, "   %d: Throw is c3 vs c2: %6.2f: ",im, throw[2]);
	}
    }
 



    /*
     *----------------------------------------------------------------------------------------------
     * evaluate side 3 (i, j+1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i,j+1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=0.5*((corners[8] - ncorners[2])  +  (corners[11] -ncorners[5]));
    bthrow=0.5*((corners[20]-ncorners[14])  +  (corners[23]-ncorners[17]));

    throw[3]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Side 3 i, j+1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 


    /*
     *----------------------------------------------------------------------------------------------
     * evaluate corner 4 (i+1, j+1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i+1,j+1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=corners[11]  - ncorners[2];
    bthrow=corners[23] - ncorners[14];
    
    throw[4]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Corner 4 i+1, j+1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 


    /*
     *----------------------------------------------------------------------------------------------
     * evaluate side 5 (i+1, j) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i+1,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=0.5*((corners[11]-ncorners[8])   +  (corners[5] -ncorners[2]));
    bthrow=0.5*((corners[23]-ncorners[20])  +  (corners[17]-ncorners[14]));

    throw[5]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Side 5 i, j+1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 


    /*
     *----------------------------------------------------------------------------------------------
     * evaluate corner 6 (i+1, j-1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i+1,j-1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=corners[5]  - ncorners[8];
    bthrow=corners[17] - ncorners[20];
    
    throw[6]=0.5*(tthrow+bthrow);

    if (debug>2) {
	xtg_speak(s,3, "Corner 6 i+1, j-1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 


    /*
     *----------------------------------------------------------------------------------------------
     * evaluate side 7 (i, j-1) 
     *----------------------------------------------------------------------------------------------
     */
    
    grd3d_corners(i,j-1,k,nx,ny,nz,p_coord_v,p_zcorn_v,ncorners,debug);
    
    tthrow=0.5*((corners[2]-ncorners[8])    +  (corners[5] -ncorners[11]));
    bthrow=0.5*((corners[14]-ncorners[20])  +  (corners[17]-ncorners[23]));

    throw[7]=0.5*(tthrow+bthrow);


    if (debug>2) {
	xtg_speak(s,3, "Side 7 i, j-1: ");
	for (im=2;im<24;im+=3) {
	    xtg_speak(s,3, "%d: This Z coord: %6.2f:         Neig Z coord %6.2f", 
		      im, corners[im], ncorners[im]);
	}
    }
 

    

    
	

    xtg_speak(s,2,"Exiting <%s>",s);
    return(1);
}


