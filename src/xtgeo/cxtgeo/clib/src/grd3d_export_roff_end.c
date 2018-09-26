/*
 * ############################################################################
 * grd3d_export_roff_end
 * Exporting a Roff end tag (mode=0 for binary)
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_export_roff_end.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_export_roff_end.c,v $ 
 *
 * $Log: grd3d_export_roff_end.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"


    
/*
 * ****************************************************************************
 *                      READ_ROFF_ASCII_GRID
 ****************************************************************************
 * The ROFF (Roxar Open File Format) is like this:
 *
 *roff-asc 
 *#ROFF file#
 *#Creator: RMS - Reservoir Modelling System, version 6.0#
 *tag filedata 
 *int byteswaptest 1           
 *char filetype  "grid"
 *char creationDate  "28/03/2000 16:59:16"
 *endtag 
 *tag version 
 *int major 2           
 *int minor 0           
 *endtag 
 *tag dimensions 
 *int nX 4           
 *int nY 4           
 *int nZ 3           
 *endtag 
 *tag translate 
 *double xoffset   4.62994625E+05
 *double yoffset   5.93379900E+06
 *double zoffset  -3.37518921E+01
 *endtag 
 *tag scale 
 *double xscale   1.00000000E+00
 *double yscale   1.00000000E+00
 *double zscale  -1.00000000E+00
 *endtag 
 *tag subgrids 
 *array int nLayers 16          
 *           1            1            5            8           10           10
 *         10           15            8           20           20           20
 *          8            6            8            2
 *endtag 
 *tag cornerLines 
 *array double data 150         
 * -7.51105194E+01  -4.10773730E+03  -1.86212000E+03  -7.51105194E+01
 * -4.10773730E+03  -1.72856909E+03  -8.36509094E+02  -2.74306006E+03
 *  ....
 *endtag 
 *tag zvalues 
 *array byte splitEnz 100         
 *  1   1   1   1   1   1   4   4   1   1   1   1
 * ....
 *endtag
 *tag active 
 *array bool data 48          
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *endtag 
 *... ETC
 * ----------------------------------------------------------------------------
 *
 */   



void grd3d_export_roff_end (
			    int     mode,
			    char    *filename,
			    int     debug
			    )


{
    
    FILE   *fc;
    char   sub[24]="grd3d_export_roff_end";
    char   ftype[3];
    xtgverbose(debug);

    xtg_speak(sub,2,"==== Entering routine ... ====");   


    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
  
    xtg_speak(sub,2,"Opening ROFF file (append)...");
    strcpy(ftype,"ab");
    if (mode==0) strcpy(ftype,"ab");
    fc=fopen(filename,ftype);
    xtg_speak(sub,2,"Opening ROFF file (append)... DONE!");
    

    xtg_speak(sub,2,"Writing endtag...");

    if (mode>0) {
	fprintf(fc,"tag eof\n");
	fprintf(fc,"endtag\n");
    }
    else{
	fwrite("tag\0eof\0",1,8,fc);
	fwrite("endtag\0",1,7,fc);
    }
    xtg_speak(sub,2,"Writing endtag...DONE!");

    fclose(fc);
    
    xtg_speak(sub,2,"==== Exiting routine ... ====");   

}














