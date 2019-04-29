/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_export_roff_psta.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export a ROFF header only, ASCII or BINARY
 *    This routine must preceede the grd3d_export_roff_prop or
 *    grd3d_export_roff_grid(?).
 *
 * ARGUMENTS:
 *    mode           i     0 for binary, 1 for ASCII
 *    nx..nz         i     Dimensions
 *    filename       i     File name to output to
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void
 *
 * TODO/ISSUES/BUGS:
 *    Make proper return codes
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                   ROFF FORMAT, see grd3d_export_roff_prop.c
 ******************************************************************************
 */

void grd3d_export_roff_pstart (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *filename,
			      int     debug
			      )

{
    int    myint;
    char   timestring[32];
    time_t now;
    FILE   *fc;
    char   s[24]="grd3d_exp.._roff_psta";

    xtgverbose(debug);

    xtg_speak(s, 2, "Entering %s", s);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

     xtg_speak(s,2,"Opening ROFF file...");
     if (mode==0) {
	 fc=fopen(filename,"wb");
     }
     else{
	 fc=fopen(filename,"wb"); /* work for output Unix Ascii on Win */
     }

     xtg_speak(s,2,"Opening ROFF file...DONE!");

    /*
     *-------------------------------------------------------------------------
     * Header of file. Most of this is presently hardcoded
     *-------------------------------------------------------------------------
     */


     now=time(NULL);
     strcpy(timestring,ctime(&now));
     timestring[strlen(timestring)-1]='\0';


     if (mode>0) {
	 fprintf(fc,"roff-asc\n");
	 fprintf(fc,"#ROFF file#\n");
	 fprintf(fc,"#Creator: CLib subsystem of XTGeo#\n");
	 fprintf(fc,"tag filedata\n");
	 fprintf(fc,"int byteswaptest 1\n");
	 fprintf(fc,"char filetype \"parameter\"\n");
	 fprintf(fc,"char creationDate \"%s\"\n",timestring);
	 fprintf(fc,"endtag\n");
	 fprintf(fc,"tag version\n");
	 fprintf(fc,"int major 2\n");
	 fprintf(fc,"int minor 0\n");
	 fprintf(fc,"endtag\n");
	 fprintf(fc,"tag dimensions\n");
	 fprintf(fc,"int nX %d\n", nx);
	 fprintf(fc,"int nY %d\n", ny);
	 fprintf(fc,"int nZ %d\n", nz);
	 fprintf(fc,"endtag\n");
     }
     else{
	 fwrite("roff-bin\0",1,9,fc);
	 fwrite("#ROFF file#\0",1,12,fc);
	 fwrite("#Creator: CLib subsystem of XTGeo#\0", 1, 35, fc);
	 fwrite("tag\0filedata\0",1,13,fc);
	 fwrite("int\0byteswaptest\0",1,17,fc); myint=1; fwrite(&myint,4,1,fc);
	 fwrite("char\0filetype\0parameter\0",1,24,fc);
	 fwrite("char\0creationDate\0",1,18,fc);
	 fwrite(timestring,1,strlen(timestring)+1,fc);
	 fwrite("endtag\0",1,7,fc);
	 fwrite("tag\0version\0",1,12,fc);
	 fwrite("int\0major\0",1,10,fc); myint=2; fwrite(&myint,4,1,fc);
	 fwrite("int\0minor\0",1,10,fc); myint=0; fwrite(&myint,4,1,fc);
	 fwrite("endtag\0",1,7,fc);
	 fwrite("tag\0dimensions\0",1,15,fc);
	 fwrite("int\0nX\0",1,7,fc); myint=nx; fwrite(&myint,4,1,fc);
	 fwrite("int\0nY\0",1,7,fc); myint=ny; fwrite(&myint,4,1,fc);
	 fwrite("int\0nZ\0",1,7,fc); myint=nz; fwrite(&myint,4,1,fc);
	 fwrite("endtag\0",1,7,fc);
     }

     fclose(fc);

     xtg_speak(s, 2,"Exiting %s", s);
}
