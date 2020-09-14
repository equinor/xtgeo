/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_export_roff_bin_start_end.c
 *
 * DESCRIPTION:
 *    Export a ROFF header or endings only, BINARY files.
 *    This routine for 'start' must be ran before the actual exports,
 *    and 'end' after.
 *
 * ARGUMENTS:
 *    option         i     0: start, 1 end
 *    ncol..nlay     i     Dimensions
 *    fc             i     File handle (file open / close muts be handled by caller)
 *
 * RETURNS:
 *    Void
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <sys/types.h>
#include <time.h>

void
grdcp3d_export_roff_bin_start_end(int option, long ncol, long nrow, long nlay, FILE *fc)
{

    if (option == 0) {
        int myint;
        char timestring[32];
        time_t now;

        logger_info(LI, FI, FU, "Binary roff export, start session...");
        now = time(NULL);
        strcpy(timestring, ctime(&now));
        timestring[strlen(timestring) - 1] = '\0';

        fwrite("roff-bin\0", 1, 9, fc);
        fwrite("#ROFF file#\0", 1, 12, fc);
        fwrite("#Creator: CLib subsystem of XTGeo#\0", 1, 35, fc);
        fwrite("tag\0filedata\0", 1, 13, fc);
        fwrite("int\0byteswaptest\0", 1, 17, fc);
        myint = 1;
        fwrite(&myint, 4, 1, fc);
        fwrite("char\0filetype\0parameter\0", 1, 24, fc);
        fwrite("char\0creationDate\0", 1, 18, fc);
        fwrite(timestring, 1, strlen(timestring) + 1, fc);
        fwrite("endtag\0", 1, 7, fc);
        fwrite("tag\0version\0", 1, 12, fc);
        fwrite("int\0major\0", 1, 10, fc);
        myint = 2;
        fwrite(&myint, 4, 1, fc);
        fwrite("int\0minor\0", 1, 10, fc);
        myint = 0;
        fwrite(&myint, 4, 1, fc);
        fwrite("endtag\0", 1, 7, fc);
        fwrite("tag\0dimensions\0", 1, 15, fc);
        fwrite("int\0nX\0", 1, 7, fc);
        myint = (int)ncol;
        fwrite(&myint, 4, 1, fc);
        fwrite("int\0nY\0", 1, 7, fc);
        myint = (int)nrow;
        fwrite(&myint, 4, 1, fc);
        fwrite("int\0nZ\0", 1, 7, fc);
        myint = (int)nlay;
        fwrite(&myint, 4, 1, fc);
        fwrite("endtag\0", 1, 7, fc);
        logger_info(LI, FI, FU, "Binary roff export, start session... done");

    } else {
        logger_info(LI, FI, FU, "Binary roff export, ending session...");
        fprintf(fc, "tag eof\n");
        fprintf(fc, "endtag\n");
        logger_info(LI, FI, FU, "Binary roff export, ending session... done");
    }
}
