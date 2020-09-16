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
 *    option         i     0: start, 1: end
 *    mode           i     0: binary, 1: ascii
 *    rtype          i     "grid" or "parameter"
 *    ncol..nlay     i     Dimensions
 *    fc             i     File handle (file open / close must be handled by caller)
 *
 * RETURNS:
 *    Void
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "logger.h"
#include "roffstuff.h"
#include <sys/types.h>
#include <time.h>

void
grdcp3d_export_roff_bin_start_end(int option,
                                  int mode,
                                  char *rtype,
                                  long ncol,
                                  long nrow,
                                  long nlay,
                                  FILE *fc)
{
    if (option == 0) {
        int myint;
        char timestring[32];
        time_t now;

        logger_info(LI, FI, FU, "Binary or ascii roff export, start session...");
        now = time(NULL);
        strcpy(timestring, ctime(&now));
        timestring[strlen(timestring) - 1] = '\0';

        if (mode == 0) {
            strwrite(mode, "roff-bin$", fc);
        } else {
            strwrite(mode, "roff-asc$", fc);
        }
        strwrite(mode, "#ROFF file#$", fc);
        strwrite(mode, "#Creator: CLib subsystem of XTGeo#$", fc);
        strwrite(mode, "tag^filedata$", fc);
        strwrite(mode, "int^byteswaptest^", fc);
        myint = 1;
        intwrite(mode, myint, fc);

        if (mode == 0 && strncmp(rtype, "grid", 4) == 0) {
            strwrite(mode, "char^filetype^grid$", fc);
        } else if (mode == 0 && strncmp(rtype, "para", 4) == 0) {
            strwrite(mode, "char^filetype^parameter$", fc);
        } else if (mode == 1 && strncmp(rtype, "grid", 4) == 0) {
            strwrite(mode, "char^filetype^\"grid\"$", fc);
        } else {
            strwrite(mode, "char^filetype^\"parameter\"$", fc);
        }

        strwrite(mode, "char^creationDate^\"", fc);
        strwrite(mode, timestring, fc);
        strwrite(mode, "\"$", fc);
        strwrite(mode, "endtag$", fc);
        strwrite(mode, "tag^version$", fc);
        strwrite(mode, "int^major^", fc);
        intwrite(mode, 2, fc);
        strwrite(mode, "int^minor^", fc);
        intwrite(mode, 0, fc);
        strwrite(mode, "endtag$", fc);
        strwrite(mode, "tag^dimensions$", fc);
        strwrite(mode, "int^nX^", fc);
        myint = (int)ncol;
        intwrite(mode, myint, fc);
        strwrite(mode, "int^nY^", fc);
        myint = (int)nrow;
        intwrite(mode, myint, fc);
        strwrite(mode, "int^nZ^", fc);
        myint = (int)nlay;
        intwrite(mode, myint, fc);
        strwrite(mode, "endtag$", fc);
        logger_info(LI, FI, FU, "Binary roff export, start session... done");

    } else {
        logger_info(LI, FI, FU, "Binary roff export, ending session...");
        strwrite(mode, "tag^eof$", fc);
        strwrite(mode, "endtag$", fc);
        logger_info(LI, FI, FU, "Binary roff export, ending session... done");
    }
}
