/*
****************************************************************************************
*
* NAME:
*    grd3d_export_roff_end.c
*
* DESCRIPTION:
*    Export ROFF end marker (mode=0 for binary)
*
* ARGUMENTS:
*    mode              i     0 for binary format
*    filename          i     File name
*
* RETURNS:
*    Void function
*
* LICENCE:
*    CF. XTGeo license
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"

void
grd3d_export_roff_end(int mode, char *filename)

{

    FILE *fc;

    fc = fopen(filename, "ab");

    if (mode == 0) {
        fwrite("tag\0eof\0", 1, 8, fc);
        fwrite("endtag\0", 1, 7, fc);
    } else {
        fprintf(fc, "tag eof\n");
        fprintf(fc, "endtag\n");
    }

    fclose(fc);
}
