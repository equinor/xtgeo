

/*
 ******************************************************************************
 *
 * Write an Eclipse input ASCII record to file, free form for small snippets
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_write_free eclinput.c
 *
 *
 * DESCRIPTION:
 *    Feed data from XTGeo to the Eclipse input format
 *
 * ARGUMENTS:
 *    fc               i     Filehandle (file must be open)
 *    recname          i     Name of record to write
 *    freetext         i     Free text to write
 *
 * RETURNS:
 *    Void
 *
 */

void
grd3d_write_free_eclinput(FILE *fc, char *recname, char *freetext)
{

    fprintf(fc, "%-8s\n", recname);
    fprintf(fc, "%s\n", freetext);
}
