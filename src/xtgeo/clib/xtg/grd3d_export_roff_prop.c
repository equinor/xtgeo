/*
****************************************************************************************
*
* NAME:
*    grd3d_export_roff_prop.c
*
* DESCRIPTION:
*    Export a property in ROFF ASCII or BINARY form
*    This routine must follow the grd3d_export_roff_pstart or
*    grd3d_export_roff_grid. The grd3d_export_roff_end must come after.
*    Author: JCR
*
* ARGUMENTS:
*    mode              i     0 for binary, 1 for ASCII
*    nx..nz            i     Dimensions
*    num_subgrds       i     Number of subgrids (use 0 if no subgrids)
*    isubgrd_to_export i     Subgrid to export (use 0 for all grid)
*    p_subgrd_v        i     Subgrid index array
*    pname             i     Name of property
*    ptype             i     Property type
*    p_int_v           i     Property array (if type INT)
*    p_double_v        i     Property array (if type FLOAT/DOUBLE)
*    ncodes            i     Number of codes (if INT)
*    codenames         i     if int: array of chars divided with | -> strings
*    codevalues        i     array of code values
*    filename          i     File name to output to
*    debug             i     Debug level
*
* RETURNS:
*    Void
*
* TODO/ISSUES/BUGS:
*    Make proper return codes
*
* LICENCE:
*    cf. XTGeo LICENSE
****************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

/*
 ***************************************************************************************
 *                         ROFF FORMAT
 ***************************************************************************************
 * The ROFF (Roxar Open File Format) is like this:
 *
 * roff-asc
 * #ROFF file#
 * #Creator: RMS - Reservoir Modelling System, version 6.0.2#
 * tag filedata
 * int byteswaptest 1
 * char filetype  "parameter"
 * char creationDate  "08/11/2000 10:45:57"
 * endtag
 * tag version
 * int major 2
 * int minor 0
 * endtag
 * tag dimensions
 * int nX 80
 * int nY 70
 * int nZ 20
 * endtag
 * tag parameter
 * char name  "aBase"
 * array float data 112000
 *     ...numbers...
 * endtag
 * tag eof
 * endtag
 *
 * Discrete:
 * tag parameter
 * char name  "EQLNUM"
 * array char codeNames 2
 * "EQNAME1"
 * "EQNAME2"
 * array int codeValues 2
 *            1            2
 * array int data 112000
 *      ...numbers...
 * -------------------------------------------------------------------------------------
 *
 */

void
grd3d_export_roff_prop(int mode,
                       int nx,
                       int ny,
                       int nz,
                       int num_subgrds,
                       int isubgrd_to_export,
                       int *p_subgrd_v,
                       char *pname,
                       char *ptype,
                       int *p_int_v,
                       double *p_double_v,
                       int ncodes,
                       char *codenames,
                       int *codevalues,
                       char *filename)

{

    int nn, ntotal, nz_true, nz1, nz2, i, j, k, kc;
    long ib, i_tmp;
    int myint;
    float myfloat;
    char mybyte, mychar;
    char mystring[ROFFSTRLEN];
    FILE *fc;
    char *token, **tmp_codenames;
    const char sep[2] = "|";

    if (strcmp(ptype, "double") == 0)
        ptype = "float";

    tmp_codenames = malloc(ncodes * sizeof(char *));
    for (i = 0; i < ncodes; i++) {
        tmp_codenames[i] = (char *)malloc(32);
    }

    /*
     *----------------------------------------------------------------------------------
     * It is possible to export just one subgrid (if isubgrd_to_export >0)
     * Must do some calculations for this here:
     *----------------------------------------------------------------------------------
     */
    nz_true = nz;
    nz1 = 1;
    nz2 = nz;

    if (isubgrd_to_export <= num_subgrds) {
        if (isubgrd_to_export > 0) {
            /* redefine nz */
            nz_true = p_subgrd_v[isubgrd_to_export - 1]; /*vector starts at 0 */

            /* find nz1 and nz2 (counted from top) */
            k = 0;
            for (kc = 0; kc < (isubgrd_to_export - 1); kc++) {
                k = k + p_subgrd_v[kc];
            }
            nz1 = k + 1;
            nz2 = k + p_subgrd_v[isubgrd_to_export - 1];
        }
    } else {
        logger_critical(LI, FI, FU, "Fatal error: isubgrd_to_export too large");
    }

    logger_info(LI, FI, FU, "Opening ROFF file (append)...");
    if (mode == 0) {
        fc = fopen(filename, "ab");
    } else {
        fc = fopen(filename, "ab");
    }

    /*
     *----------------------------------------------------------------------------------
     * Header of file exists by routines export_grd3d_roff_grid (ROFF
     * grid+prop files) or export_grd3d_roff_pstart (ROFF prop only files)
     *----------------------------------------------------------------------------------
     */

    ntotal = nx * ny * nz_true;

    if (mode == 1) {
        fprintf(fc, "tag parameter\n");
        fprintf(fc, "char name \"%s\"\n", pname);
    } else if (mode == 0) {
        fwrite("tag\0parameter\0", 1, 14, fc);
        fwrite("char\0name\0", 1, 10, fc);
        for (i = 0; i <= 100; i++) {
            mychar = pname[i];
            fwrite(&mychar, 1, 1, fc);
            if (pname[i] == '\0')
                break;
        }
    }

    /*
     * A codeName/codeValues table may exist and must be printed
     */
    if (ncodes > 0) {

        /* need to make a list of keywords from the 1D stuff
           input format is "name1|name2|..."
        */
        token = strtok(codenames, sep);
        nn = 0;
        while (token != NULL) {
            strcpy(tmp_codenames[nn], token);
            token = strtok(NULL, sep);
            nn++;
        }

        if (mode > 0 && mode < 2) { /* MODE = 2 ?? */
            fprintf(fc, "array char codeNames %d\n", ncodes);
            for (i = 0; i < ncodes; i++) {
                for (j = 0; j < 33; j++) {
                    mystring[j] = tmp_codenames[i][j];
                    if (mystring[j] == '\0')
                        break;
                }
                fprintf(fc, "\"%s\"\n", mystring);
            }
            i_tmp = 1;
            fprintf(fc, "array int codeValues %d\n", ncodes);
            for (i = 0; i < ncodes; i++) {
                fprintf(fc, " %d", codevalues[i]);
                i_tmp++;
                if (i_tmp > 12) {
                    i_tmp = 1;
                    fprintf(fc, "\n");
                }
            }
            if (i_tmp != 1)
                fprintf(fc, "\n");
        } else {
            fwrite("array\0char\0codeNames\0", 21, 1, fc);
            myint = ncodes;
            fwrite(&myint, 4, 1, fc);
            for (i = 0; i < ncodes; i++) {
                for (j = 0; j < 33; j++) {
                    mystring[j] = tmp_codenames[i][j];
                    if (mystring[j] == '\0')
                        break;
                }
                fwrite(mystring, 1, j + 1, fc);
            }
            fwrite("array\0int\0codeValues\0", 21, 1, fc);
            myint = ncodes;
            fwrite(&myint, 4, 1, fc);
            for (i = 0; i < ncodes; i++) {
                myint = codevalues[i];
                fwrite(&myint, 4, 1, fc);
            }
        }
    }

    /*
     * The array itself...

     */
    if (mode == 1) {
        fprintf(fc, "array %s data %d\n", ptype, ntotal);
    } else if (mode == 0) {
        /* double exported as floats */
        if (strcmp(ptype, "double") == 0 || strcmp(ptype, "float") == 0) {
            fwrite("array\0float\0data\0", 1, 17, fc);
        } else if (strcmp(ptype, "int") == 0) {
            fwrite("array\0int\0data\0", 1, 15, fc);
        } else if (strcmp(ptype, "byte") == 0) {
            fwrite("array\0byte\0data\0", 1, 16, fc);
        }
        myint = ntotal;
        fwrite(&myint, 4, 1, fc);
    }

    i_tmp = 1;
    for (i = 1; i <= nx; i++) {
        for (j = 1; j <= ny; j++) {
            for (k = nz2; k >= nz1; k--) {
                ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                /* double is exported as float */
                if (strcmp(ptype, "float") == 0) {
                    if (mode > 0) {
                        fprintf(fc, "  %e", p_double_v[ib]);
                        i_tmp++;
                        if (i_tmp > 6) {
                            i_tmp = 1;
                            fprintf(fc, "\n");
                        }
                    } else {
                        myfloat = p_double_v[ib];
                        fwrite(&myfloat, 4, 1, fc);
                    }
                } else if (strcmp(ptype, "int") == 0) {
                    if (mode > 0) {
                        fprintf(fc, "  %d", p_int_v[ib]);
                        i_tmp++;
                        if (i_tmp > 6) {
                            i_tmp = 1;
                            fprintf(fc, "\n");
                        }
                    } else {
                        myint = p_int_v[ib];
                        fwrite(&myint, 4, 1, fc);
                    }
                } else if (strcmp(ptype, "byte") == 0) {
                    if (p_int_v[ib] == UNDEF_ROFFINT) {
                        mybyte = 255;
                    } else {
                        mybyte = p_int_v[ib];
                    }
                    if (mode > 0) {
                        fprintf(fc, "  %d", mybyte);
                        i_tmp++;
                        if (i_tmp > 6) {
                            i_tmp = 1;
                            fprintf(fc, "\n");
                        }
                    } else {
                        fwrite(&mybyte, 1, 1, fc);
                    }
                }
            }
        }
    }

    if (mode == 1) {
        if (i_tmp != 1)
            fprintf(fc, "\n");
        fprintf(fc, "endtag\n");
    } else if (mode == 0) {
        fwrite("endtag\0", 1, 7, fc);
    }

    fclose(fc);

    for (i = 0; i < ncodes; i++) {
        free(tmp_codenames[i]);
    }
    free(tmp_codenames);
}
