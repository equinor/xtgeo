/*
 * ############################################################################
 * xtg_msg.c
 * High level message handling for GPLExt routines
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: xtg_msg.c,v 1.2 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/xtg_msg.c,v $ 
 *
 * $Log: xtg_msg.c,v $
 * Revision 1.2  2001/03/14 08:02:29  bg54276
 * *** empty log message ***
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 * General description:
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              GTC_MSG
 * ****************************************************************************
 * Messages produced from the C sub-system in GPLib. I found the way of doing
 * the variable function list from Kernigan & Ritchie (1981, p.181)
 * ----------------------------------------------------------------------------
 *
 */
int xtg_msg (int mtype, int dbg_level, char* fmt, ...)
{

    va_list ap;
    int     jverbose;

    /* getting current verbose level */
    jverbose=xtgverbose(-1);

    /* return if jverbose is less than dbg_level; but not ERROR msg */
    if (dbg_level > jverbose && mtype < 3) return 0;


    if (mtype == 1) printf("** GTC<%d> ",dbg_level);
    if (mtype == 2) printf("## GTC<%d> ",dbg_level);
    
    /* Fatal errors. Here, the verbose level does not matter!! */
    if (mtype == 3) printf("!! GTC<-> ");

    /* initialize rest if variable list */
    va_start(ap,fmt);

    /* use vprintf to get all the utilities from printf */
    vprintf(fmt,ap);
    va_end(ap);

    /* finally print NEWLINE */
    printf("\n");

    if (mtype == 3) exit(2);

    return 0;
}

    
