/*
 * ############################################################################
 * u_eightletter.c
 * Convert Eclipse Names to 8 letters including blanks
 * ############################################################################
 * $Id: eightletter.c,v 1.1 2000/12/12 17:24:54 bg54276 Exp $ 
 * $Source: /h/bg54276/jcr/prg/lib/gplext/GPLExt/RCS/eightletter.c,v $ 
 *
 * $Log: eightletter.c,v $
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 *
 */


#include <string.h>
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              EIGTHLETTER
 * ****************************************************************************
 * A "private" routine for getting 8 CHAR fields in a string. Thats because
 * fscanf etc. seems to truncate at last blank field. In addition, the
 * routine will also replace "." in strings
 * ----------------------------------------------------------------------------
 *
 */

int u_eightletter (
		 char *cs
		 )
{
    
    int i, nlen;

    nlen=strlen(cs);

    if (nlen == 1) {
	strcat(cs,"       ");
    }
    else if (nlen == 2) {
	strcat(cs,"      ");
    }	
    else if (nlen == 3) {
	strcat(cs,"     ");
    }	
    else if (nlen == 4) {
	strcat(cs,"    ");
    }	
    else if (nlen == 5) {
	strcat(cs,"   ");
    }	
    else if (nlen == 6) {
	strcat(cs,"  ");
    }	
    else if (nlen == 7) {
	strcat(cs," ");
    }	
    else if (nlen == 8) {
	/* may now need to replace "." with " " */
	for (i=0;i<8;i++) {
	    if (cs[i] == '.') cs[i]=' ';
	}
    }	

    return nlen;

}
