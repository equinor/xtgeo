/*
 * ############################################################################
 * Some wrappers on common C function to avoid warning messages while compile:
 *     - fgets
 * 
 * Author: JRIV
 * ############################################################################
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void x_fgets (char *str, int n, FILE *stream)
{
    char s[24]="x_fgets";

    if (fgets(str,n,stream)==NULL) {
	xtg_error(s,"Error in read (fgets)... contact JRIV");
    }
}

void x_fread (void *ptr, size_t size, size_t nmemb, FILE *stream, char *caller, int cline)
{
    char s[24]="x_fread";
    size_t   ier;
    ier=fread(ptr,size,nmemb,stream);
    if (ier != nmemb) {
	xtg_shout(s,"Problem in read (fread)... IER=%d nmemb=%d (%s line %d)",ier,nmemb, caller, cline);
    }
}
