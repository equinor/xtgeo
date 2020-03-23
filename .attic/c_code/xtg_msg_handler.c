
/*
 * If value is -1, the current islent level is returned
 */


#include <stdio.h>
#include <string.h>

int xtg_silent (int value){    

    static int isilent;

    if (value>-1) {
	isilent=value;
    }
    return isilent;
}

/*
 * Get and or set the verbose file. If the rotune is called by "NONE", the
 * caller should recognize that no output file is active.
 * If the routine is called by XXXX, the actual filename will be returned
 */


char *xtg_verbose_file(char *filename) {

    static char usefile[160];

    if (strncmp(filename,"XXXX",4)!=0) {
	    strcpy(usefile, filename);
    }
    /* printf("Usefile is %s\n",&usefile); */
    return usefile;
}

