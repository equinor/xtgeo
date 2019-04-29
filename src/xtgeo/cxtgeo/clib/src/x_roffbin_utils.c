/*
 *******************************************************************************
 *
 * Utilities for importing ROFF binary files
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    x_roffbin_utils.c (file name)
 *    x_roffbinstring
 *    x_roffgetfloatvalue
 *    x_roffgetintvalue
 *    x_roffgetfloatarray
 *    x_roffgetbytearray
 *    x_roffgetintarray
 *    x_roffgetchararray
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Imports maps on Irap binary formats.
 *
 * The ROFF (Roxar Open File Format) is like this:
 *
 *roff-asc
 *#ROFF file#
 *#Creator: RMS - Reservoir Modelling System, version 6.0#
 *tag filedata
 *int byteswaptest 1
 *char filetype  "grid"
 *char creationDate  "28/03/2000 16:59:16"
 *endtag
 *tag version
 *int major 2
 *int minor 0
 *endtag
 *tag dimensions
 *int nX 4
 *int nY 4
 *int nZ 3
 *endtag
 *tag translate
 *float xoffset   4.62994625E+05
 *float yoffset   5.93379900E+06
 *float zoffset  -3.37518921E+01
 *endtag
 *tag scale
 *float xscale   1.00000000E+00
 *float yscale   1.00000000E+00
 *float zscale  -1.00000000E+00
 *endtag
 *tag cornerLines
 *array float data 150
 * -7.51105194E+01  -4.10773730E+03  -1.86212000E+03  -7.51105194E+01
 * -4.10773730E+03  -1.72856909E+03  -8.36509094E+02  -2.74306006E+03
 *  ....
 *endtag
 *tag zvalues
 *array byte splitEnz 100
 *  1   1   1   1   1   1   4   4   1   1   1   1
 * ....
 *endtag
 *tag active
 *array bool data 48
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *endtag
 *... ETC
 *
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */


/*
 *******************************************************************************
 * Reading a string in binary file terminated with NULL
 * Something is strange here(??); see result if x_fread are used...
 *******************************************************************************
 */

int x_roffbinstring(char *bla, FILE *fc)
{
    int i, ier;
    char mybyte;

    for (i=0;i<ROFFSTRLEN;i++) {
        /* x_fread(&mybyte,1,1,fc,__FILE__,__LINE__); */
        ier=fread(&mybyte,1,1,fc);
	bla[i]=mybyte;
        if (mybyte==0) break;
    }
    return 0;
}


/*
 *******************************************************************************
 * Reading a float
 * e.g: <float xoffset   4.62994625E+05>
 *******************************************************************************
 */
float x_roffgetfloatvalue(char *name, FILE *fc)
{
    char bla[ROFFSTRLEN];
    float myfloat;

    x_roffbinstring(bla, fc);
    if (strcmp(bla,"float")==0) {
	x_roffbinstring(bla, fc);
	if (strcmp(bla,name)==0) {
	    x_fread(&myfloat,4,1,fc,__FILE__,__LINE__);
	    if (x_byteorder(-1)>1) SWAP_FLOAT(myfloat);
	    return myfloat;
	}
    }
    return -1.0;
}



/*
 *******************************************************************************
 * Reading an int, e.g. <int nX 4>
 *******************************************************************************
 */
int x_roffgetintvalue(char *name, FILE *fc)
{
    char bla[ROFFSTRLEN];
    int  myint;

    if (strcmp(name,"array")==0) {

	/* return -1 in case not a array as suggested... */
	x_roffbinstring(bla, fc); /* array */
	if (strcmp(bla,"array")!=0) {
	    return -1;
	}
	x_roffbinstring(bla, fc); /* int or float */
	x_roffbinstring(bla, fc); /* data */
	x_fread(&myint,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_INT(myint);
	return myint;
    }
    else{
	x_roffbinstring(bla, fc);
	if (strcmp(bla,"int")==0) {
	    x_roffbinstring(bla, fc);
	    if (strcmp(bla,name)==0) {
		x_fread(&myint,4,1,fc,__FILE__,__LINE__);
		if (x_byteorder(-1)>1) SWAP_INT(myint);
		return myint;
	    }
	}
    }
    return -1;
}


/*
 *******************************************************************************
 * Reading a  float array
 *******************************************************************************
 */
void x_roffgetfloatarray(float *array, int num, FILE *fc)
{
    float afloat;
    int   i;

    for (i=0;i<num;i++) {
	x_fread(&afloat,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_FLOAT(afloat);
	array[i]=afloat;
    }

    /* PREVIOUS WAY: fread(array,4,num,fc); */
}


/*
 *******************************************************************************
 * Reading a byte array, e.g:
 * array bool data 48 (this is not read here but by read int...)
 *   1   1   1   1   1 ...
 *******************************************************************************
 */
void x_roffgetbytearray(unsigned char *array, int num, FILE *fc)
{
    int  i;
    unsigned char abyte;

    for (i=0;i<num;i++) {
	x_fread(&abyte,1,1,fc,__FILE__,__LINE__);
	array[i]=abyte;
    }
}


/*
 *******************************************************************************
 * Reading a int array
 *******************************************************************************
 */
void x_roffgetintarray(int *array, int num, FILE *fc)
{
    int  i;
    int aint;

    for (i=0;i<num;i++) {
	x_fread(&aint,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_SHORT(aint);   /* SHORT? */
	array[i]=aint;
    }
}

/*
 *******************************************************************************
 * Reading a char array
 *
 * Get the chararray as a 1D array, and separate each word with |
 * This is because SWIG works best with 1D char arrays. Note that
 * num is the number of words. It is assumed that the allocation of
 * array is number of total chars, and sufficient large enough
 *******************************************************************************
 */
void x_roffgetchararray(char *array, int num, FILE *fc)
{
    int  i, ic, j, m, n, nc;
    char c[ROFFSTRLEN];
    char sub[24]="x_roffgetchararray";
    char xstr[8];

    ic=0;
    for (i=0;i<num;i++) {
	x_roffbinstring(c,fc);
	xtg_speak(sub,4,"Reading: <%s>",c);
	for (j=0;j<99999;j++) {
	    if (c[j] == '\0' && j==0) {
		/* to replace missing string with a code */
		m = i + 1;
		sprintf(xstr, "%d", m);
		n = strlen(xstr);
		for (nc=0; nc<n; nc++) {
		    array[ic]=xstr[nc];
		    ic++;
		    if (nc == (n-1)) {
			array[ic]='|';
			ic++;
			break;
		    }
		}
		break;
	    }
	    else if (c[j] == '\0') {
		array[ic]='|';
		ic++;
		break;
	    }
	    else{
		array[ic]=c[j];
		ic++;
	    }
	}
    }
    /* terminate char array with null string */
    //array[ic-1]='\0';
}
