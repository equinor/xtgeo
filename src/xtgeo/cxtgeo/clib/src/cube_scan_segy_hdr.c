/*
 ******************************************************************************
 *
 * NAME:
 *    cube_scan_segy_hdr.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Scan a SEGY header file.
 *    Special thanks to Gunnar Halvorsen's (GUNHAL) code readsegy.cpp; I would
 *    never have guessed how to do this...
 *    CF: http://www.seg.org/documents/10161/77915/seg_y_rev1.pdf
 *
 * ARGUMENTS:
 *    file                i     File name
 *    gn_bitsheader       o     Bits header number
 *    gn_formatcode       o     Format code...
 *    gf_segyformat       o     Dimensions
 *    gn_samplespertrace  o     Number of samples per trace
 *    gn_measuresystem    o     Measure system...
 *    option              o     Options. 1=print to stdout
 *    debug               i     Debug level flag
 *
 * RETURNS:
 *    Result pointers are updated
 *
 * TODO/ISSUES/BUGS/NOTES:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


void cube_scan_segy_hdr (
                         char  *file,
                         /* return stuff: */
                         int   *gn_bitsheader,
                         int   *gn_formatcode,
                         float *gf_segyformat,
                         int   *gn_samplespertrace,
                         int   *gn_measuresystem,
                         /* flags: */
                         int   option,
                         char  *outfile,
                         int   debug
                         )
{

    /* locals */
    char   ebcdicheader[3200];
    FILE   *fc, *fout=NULL;
    char   s[24]="cube_scan_segy_hdr";
    int    swap, n, i, j, nb, ic;
    int    n4;
    short  n2;


    /* For conversion from EBCDIC to ASCII. ... */

    unsigned char ebcdic2ascii[256]={
        0x00,0x01,0x02,0x03,0x9c,0x09,0x86,0x7f,0x97,0x8d,0x8e,0x0b,0x0c,
        0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x9d,0x85,0x08,0x87,0x18,0x19,
        0x92,0x8f,0x1c,0x1d,0x1e,0x1f,0x80,0x81,0x82,0x83,0x84,0x0a,0x17,
        0x1b,0x88,0x89,0x8a,0x8b,0x8c,0x05,0x06,0x07,0x90,0x91,0x16,0x93,
        0x94,0x95,0x96,0x04,0x98,0x99,0x9a,0x9b,0x14,0x15,0x9e,0x1a,0x20,
        0xa0,0xa1,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0x5b,0x2e,0x3c,0x28,
        0x2b,0x21,0x26,0xa9,0xaa,0xab,0xac,0xad,0xae,0xaf,0xb0,0xb1,0x5d,
        0x24,0x2a,0x29,0x3b,0x5e,0x2d,0x2f,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,
        0xb8,0xb9,0x7c,0x2c,0x25,0x5f,0x3e,0x3f,0xba,0xbb,0xbc,0xbd,0xbe,
        0xbf,0xc0,0xc1,0xc2,0x60,0x3a,0x23,0x40,0x27,0x3d,0x22,0xc3,0x61,
        0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0xc4,0xc5,0xc6,0xc7,0xc8,
        0xc9,0xca,0x6a,0x6b,0x6c,0x6d,0x6e,0x6f,0x70,0x71,0x72,0xcb,0xcc,
        0xcd,0xce,0xcf,0xd0,0xd1,0x7e,0x73,0x74,0x75,0x76,0x77,0x78,0x79,
        0x7a,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xdb,0xdc,0xdd,
        0xde,0xdf,0xe0,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0x7b,0x41,0x42,
        0x43,0x44,0x45,0x46,0x47,0x48,0x49,0xe8,0xe9,0xea,0xeb,0xec,0xed,
        0x7d,0x4a,0x4b,0x4c,0x4d,0x4e,0x4f,0x50,0x51,0x52,0xee,0xef,0xf0,
        0xf1,0xf2,0xf3,0x5c,0x9f,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,
        0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0x30,0x31,0x32,0x33,0x34,0x35,0x36,
        0x37,0x38,0x39,0xfa,0xfb,0xfc,0xfd,0xfe,0xff};

    char *asciiheader[40];
    unsigned char cx;
    unsigned int iindex;

    /* stuff needed to read the binary header */
    short  unassign;
    short  frevnumber;

    int               n4set1[3];
    short             n2set1[24];
    char              unassigned[240];
    short             n2set2[2];

    /* Here are all the descriptions (max 34 char per line) */
    /*                                                      V       */
    /*                     1234567890123456789012345678901234567890 */
    char   des[31][40] = {"Job id",
                          "Line number",
                          "Reel number",
                          "Num. data traces",
                          "Num. aux traces",
                          "Sample interv (us)",
                          "Sample interv. orig (us)",
                          "N samples per trace",
                          "N samples per trace orig",
                          "Data sample format code",
                          "Ensemble fold",
                          "Trace sorting code",
                          "Vertical sum code",
                          "Sweep start freq (Hz)",
                          "Sweep end freq (Hz)",
                          "Sweep length (ms)",
                          "Sweep type code",
                          "Trace number of sweep channel",
                          "Sweep trace length @start (ms)",
                          "Sweep trace length @end (ms)",
                          "Taper type",
                          "Correlated data traces",
                          "Binary gain reovered",
                          "Amplitude recovery method",
                          "Measurement system",
                          "Impulse signal polarity",
                          "Vibratory polarity code",
                          "Unassigned",
                          "Format rev number",
                          "Fixed length trace flag",
                          "No of 3200-byte ext. hdrs."};


    xtgverbose(debug);

    xtg_speak(s,1,"Entering %s",s);

    swap=x_swap_check();

    xtg_speak(s,1,"Swap status: %d",swap);

    /* The caller should do a check if file exist! */
    xtg_speak(s,2,"Opening file %s",file);
    fc=fopen(file,"rb");

    if (fc == NULL) {
        xtg_error(s, "Could not open file");
    }
    /*
     *-------------------------------------------------------------------------
     * Read the EBCDIC header, which usually only contains some -
     * not usable(?) - meta info 3200 bytes
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Work 1");

    n = fread (ebcdicheader, 3200, 1, fc);
    if (n != 1) {
        xtg_error(s,"Error reading SEGY EBCDIC header");
    }

    /*
     *-------------------------------------------------------------------------
     * Convert the EBCDIC header to ASCII, and print it (I would never
     * have guessed this myself...)
     *-------------------------------------------------------------------------
     */

     for (i=0; i<40; i++) {
         asciiheader[i] = calloc(81, sizeof(char));
         if (asciiheader[i] == 0){
             xtg_error(s,"Memory allocation of asciiheader failed");
             return;
         }
     }

     for (i=0; i<40; i++){
         for (j=0; j<80; j++){
             cx = ebcdicheader[i*80 + j];
             iindex = (unsigned int)cx;
             cx = ebcdic2ascii[iindex];
             if (cx<32 || cx>126)
                 asciiheader[i][j] = 32;
             else
                 asciiheader[i][j] = cx;
         }
         asciiheader[i][80] = 0;
     }

    xtg_speak(s,2,"Work 2");

     /* print to screen if option is 1 */
     if (option==1) {
         fout = fopen(outfile, "w");

         fprintf(fout, "\nSTART EBCDIC or ASCII HEADER\n");
         for(i = 0; i < 80; i++) putc('=', fout);
         fprintf(fout, "\n");
         for (i=0; i<40; i++){
             fprintf(fout, "%s\n",asciiheader[i]);
         }
         for(i = 0; i < 80; i++) putc('=', fout);
         fprintf(fout, "\n");
         fprintf(fout, "END EBCDIC or ASCII HEADER\n\n");
     }

    /*
     *-------------------------------------------------------------------------
     * The binary header; this one contains some interesting stuff...
     * The code here is explicit (and somewhat lack of elegance), but
     * gives control
     *-------------------------------------------------------------------------
     */
     if (option==1) {
         fprintf(fout, "BINARY HEADER >>>>>>>>>>\n");
         fprintf(fout, "##         Description                        Byte range"
                "  <bytes>       Value   (Descr.)\n");
         fprintf(fout, "--------------------------------------------------------"
                "---------------------\n");
     }
     nb=1; /* start byte, for info */
     ic=0;
     /* read each chunk of either 4 byte or 2 byte integers */
     for (i=0;i<(sizeof(n4set1)/sizeof(n4set1[0])); i++) {
         n4        = u_read_segy_bitem(ic, i, &n4,sizeof(n4),1,fc, fout, swap,
                                       des[ic],&nb,option,debug);
         n4set1[i] = n4;
         ic++;
     }
     for (i=0;i<(sizeof(n2set1)/sizeof(n2set1[0])); i++) {
         n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout, swap,
                                       des[ic],&nb,option,debug);
         n2set1[i] = n2;
         ic++;
     }

     i=0;
     unassign     = u_read_segy_bitem(ic, i, &unassigned,sizeof(unassigned),
                                      1,fc, fout, swap,"Unassigned",
                                      &nb,option,debug);
     ic++;

     /* frev number is a bit special, see the SEG documentation; handled by
        passing option 2 (noprint) or 3 (print)*/

     i=0;
     option=option+2;
     frevnumber    = u_read_segy_bitem(ic, i, &frevnumber,sizeof(frevnumber),
                                       1,fc, fout, swap,
                                       "Format rev number", &nb,option,debug);
     option=option-2;
     ic++;

     for (i=0;i<(sizeof(n2set2)/sizeof(n2set2[0])); i++) {
         n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout,
                                       swap,des[ic],&nb,option,debug);
         n2set2[i] = n2;
         ic++;
     }


     if (option==1) {
         fprintf(fout, "--------------------------------------------------------"
                "---------------------\n");
     }

    fclose(fc);

    /*
     *--------------------------------------------------------------------------
     * The things to reurn from the scanning of the header... :
     * Thse are kinda global and integer/(float), hence the gn or gf prefix
     *--------------------------------------------------------------------------
     */

    /* I think the headers is always 400 bytes (3600 bits), but
       in case we have it as a return value */
    *gn_bitsheader = 3600;

    /* data sample format code, 1 = IBM 23 bit float, etc */
    *gn_formatcode  = n2set1[6];

    /* SEGY version; return it as float in case a minor version is
       needed later... */
    *gf_segyformat  = (float)frevnumber;

    /* number of samples per trace */
    *gn_samplespertrace = n2set1[4];

    *gn_measuresystem   = n2set1[21];

    if (option == 1) fclose(fout);

}

/* a function to simplify reading the binary items in the primary
   binary header and trace headers */
int u_read_segy_bitem (int ncount, int icount, void *ptr, size_t size,
                       size_t nmemb, FILE *fc, FILE *fout, int swap, char *txt,
                       int *nb, int option, int debug)
{

    int    ier = 0, icode, myreturn = -9, n1, n2;
    short  scode;
    unsigned char f1=0, f2=0;
    char   s[24]="cube_read_segy_item";
    char   comment[30]="";

    n1 = *nb;
    n2 = n1 + size - 1;

    xtgverbose(debug);

    xtg_speak(s,4,"Size of item is %d",size);

    if (option<=1){
        ier=fread(ptr,size,nmemb,fc);
        if (ier != 1) {
            xtg_error(s,"Error in reading SEGY item... IER = %d", ier);
        }

        if (size == 4 && swap) {
            /* this is apparantly the ok way to dereference a void pointer...*/
            icode=*(int*)ptr;
            SWAP_INT(icode);
            myreturn = icode;
        }
        else if (size == 2 && swap) {
            scode=*(short*)ptr;
            SWAP_SHORT(scode);
            myreturn = scode;
        }

        if (size>4) myreturn=0;
    }
    /* special treatment, the record will be read as two single bytes */
    else{
        ier=fread(&f1,1,1,fc);
        if (ier != 1) xtg_error(s,"Error in reading SEGY item...");
        ier=fread(&f2,1,1,fc);
        if (ier != 1) xtg_error(s,"Error in reading SEGY item...");
        sprintf(comment,"SEGY version %02d.%02d",f1,f2);
        /* just return the main version */
        myreturn=f1;
    }


    if (option==1 || option==3) {
        /* Note the SEG defines also the bytes counted from the ASCII
           header at 3200; hence n1+3200 etc */

        fprintf(fout, "%02d (%2d) -> %-34s [%3d - %3d] <%3d Bytes>:  %d  %s\n",
                ncount+1,icount,txt,
                n1,n2,(n2-n1+1),myreturn,comment);
    }

    /*update nb */
    *nb = n2+1;


    return (myreturn);
}
