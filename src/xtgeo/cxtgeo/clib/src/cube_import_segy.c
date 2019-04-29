/*
 ******************************************************************************
 *
 * NAME:
 *    cube_export_rms_regular.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Imports segY, at least some variants...
 *
 *    Special thanks to Gunnar Halvorsen's (GUNHAL) code readsegy.cpp;
 *    I would never have guessed how to do this...
 *    CF: http://www.seg.org/documents/10161/77915/seg_y_rev1.pdf
 *
 *    Note the cube_scan_segy_hdr is need to be called first
 *
 * ARGUMENTS:
 *    file           i     SEGY file
 *    gn*            i     Some metadata from scan routine
 *    nx..nz        i/o    Cube dimensions ; i/o dependent on flags
 *    p_val_v        o     The 1D cube array
 *    xori..zinc     o     Cube metadata for geometries
 *    rotation       o     Cube roatation: inline compared with UTMX, angle in
 *                         degrees, anti-clock, always in [0, 360>
 *    yflip          o     1 for normal, -1 if flipped in Y (think rotation=0)
 *    zflip          o     Currently not in use
 *    file           i     File to export to
 *    optscan        i     Flag: 1 means scanning mode, 0 means read data
 *    option         i     option = 1: write info to outfile (if optscan=1)
 *    outfile        i     File name to print info too, if option=1 and optscan=1
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *
 * TODO/ISSUES/BUGS:
 *    - update ARGUMENT list above
 *    - yflip handling?
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include <limits.h>
#include "libxtg.h"
#include "libxtg_.h"

void cube_import_segy (
                       char    *file,
                       int     gn_hbitoffset,
                       int     gn_formatcode,
                       float   gf_segyformat,
                       int     gn_samplespertrace,

                       int     *nx,
                       int     *ny,
                       int     *nz,
                       float   *p_val_v,
                       double  *xori,
                       double  *xinc,
                       double  *yori,
                       double  *yinc,
                       double  *zori,
                       double  *zinc,
                       double  *rotation,
                       int     *yflip,
                       int     *zflip,
                       double  *minval,
                       double  *maxval,

                       int     optscan,
                       int     option,
                       char    *outfile,
                       int     debug
                       )
{

    FILE   *fc, *fout=NULL;
    char   s[24]="cube_import_segy";
    int    swap, i, nb, ic=0, offset=0, nzbytes=4,  ier;
    int    n4;
    short  n2;
    int    ntrace[2], ninline[2], nxline[2], ntracecount=0;
    int    ntraces=0, ninlines=0, nxlines=0, ntsamples, mi, mj, k;
    long   ntotal, it, ib;
    int    ii, jj, kk, optscan2;

    double zscalar, xyscalar, xpos[4], ypos[4];

    int       *itracedata = NULL;
    short     *stracedata = NULL;

    char      *ctracebuffer = NULL;
    char      *ctracedata = NULL;
    float     *ftracedata = NULL;

    double    trmin, trmax, ss, rot, rot2, rotrad;

    float     tracepercent=0.0, tracepercentcount=0.0, myfloat=0.0;


    /*
     * stuff needed to read the trace headers; consists of sets of integers
     * (4 byte) or shorts (2 byte)
    */
    int    n4set1[7];
    short  n2set1[4];
    int    n4set2[8];
    short  n2set2[2];  /* to "scalar to be applied at all coordinates */
    int    n4set3[4];
    short  n2set3[46];

    int    n4set4[5];
    short  n2set4[2];

    int    n4set5[1];
    short  n2set5[4];

    char   unassigned[24];
    int    unassign;

    /* Here are all the descriptions (max 34 char per line) */
    /*                                                      V       */
    /*                     1234567890123456789012345678901234567890 */
    char   des[86][40] = {"Trace seq. number within line",
                          "Trace seq. number",
                          "Orig field record number",
                          "Trace number (in orig. field)",
                          "Energy source point number",
                          "Ensemble number",
                          "Trace number within ensemble",
                          "Trace identification code",
                          "Num. vert. summed traces yielding",
                          "Num. hori. stack traces yielding",
                          "Data use",
                          "Distance from center",
                          "Receiver group elevation",
                          "Surface elevation at source",
                          "Source depth below surface",
                          "Datum elevation at receiver group",
                          "Datum elevation at source",
                          "Water depth at source",
                          "Water depth at group",
                          "Scalar to be appl. to all elev.",
                          "Scalar to be appl. to all coord.",
                          "Source coordinate - X",
                          "Source coordinate - Y",
                          "Group coordinate - X",
                          "Group coordinate - Y",
                          "Coordinate units",
                          "Weathering velocity",
                          "Subweathering velocity",
                          "Uphole time at source (ms)",
                          "Uphole time at group (ms)",
                          "Source static correction (ms)",
                          "Group static correction (ms)",
                          "Total static applied (ms)",
                          "Lag time A",
                          "Lag time B",
                          "Delay recording time",
                          "Mute time - start time (ms)",
                          "Mute time - end time (ms)",
                          "Number of samples in this trace",
                          "Sample interval in microsecs (us)",
                          "Gain type of field instruments",
                          "Instrument gain constant (dB)",
                          "Instrument early/initial gain (dB)",
                          "Correlated",
                          "Sweep frequency at start (Hz)",
                          "Sweep frequency at end (Hz)",
                          "Sweep length in millisecs (ms)",
                          "Sweep type",
                          "Sweep trace taper len @start (ms)",
                          "Sweep trace taper len @end (ms)",
                          "Taper type",
                          "Alias filter frequency (Hz)",
                          "Alias filter slope (dB/octave)",
                          "Notch filter frequency (Hz)",
                          "Notch filter slope (dB/octave)",
                          "Low-cut frequency (Hz)",
                          "High-cut frequency (Hz)",
                          "Low-cut slope (dB/octave)",
                          "High-cut slope (dB/octave)",
                          "Year data recorded",
                          "Day of year",
                          "Hour of day",
                          "Minute of hour",
                          "Second of minute",
                          "Time basis code",
                          "Time weighting factor",
                          "Geophone group",
                          "Geophone group",
                          "Geophone group",
                          "Gap size",
                          "Over travel",
                          "X coordinate of ensemble (CDP)",
                          "Y coordinate of ensemble (CDP)",
                          "Inline number",
                          "Crossline number",
                          "Shotpoint number",
                          "Scalar",
                          "Trace value measurement unit",
                          "Transduction Constant Mantissa",
                          "Transduction Constant Pow. of 10",
                          "Transduction units",
                          "Device/Trace Identifier",
                          "Scalar to be applied",
                          "Source xxx unassigned stuff..."};


    /* some initialisation: */
    for (i=0; i<4; i++) {
        xpos[i] = 0.0;
        ypos[i] = 0.0;
        if (i<2) {
            nxline[i] = 0;
            ninline[i] = 0;
        }
    }

    xtgverbose(debug);

    xtg_speak(s,1,"Entering %s",s);

    if (optscan == 1) {
        xtg_speak(s,1,"Scanning mode");
    }

    swap=x_swap_check();

    optscan2 = optscan;

    trmin=1000000000;
    trmax=-1000000000;

    if (gn_formatcode == 1) {
        nzbytes = 4;  /* 4 byte IBM float */
    }
    else if (gn_formatcode == 2) {
        nzbytes = 4;  /* 4 byte signed integer */
    }
    else if (gn_formatcode == 3) {
        nzbytes = 2;  /* 2 byte signed integer */
    }
    else if (gn_formatcode == 4) {
        nzbytes = 4;  /* 4 byte signed integer */
    }
    else if (gn_formatcode == 5) {
        nzbytes = 4;  /* 4 byte IEEE float */
    }
    else if (gn_formatcode == 8) {
        nzbytes = 1;  /* 1 byte signed integer */
    }
    else{
        xtg_error(s,"Format code for traces is not supported: %d",
                  gn_formatcode);
    }

    xtg_speak(s,2,"Swap status: %d",swap);

    /* The caller should do a check if file exist! */
    xtg_speak(s,2,"Opening file %s",file);
    fc=fopen(file,"rb");


    if (option == 1 && optscan == 1) {
        fout = fopen(outfile, "w");
    }

    /*
     *-------------------------------------------------------------------------
     * Skip the ascii and binary reel headers
     *-------------------------------------------------------------------------
     */



    fseek(fc, gn_hbitoffset, SEEK_SET);


    /*
     *=========================================================================
     * Loop and read the trace header and the traces.
     * If the scan mode in active, then take only
     * the first and last IF sufficient info. If not (? will it happen),
     * I need to scan all trace headers. OK some codes her, a bit complicated,
     * but efficient(?):
     * optscan = 1, means scanning only of first and last trace;
     * after scanning or last, it is set to 9.
     * -or-
     * optscan = 0 means that data shall be read; however a scan is needed
     * first to allocate memory.
     * Hence:
     * optscan = 0 scan first trace, then set
     * optscan to 8 and scan last trace, then set
     * optscan 5 which do allocation and stuff, run fseek back to start of
     * traces, and then start reading all traces. Got it...?
     *=========================================================================
     */

    for (it=0; it<2000000; it++) {

        if (ntracecount == ntraces) {
            xtg_speak(s,2,"NTRACES vs COUNT %d vs %d (OPTSCAN = %d)",ntraces,
                      ntracecount, optscan);
        }

        if (it % 1000 == 0) {
            xtg_speak(s, 3, "IT is %ld", it);
        }

        if (it < 10) {
            xtg_speak(s, 2, "TMP: IT(3) is %ld", it);
        }

        /*
         *---------------------------------------------------------------------
         * The trace header; this one contains some interesting
         * (and some uninteresting) stuff...
         * The code here is explicit (and somewhat lack of elegance), but
         * gives control
         *---------------------------------------------------------------------
         */
        if (option == 1 && optscan == 1) {
            fout = fopen(outfile, "w");
            fprintf(fout, "TRACE HEADER FIRST >>>>>>>>>>\n");
            fprintf(fout, "         Description                         Byte range "
                   "local + total       Value\n");
            fprintf(fout, "---------------------------------------------------------"
                   "---------------------------\n");
        }

        if (option==1 && optscan==9) {
            fprintf(fout, "TRACE HEADER LAST >>>>>>>>>>\n");
            fprintf(fout, "        Description                         Byte range "
                   "local + total        Value\n");
            fprintf(fout, "---------------------------------------------------------"
                   "---------------------------\n");
        }
        ic=0;
        nb=1;
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
        for (i=0;i<(sizeof(n4set2)/sizeof(n4set2[0])); i++) {
            n4        = u_read_segy_bitem(ic, i, &n4,sizeof(n4),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n4set2[i] = n4;
            ic++;
        }
        for (i=0;i<(sizeof(n2set2)/sizeof(n2set2[0])); i++) {
            n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n2set2[i] = n2;
            ic++;
        }
        for (i=0;i<(sizeof(n4set3)/sizeof(n4set3[0])); i++) {
            n4        = u_read_segy_bitem(ic, i, &n4,sizeof(n4),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n4set3[i] = n4;
            ic++;
        }
        for (i=0;i<(sizeof(n2set3)/sizeof(n2set3[0])); i++) {
            n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n2set3[i] = n2;
            ic++;
        }
        for (i=0;i<(sizeof(n4set4)/sizeof(n4set4[0])); i++) {
            n4        = u_read_segy_bitem(ic, i, &n4,sizeof(n4),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n4set4[i] = n4;
            ic++;
        }
        for (i=0;i<(sizeof(n2set4)/sizeof(n2set4[0])); i++) {
            n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n2set4[i] = n2;
            ic++;
        }

        for (i=0;i<(sizeof(n4set5)/sizeof(n4set5[0])); i++) {
            n4        = u_read_segy_bitem(ic, i, &n4,sizeof(n4),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n4set5[i] = n4;
            ic++;
        }
        for (i=0;i<(sizeof(n2set5)/sizeof(n2set5[0])); i++) {
            n2        = u_read_segy_bitem(ic, i, &n2,sizeof(n2),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
            n2set5[i] = n2;
            ic++;
        }

        i=0;
        unassign      = u_read_segy_bitem(ic, i, &unassigned,
                                          sizeof(unassigned),1,fc, fout, swap,
                                          des[ic],&nb,option,debug);
        ic++;


        if (option==1 && (optscan==1 || optscan==9)) {
            fprintf(fout, "---------------------------------------------------------"
                   "--------------------------\n");
        }

        /*
         *---------------------------------------------------------------------
         * Collect the important stuff /first, last inline, xline,
         * coordinates etc
         *---------------------------------------------------------------------
         */

        /*scalar to scale Z coordinates, -100 means divide on 100, 100 means
          multiply with 100 etc*/
        zscalar=1;
        zscalar=n2set2[0];
        if (n2set2[0] < 0) {
            zscalar=-1.0/(double)n2set2[0];
        }
        else if (n2set2[0] == 0) {
            xtg_speak(s,3,"The Trace header Z scalar (field 20) is zero. "
                      "Assume no scaling");
            zscalar=1.0;
        }

        /*scalar to scale XY coordinates, -100 means divide on 100, 100
          means multiply with 100 etc*/
        xyscalar=1.0;
        xyscalar=n2set2[1];

        if (n2set2[1] < 0) {
            xyscalar = -1.0/(double)n2set2[1];
        }
        else if (n2set2[1] == 0) {
            xtg_error(s,"The Trace header XY scalar (field 21) cannot be "
                      "zero. STOP!");
        }

        xtg_speak(s,2,"The XY scalar is: %f (from %d) and Z scalar is %f "
                  "(from %d) ",xyscalar,n2set2[1], zscalar, n2set2[0]);
        ntsamples  = n2set3[13];
        *zinc      = n2set3[14]/1000.0; /* from microseconds to milliseconds */
        *zori      = n2set3[10];
        xtg_speak(s,2,"Number of Z samples in this trace is %d ",ntsamples);


        /* get the actual inline and x line */
        mi = n4set4[2];
        mj = n4set4[3];
        xtg_speak(s,3,"Inline %d and xline %d",mi,mj);


        if (optscan <= 1) {
            ntrace[0]  = n4set1[0];
            ninline[0] = n4set4[2];
            nxline[0]  = n4set4[3];
            xpos[0]    = (double)n4set4[0]*xyscalar;
            ypos[0]    = (double)n4set4[1]*xyscalar;
            xtg_speak(s,2,"First point on first inline: %f %f",xpos[0], ypos[0]);

        }


        if (optscan >= 8) {
            ntrace[1]  = n4set1[0];
            ninline[1] = n4set4[2];
            nxline[1]  = n4set4[3];
            /* last corner */
            xpos[3]    = (double)n4set4[0]*xyscalar;
            ypos[3]    = (double)n4set4[1]*xyscalar;
            xtg_speak(s,2,"Last point on last inline: %f %f",xpos[3], ypos[3]);

            ninlines   = ninline[1] - ninline[0] + 1;
            nxlines    = nxline[1]  - nxline[0]  + 1;
            ntraces    = ninlines * nxlines;
            xtg_speak(s,1,"Inlines (rows) and xlines (columns): %d %d",
                      ninlines, nxlines);

        }





        /* print a summary if optscan is 9 (ie a summary for
           both first and last trace)*/
        if (optscan==9 && option==1) {
            fprintf(fout, "\nSummary >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
            fprintf(fout, "First inline: %6d    Last inline: %6d (N inlines = %6d)\n",
                   ninline[0],ninline[1], ninlines);
            fprintf(fout, "First xline : %6d    Last xline : %6d (N xlines  = %6d)\n",
                   nxline[0],nxline[1], nxlines);
            fprintf(fout, "Total number of traces is: %9d\n\n",ntraces);
            fprintf(fout, "First X position: %11.2f    Last X position: %11.2f\n",
                   xpos[0],xpos[3]);
            fprintf(fout, "First Y position: %11.2f    Last Y position: %11.2f\n",
                   ypos[0],ypos[3]);
            fprintf(fout, "Number of samples per trace: %d\n",ntsamples);
            ntotal = ntsamples*nxlines;
            ntotal = ntotal*ninlines;
            xtg_speak(s,2,"Num cells: %ld", ntotal);
            xtg_speak(s,2,"Lim int32: %ld", INT_MAX);
            fprintf(fout, "Number of cells total is: %ld (%d %d %d)\n", ntotal,
                   ninlines,  nxlines, ntsamples);

        }


        /*
         *---------------------------------------------------------------------
         * Work further based on optscan value
         *---------------------------------------------------------------------
         */

        if (option==1 && (optscan==1 || optscan==9)) {
            fprintf(fout, "--------------------------------------------------"
                    "---------------------------------\n");
        }

        /*
         * In case scanning mode, should now jumpt to the last trace. This
         * trace is spooled from the END of the file, to be:
         *    240 bytes (binary trace header
         * +  Nsamples * Nbytes_in_data format
         */

        xtg_speak(s,3,"Working ... OPTSCAN is %d", optscan);

        if (optscan<=1) {;
            /* test 240 + 251*4  =  1244 */
            offset =-1*(240 + ntsamples*nzbytes);
            xtg_speak(s,3,"Offset from end of file is %d ",offset);
            fseek(fc, offset, SEEK_END);
            optscan=optscan+8;
        }
        else if (optscan == 8) {
            /* now both first and last trace are read, so time for memory
               allocation */

            /* jump again to first trace in the SEGY file*/
            fseek(fc, gn_hbitoffset, SEEK_SET);

            /* allocate space for traces */
            ctracebuffer = calloc(4*ntsamples,sizeof(char));
            if (ctracebuffer == 0) {
                xtg_error(s,"Memory allocation failure of traces. STOP");
            }
            ctracedata   = ctracebuffer; /* why + 240?? */
            itracedata   = (int*)ctracedata;
            ftracedata   = (float*)ctracedata;
            stracedata   = (short*)ctracedata;


            /* /\* reallocate the result *\/ */
            /* xtg_speak(s,2,"Reallocate memory ... "); */
            /* p_val_v = realloc(p_val_v, ninlines*nxlines*ntsamples*
               sizeof(p_val_v[0])); */

            /* if (p_val_v == NULL) { */
            /*  xtg_error(s,"Could not reallocate cube pointer. STOP!"); */
            /* } */
            /* xtg_speak(s,2,"Reallocate memory ... DONE"); */

            optscan = 5;

            /* count traces */
            ntracecount=0;
            tracepercent=0.001;
            tracepercentcount=0.0;
        }
        else if (optscan == 5) {

            ntracecount++;
            xtg_speak(s,2,"Formatcode is %d, NTsmaples = %d", gn_formatcode,
                      ntsamples);


            /* read the trace */
            ier = fread(ctracebuffer, nzbytes*ntsamples, 1, fc);
            if (ier != 1) {
                xtg_error(s,"Something is rotten i Denmark");
            }

            ii = mi-ninline[0]  + 1;
            jj = mj-nxline[0]   + 1;

            tracepercent=100*(double)ntracecount/(double)ntraces;
            xtg_speak(s,4,"I J %d %d of total traces %d (tracepercent is %4.1f)",
                      ii,jj,ninlines*nxlines,tracepercent);
            if (tracepercent > tracepercentcount || (ntracecount==ntraces)) {
                xtg_speak(s,1,"Trace %4.1f of 100 percent",tracepercent);
                tracepercentcount+=10;
            }

            /* need to store some corners for geometry computations (angles,
               dx, etc)*/
            if (ii==1 && jj==nxlines) {
                xpos[2]    = (double)n4set4[0]*xyscalar;
                ypos[2]    = (double)n4set4[1]*xyscalar;
                xtg_speak(s,2,"Last point on first inline: %f %f",
                          xpos[2], ypos[2]);
            }

            if (ii==ninlines && jj==1) {
                xpos[1]    = (double)n4set4[0]*xyscalar;
                ypos[1]    = (double)n4set4[1]*xyscalar;
                xtg_speak(s,2,"Last point on first xline: %f %f",
                          xpos[1], ypos[1]);
            }

            /* 32 bit IBM float format */
            if (gn_formatcode == 1){

                xtg_speak(s,2,"Read and convert... (1)");

                /* convert... */;
                u_ibm_to_float(itracedata, itracedata, ntsamples, 1, swap);

                for (k=0; k<ntsamples; k++) {
                    /* the cube coordinates are ii, jj, kk startin in 1 */
                    kk=k+1;

                    ib = x_ijk2ib(ii,jj,kk, ninlines, nxlines, ntsamples,0);

                    if (ib<0) {
                        xtg_error(s,"Something is very wrong; negative IB");
                        exit(9);
                    }


                    p_val_v[ib] = ftracedata[k];


                    if (p_val_v[ib]<trmin) trmin = p_val_v[ib];
                    if (p_val_v[ib]>trmax) trmax = p_val_v[ib];

                    if (ii<3 && jj<3) {
                        xtg_speak(s, 3, "Coordinate %5d %5d %5d value is %f "
                                  "index %ld", ii, jj, kk, p_val_v[ib], ib);
                    }

                }

            }
            else if (gn_formatcode == 5){
                /* IEEE 4 byte float */
                for (k=0; k<ntsamples; k++) {
                    /* the cube coordinates are ii, jj, kk startin in 1 */
                    kk=k+1;

                    ib = x_ijk2ib(ii,jj,kk, ninlines, nxlines, ntsamples,0);
                    myfloat = ftracedata[k];

                    if (swap) SWAP_FLOAT(myfloat);

                    p_val_v[ib] = myfloat;

                    if (p_val_v[ib]<trmin) trmin = p_val_v[ib];
                    if (p_val_v[ib]>trmax) trmax = p_val_v[ib];

                    if (debug>3) {
                        xtg_speak(s,4,"Coordinate %4d %4d %4d value is %f",ii,
                                  jj,kk,p_val_v[ib]);
                    }
                }

            }

            else{
                xtg_error(s,"Reading SEGY, unsupported format code: %d. STOP!",
                          gn_formatcode);
            }




            if (ntracecount==ntraces) {
                xtg_speak(s,2,"This was the last trace, %d of %d",ntracecount,
                          ntraces);
                break;
            }

        }
        else if (optscan==9) {
            break;
        }


    }

    *nx        = ninlines;
    *ny        = nxlines;
    *nz        = ntsamples;

    if (optscan == 1 || optscan==9) {
        fclose(fc);
    }
    else{
        *minval    = trmin;
        *maxval    = trmax;

        *rotation  = -9;
        *xori      = xpos[0];
        *yori      = ypos[0];


        xtg_speak(s,1,"Min and max value is %f   %f ",trmin, trmax);

        *yflip = 0;
        *zflip = 0;

        /* compute geometries */
        if (optscan2 == 0) {

            /* rotation; compute for first inline... */

            x_vector_info2(xpos[0],xpos[1],ypos[0], ypos[1], &ss, &rotrad,
                           &rot, 1, debug);

            *rotation = rot;

            /* deltas for X dir */
            *xinc = ss/(ninlines-1);


            /* Y dir */
            x_vector_info2(xpos[0],xpos[2],ypos[0], ypos[2], &ss, &rotrad,
                           &rot2, 1, debug);

            /* deltas for Y dir */
            *yinc = ss/(nxlines-1);

            /* eval compute xline angle should be 90 if no YFLIP,
               -90 if YFLIP*/
            if (rot <= 270 && (rot2-rot) > 80 && (rot2-rot) < 100) {
                *yflip=1;
            }
            else if (rot > 270 && ((rot2-rot) < 80 || (rot2-rot) > 100)) {
                *yflip=1;
            }
            else{
                *yflip=-1;
            }

            *zflip = 1;


            xtg_speak(s,1,"XANGLE, YANGLE    %6.2f  %6.2f DIFF: %6.2f",
                      rot, rot2, rot2-rot);
            xtg_speak(s,1,"XORI, XINC, ROTATION  %11.2f  %11.2f   %6.2f ",
                      *xori, *xinc, *rotation);
            xtg_speak(s,1,"YORI, YINC, YFLIP     %11.2f  %11.2f   %6d",
                      *yori, *yinc, *yflip);
            xtg_speak(s,1,"ZORI, ZINC            %11.2f  %11.2f",
                      *zori, *zinc);

        }


        fclose(fc);
        xtg_speak(s,1,"SEGY import done, OK");

    }

    if (option == 1 && optscan == 1) {
        fclose(fout);
    }
}


/*
 *... (from seismicunix mailing list. Modified to handle swap)
 ******************************************************************************
 ibm_to_float - convert between 32 bit IBM and IEEE floating numbers
 ******************************************************************************
 Input::
 from           input vector
 to             output vector, can be same as input vector
 endian         byte order =0 little endian (DEC, PC's) =1 other systems
 ******************************************************************************
 Notes:
 Up to 3 bits lost on IEEE -> IBM

 Assumes sizeof(int) == 4

 IBM -> IEEE may overflow or underflow, taken care of by
 substituting large number or zero

 Only integer shifting and masking are used.
 ******************************************************************************
 Credits: CWP: Brian Sumner,  c.1985
 ******************************************************************************
 */

void u_ibm_to_float(int *from, int *to, int n, int endian, int swap)
{
    int         fconv, fmant, i, j, t;
    char        *cptr1 = NULL;
    char        *cptr2 = NULL;

    if (swap) cptr1 = (char*)&fconv;


    for (i = 0;i < n; ++i) {

        if (swap==1) {
            cptr2 = (char*)&from[i];
            for (j=0; j<4; j++)
                cptr1[j] = cptr2[3 - j];
        }
        else{
            fconv = from[i];
        }
        /* if little endian, i.e. endian=0 do this */
        if (endian == 0) fconv = (fconv << 24) | ((fconv >> 24) & 0xff) |
                           ((fconv & 0xff00) << 8) | ((fconv & 0xff0000) >> 8);

        if (fconv) {

            fmant = 0x00ffffff & fconv;

            if (fmant == 0) {
                fconv = 0;
            }
            else {
                t = (int) ((0x7f000000 & fconv) >> 22) - 130;
                while (!(fmant & 0x00800000)) { --t; fmant <<= 1; }
                if (t > 254) fconv = (0x80000000 & fconv) | 0x7f7fffff;
                else if (t <= 0) fconv = 0;
                else fconv =   (0x80000000 & fconv) | (t << 23)
                         | (0x007fffff & fmant);
            }
        }
        to[i] = fconv;
    }
    return;
}
