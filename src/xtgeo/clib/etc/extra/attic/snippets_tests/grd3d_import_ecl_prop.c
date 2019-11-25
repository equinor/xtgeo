/*
 * #################################################################################################
 * grd3d_import_ecl_prop.c
 * Basic routines to handle import of 3D grids/props from Eclipse
 * #################################################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * *************************************************************************************************
 *                        GRD3D_IMPORT_ECLIPSE_PROP
 * *************************************************************************************************
 * This is the format for INIT and restart files. 
 *
 * 'INTEHEAD'         200 'INTE'
 * -1617152669        9701           2       -2345       -2345       -2345
 *       -2345       -2345          20          15           8        1639
 *         246       -2345           7       -2345           0           8
 *          0          15           4           0           6          21
 * ETC!....
 *
 * For UNIFIED restart files, the timestep is indicated by a SEQNUM keyword:
 *
 * 'SEQNUM  '           1 'INTE'
 *           0
 * The actual DATE is item number  65, 66, 67 within INTEHEAD (counted from 1)
 *
 * For the binary form, the record starts and ends with a 4 byte integer, that
 * says how long the current record is, in bytes.
 *
 * -------------------------------------------------------------------------------------------------
 * Parameter list:
 * ftype                Type of file: 
 *                      1 = Binary INIT file
 *                      2 = ASCII  INIT file
 *                      3 = Binary RESTART file (non-unified)
 *                      4 = ASCII  RESTART file (non-unified)
 *                      5 = Binary RESTART file (unified)
 *                      6 = ASCII  RESTART file (unified)
 *
 * nxyz                 Total number of cells, estimated by calling routine
 * p_actnum_v           Active cell array (input)
 * nklist               Number of keywords to look for (input)
 * klist                Keywords to look for (one string PORO|PERMX|...|)
 * ndates               Number of dates entries to look for
 * day                  array of days (1..31)
 * month                array of month (1..12)
 * year                 array of year (e.g. 2010)
 * filename             Name of file to read from
 * dvector_v            Array of double; will be filled. NB stores also float and ints!
 * nktype               Returning the _type_ the _actual_ keys found, 
 *                      1=int, 2=float/real, 3=double, -1=means undef (ie keyword not found)
 *                      nktype[0]=2 means that first keyword asked for is FLOAT (or REAL) 
 * norder               Returning the actual order. For instance, the input list
 *                      0=PORO,1=PERMX,2=PORV,3=PERMY. This can for example return the actual
 *                      "found-in-order" list:
 *                      norder[0]=2,norder[1]=3,norder[2]=0, etc. If keyword no 5 is not found,
 *                      e get norder[5]=-1.
 *                      NOTE! for restart data (keywords with a date):
 *                      Both nktype and norder will be repeated per date
 *
 * ndatefound           Return a list of success of dates, e.g. (1,-1,-1) meaning that first date
 *                      reuested was found, but not the two next
 * debug                Debug (verbose) level
 * -------------------------------------------------------------------------------------------------
 */   

void grd3d_import_ecl_prop (
			     int    ftype,
			     int    nxyz,
			     int    *p_actnum_v,
			     int    nklist,
			     char   *klist,
			     int    ndates,
			     int    *day,
			     int    *month,
			     int    *year,			     
			     char   *filename,
			     double *dvector_v,
			     int    *nktype,
			     int    *norder,
			     int    *dsuccess,
			     int    debug
			     )
{


    int     i, nact, ios=0, iproplook=0;
    int     iprop=0;
    char    cname[9], ctype[5], keyword[9], fmode[3];
    char    s[24]="grd3d_import_ecl_prop";


    /* The calling Perl routine knows the total gridsize (nxyz).
     */

    int     *tmp_int_v, aday, amonth, ayear, nkfound=0, nrfound=0, ntorder=0;
    int     ib, nn, nd_counter=0;
    float   *tmp_float_v;
    double  *tmp_double_v;
    int     *tmp_logi_v;
    int     max_alloc_int, max_alloc_float, max_alloc_double;
    int     max_alloc_char, max_alloc_logi;
    char    **tmp_string_v;
    char    *token, tmp_keywords[nklist][9];
    const char sep[2]="|";
    FILE    *fc;


    xtgverbose(debug);

    /*
     *----------------------------------------------------------------------------------------------
     * Process the keyword list; this is along list that I need to break up into words
     * .e.g <PORV    |PORO    |> --> [PORV    ] [PORO    ]
     *----------------------------------------------------------------------------------------------
     */

    /* get the first token */
    token = strtok(klist, sep); 
    nn=0;
    while (token != NULL) {
       strcpy(tmp_keywords[nn],token);
       xtg_speak(s,1,"Keyword no %d: [%s]",nn,tmp_keywords[nn]);       
       token = strtok(NULL, sep);
       nn++;
    }


    /*
     *----------------------------------------------------------------------------------------------
     * Initialize...
     * I now need to allocate space for tmp_* arrays
     *----------------------------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Allocating memory ...");       
    // max_alloc_int = system('kw_list_largest.pl ) something like this ...
       
    max_alloc_int    = 100*nxyz;
    max_alloc_float  = 100*nxyz;
    max_alloc_double = 100*nxyz;
    max_alloc_char   = 1000;
    max_alloc_logi   = nxyz;

    tmp_int_v    = calloc(max_alloc_int, sizeof(int));
    tmp_float_v  = calloc(max_alloc_float, sizeof(float));
    tmp_double_v = calloc(max_alloc_double, sizeof(double));
    tmp_logi_v   = calloc(max_alloc_logi, sizeof(int));


    /* the string vector is 2D */
    tmp_string_v=calloc(max_alloc_char, sizeof(char *)); 
    for (i=0; i<max_alloc_char; i++) tmp_string_v[i]=calloc(9, sizeof(char));

    xtg_speak(s,2,"Allocating memory ... DONE");       

    /*
     *----------------------------------------------------------------------------------------------
     * Open file and more initial work
     *----------------------------------------------------------------------------------------------
     */
    xtg_speak(s,2,"Opening %s",filename);

    strcpy(fmode,"r");
    if (ftype==1 || ftype==3 || ftype==5) strcpy(fmode,"rb"); 

    fc=fopen(filename,fmode);

    xtg_speak(s,2,"Finish opening %s",filename);
    
    xtg_speak(s,2,"Number of requested keys: %d",nklist);
    
    /* initialise a success array for dates */
    for (i=0;i<ndates;i++) {
	dsuccess[i]=0;
    }

    for (i=0;i<nklist;i++) {
	nktype[i]=-1;
	norder[i]=-1;
	xtg_speak(s,1,"Initial order for property element %d is %d",i,norder[i]);
    }



    /*
     *==============================================================================================
     * START READING!
     *==============================================================================================
     */
    
    xtg_speak(s,1,"Looking for data... scanning file...");


    while (ios == 0) {
	

	if (ftype == 1 || ftype==3 || ftype==5) {
	    ios=u_read_ecl_bin_record (
				       cname,
				       ctype,
				       &nact,
				       max_alloc_int,
				       max_alloc_float,
				       max_alloc_double,
				       max_alloc_char,
				       max_alloc_logi,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}
	else{
	    /* the ASCII reader is not optimal (trouble with CHAR records) */
	    ios=u_read_ecl_asc_record (
				       cname,
				       ctype,
				       &nact,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}	    
	

	if (ios != 0) break;


	xtg_speak(s,2,"Reading <%s>",cname);



	

	xtg_speak(s,3,"NKFOUND is %d and $nklist is %d", nkfound, nklist);



	/* a SEQNUM will always come before the INTEHEAD in restart files */
	if (strncmp(cname, "SEQNUM  ",8) == 0) {
	    xtg_speak(s,2,"New SEQNUM record");
	    if (iproplook==1) {
		iproplook=0;
	    }
	}


	/* 
	 * -----------------------------------------------------------------------------------------
	 * The RESTART LOOP
	 * See if dates are matching, and if so, turn on "look for property flag": iproplook
	 * -----------------------------------------------------------------------------------------
	 */
	if (ndates>0 && iproplook==0) {
	    xtg_speak(s,2,"Looking for dates...");
	    
	    nkfound=0;


	    if (strncmp(cname, "INTEHEAD",8) == 0) {
		aday   = tmp_int_v[64];
		amonth = tmp_int_v[65];
		ayear  = tmp_int_v[66];
		
		xtg_speak(s,2,"INTEHEAD found...");
		xtg_speak(s,1,"Restart date is %d %d %d ...",aday, amonth, ayear);
	    
		/* see if input dates matches */
		for (i=0; i<ndates; i++) {
		    if (day[i] == aday && month[i] == amonth && year[i] == ayear) {
			iproplook=1;
			xtg_speak(s,1,"Date match found for %d %d %d",day[i], month[i], year[i]);
			dsuccess[i]=1;
			nrfound++;
			nd_counter=0;
			break;
		    }
		    else{
			iproplook=0;
		    }
		}
	    }
	}
	else{
	    iproplook=1;
	}


	/* 
	 * -----------------------------------------------------------------------------------------
	 * The PROPERTY LOOP
	 * See if keywords are matching (within a date SEQNUM if restart)
	 * -----------------------------------------------------------------------------------------
	 */

	/* test if cname matches keyword wanted... */

	if (iproplook==1 && nkfound<nklist) {
	    xtg_speak(s,2,"Property to look for is <%s>",cname);
	    
	    
	    for (nn=0; nn<nklist; nn++) {
		strcpy(keyword,tmp_keywords[nn]);
		xtg_speak(s,2,"Looking for <%s>",keyword); 
		if (strncmp(cname, keyword,8) == 0) {
		    
		    xtg_speak(s,2,"--> Found keyword: <%s>",cname);
		    nkfound++;		    

		    /* map the tmp_* to the actual keyword */

 		    if (strcmp(ctype, "REAL") == 0) {
			xtg_speak(s,2,"--> Reading REALs...");
			
			nktype[nn]  = 2;
			norder[nn]  = nd_counter;
			nd_counter++;
			
			i=0;
			for (ib=0;ib<nxyz;ib++) {
			    /* only active cells are stored... normal for PORO etc*/
			    if (nact<nxyz) {
				if (p_actnum_v[ib]==1) {
				    dvector_v[iprop]=tmp_float_v[i];
				    i++;
				}
				else{
				    dvector_v[iprop]=UNDEF;
				}
			    }
			    
			    /* all cells are stored... normal for PORV */
			    else{
				xtg_speak(s,4,"IF and IB  %d  %d", iprop, ib);
				
				dvector_v[iprop]=tmp_float_v[i];
				i++;
			    }
			    iprop++;
			}
		    }
		    else  if (strcmp(ctype, "DOUB") == 0) {
			xtg_speak(s,2,"--> Reading DOUBs...");
			
			nktype[nn]  = 3;
			norder[nn]  = nd_counter;
			nd_counter++;
			
			i=0;
			for (ib=0;ib<nxyz;ib++) {
			    /* only active cells are stored... normal for PORO etc*/
			    if (nact<nxyz) {
				if (p_actnum_v[ib]==1) {
				    dvector_v[iprop]=tmp_double_v[i];
				    i++;
				}
				else{
				    dvector_v[iprop]=UNDEF;
				}
			    }
			    
			    /* all cells are stored... normal for PORV */
			    else{
				xtg_speak(s,4,"IF and IB  %d  %d", iprop, ib);
				
				dvector_v[iprop]=tmp_double_v[i];
				i++;
			    }
			    iprop++;
			}
		    }
 		    else if (strcmp(ctype, "INTE") == 0) {
			xtg_speak(s,2,"--> Reading INTEs...");
			
			nktype[nn]  = 1;
			norder[nn] = nd_counter;
			nd_counter++;
			
			i=0;
			for (ib=0;ib<nxyz;ib++) {
			    /* only active cells are stored... normal for PORO etc*/
			    if (nact<nxyz) {
				if (p_actnum_v[ib]==1) {
				    dvector_v[iprop]=tmp_int_v[i];
				    i++;
				}
				else{
				    dvector_v[iprop]=UNDEF_INT;
				}
			    }
			    
			    /* all cells are stored... normal for PORV */
			    else{
				xtg_speak(s,4,"IF and IB  %d  %d", iprop, ib);
				
				dvector_v[iprop]=tmp_int_v[i];
				i++;
			    }
			    iprop++;
			}
		    }
		}
	    }
	}
	if (nrfound==ndates && nkfound==nklist) {
	    xtg_speak(s,1,"All requested properties are found");
	    break;
	}

    }



    /* check success array for dates */
    for (i=0;i<ndates;i++) {
	if (dsuccess[i]==0) {
	    xtg_speak(s,1,"Date did not match anywhere: day=%d month=%d year=%d",day[i],month[i],year[i]);
	    
	}
    }

    /* give a summary */

    for (i=0;i<ndates;i++) {
	if (dsuccess[i]==1) {
	    xtg_speak(s,2,"Date: day=%d month=%d year=%d:",day[i],month[i],year[i]);

	    for (nn=0; nn<nklist; nn++) {
		strcpy(keyword,tmp_keywords[nn]);
		xtg_speak(s,2,"Keyword is <%s>, type is <%d> and order is <%d>",keyword,nktype[nn],norder[nn]); 
	    }
	}
    }



    xtg_speak(s,2,"Leaving %s:",s);


    /* free allocated space ... not needed?? */
    
    /* if (tmp_int_v != NULL) { */
    /* 	xtg_speak(s,3,"Freeing allocated tmp arrays for INT:"); */
    /* 	free(tmp_int_v); */
    /* 	tmp_int_v = NULL; */
    /* } */
    
    /* if (tmp_logi_v != NULL) { */
    /* 	xtg_speak(s,3,"Freeing allocated tmp arrays for LOGI:"); */
    /* 	free(tmp_logi_v); */
    /* 	tmp_logi_v = NULL; */
    /* } */
    
    /* if (tmp_double_v != NULL) { */
    /* 	xtg_speak(s,3,"Freeing allocated tmp arrays for DOUBLE:"); */
    /* 	free(tmp_double_v); */
    /* 	tmp_double_v = NULL; */
    /* } */

    /* if (tmp_float_v != NULL) { */
    /* 	xtg_speak(s,3,"Freeing allocated tmp arrays for FLOAT:"); */
    /* 	free(tmp_float_v); */
    /* 	tmp_float_v = NULL; */
    /* } */

    /* xtg_speak(s,3,"Freeing allocated tmp arrays for STRING:"); */
    
    /* for (i=0; i<max_alloc_char; i++) free(tmp_string_v[i]);  */
    /* free(tmp_string_v);  */


    xtg_speak(s,4,"Freeing allocated tmp arrays...Finished!"); 
    
}


