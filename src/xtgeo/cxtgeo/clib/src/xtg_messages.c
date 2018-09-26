/*
 * ############################################################################
 * xtg_messages.c
 * High level message handling for XTGeo routines
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              xtg_messages
 * ****************************************************************************
 * Messages produced from the C sub-system in XTGeo. I found the way of doing
 * the variable function list from Kernigan & Ritchie (1981, p.181)
 * Well, hope this will do (I am talking about the va_list issue)
 * ----------------------------------------------------------------------------
 *
 */
int xtg_speak (char* subname, int dbg_level, char* fmt, ...)
{

    va_list ap;
    int     jverbose, fileflag=0;
    FILE    *flog=NULL;

    /* printf("FILE IS NOW <%s> \n\n",XTG_verbose_file); */

    /* getting current verbose level */
    jverbose=xtgverbose(-1);

    /* this will avoid unbuffered output */
    setvbuf(stdout,NULL,_IONBF,1);


    /* return if jverbose is less than dbg_level; but not ERROR msg */
    if (dbg_level > jverbose) return 0;

    if (strncmp(xtg_verbose_file("XXXX"),"NONE",4)!=0) fileflag=1;


    //fileflag=0;
    if (fileflag==1) {
	//printf("Opening file...<%s>\n",xtg_verbose_file("X"));
	flog=fopen(xtg_verbose_file("XXXX"),"ab");
    }



    if (xtg_silent(-1) < 1) {
	if (jverbose>1) {
	    printf("** <%d> [XTGeo::CLib            ->%33s] ",dbg_level,subname);
	}
	else{
	    printf("**_");
	}	
	/* initialize rest if variable list */
	va_start(ap,fmt);
	
	/* use vprintf to get all the utilities from printf */
	vprintf(fmt,ap);
	va_end(ap);
	/* finally print NEWLINE */
	printf("\n");
    }

    // printf("TEST, fileflag is %d\n", fileflag);

    if (fileflag == 1) {

	if (jverbose>1) {
	    fprintf(flog,"** <%d> [XTGeo::CLib            ->%33s] ",dbg_level,subname);
	}
	else{
	    fprintf(flog,"**_");
	}	
	/* initialize rest if variable list */

	va_start(ap,fmt);
	
	/* use vprintf to get all the utilities from printf */
	vfprintf(flog,fmt,ap);
	va_end(ap);
	/* finally print NEWLINE */

	fprintf(flog,"\n");
	fclose(flog);
    }

    // printf("TEST2\n");
    
    return 0;
}


int xtg_warn (char* subname, int dbg_level, char* fmt, ...)
{

    va_list ap;
    int     jverbose, fileflag=0;
    FILE    *flog=NULL;

    /* printf("FILE IS NOW <%s> \n\n",xtg_verbose_file("X")); */


    /* getting current verbose level */
    jverbose=xtgverbose(-1);

    /* this will avoid unbuffered output */
    setvbuf(stdout,NULL,_IONBF,1);


    /* return if jverbose is less than dbg_level; but not ERROR msg */
    if (dbg_level > jverbose) return 0;

    if (strncmp(xtg_verbose_file("XXXX"),"NONE",4)!=0) fileflag=1;

    /* fileflag=1; */
    if (fileflag==1) {
	/*	printf("Opening file...<%s>\n",xtg_verbose_file("X")); */
	flog=fopen(xtg_verbose_file("XXXX"),"ab");
    }

    if (jverbose>1) {
	printf("## <%d> [XTGeo::CLib            ->%33s] ",dbg_level,subname);
    }
    else{
	printf("##_");
    }	
    /* initialize rest if variable list */
    va_start(ap,fmt);
    
    /* use vprintf to get all the utilities from printf */
    vprintf(fmt,ap);
    va_end(ap);
    /* finally print NEWLINE */
    printf("\n");
   
    if (fileflag == 1) {
	if (jverbose>1) {
	    fprintf(flog,"## <%d> [XTGeo::CLib            ->%33s] ",dbg_level,subname);
	}
	else{
	    fprintf(flog,"##_");
	}	
	/* initialize rest if variable list */
	/* printf("HEI\n"); */

	va_start(ap,fmt);
	
	/* use vprintf to get all the utilities from printf */
	vfprintf(flog,fmt,ap);
	va_end(ap);
	/* finally print NEWLINE */

	fprintf(flog,"\n");
	fclose(flog);
    }
    
    return 0;
}

int xtg_error (char* subname, char* fmt, ...)
{

    va_list ap;
    int     jverbose, fileflag=0;
    FILE    *flog=NULL;

    /* printf("FILE IS NOW <%s> \n\n",xtg_verbose_file("X")); */


    /* getting current verbose level */
    jverbose=xtgverbose(-1);

    /* this will avoid unbuffered output */
    setvbuf(stdout,NULL,_IONBF,1);

    if (strncmp(xtg_verbose_file("XXXX"),"NONE",4)!=0) fileflag=1;

    /* fileflag=1; */
    if (fileflag==1) {
	/*	printf("Opening file...<%s>\n",xtg_verbose_file("X")); */
	flog=fopen(xtg_verbose_file("XXXX"),"ab");
    }

    if (jverbose>1) {
	printf("!! <*> [XTGeo::CLib            ->%33s] ",subname);
    }
    else{
	printf("!!_");
    }	
    /* initialize rest if variable list */
    va_start(ap,fmt);
    
    /* use vprintf to get all the utilities from printf */
    vprintf(fmt,ap);
    va_end(ap);
    /* finally print NEWLINE */
    printf("\n");
   
    if (fileflag == 1) {
	if (jverbose>1) {
	    fprintf(flog,"!! <*> [XTGeo::CLib            ->%33s] ",subname);
	}
	else{
	    fprintf(flog,"!!_");
	}	
	/* initialize rest if variable list */
	/* printf("HEI\n"); */

	va_start(ap,fmt);
	
	/* use vprintf to get all the utilities from printf */
	vfprintf(flog,fmt,ap);
	va_end(ap);
	/* finally print NEWLINE */

	fprintf(flog,"\n");
	fclose(flog);
    }
    
    exit(-1);
    return 0;
}


/* this one displays messages to screen (e.g. listing) irrespective of debug level */
int xtg_shout (char* subname, char* fmt, ...)
{

    va_list ap;
    int     jverbose, fileflag=0;
    FILE    *flog=NULL;


    /* getting current verbose level */
    jverbose=xtgverbose(-1);

    /* this will avoid unbuffered output */
    setvbuf(stdout,NULL,_IONBF,1);

    if (strncmp(xtg_verbose_file("XXXX"),"NONE",4)!=0) fileflag=1;

    /* fileflag=1; */
    if (fileflag==1) {
	/*	printf("Opening file...<%s>\n",xtg_verbose_file("X")); */
	flog=fopen(xtg_verbose_file("XXXX"),"ab");
    }

    if (jverbose>1) {
	printf("++ <*> [XTGeo::CLib            ->%33s] ",subname);
    }
    else{
	printf("   ");
    }	
    /* initialize rest if variable list */
    va_start(ap,fmt);
    
    /* use vprintf to get all the utilities from printf */
    vprintf(fmt,ap);
    va_end(ap);
    /* finally print NEWLINE */
    printf("\n");
   
    if (fileflag == 1) {
	if (jverbose>1) {
	    fprintf(flog,"++ <*> [XTGeo::CLib            ->%33s] ",subname);
	}
	else{
	    fprintf(flog,"   ");
	}	

	va_start(ap,fmt);
	
	/* use vprintf to get all the utilities from printf */
	vfprintf(flog,fmt,ap);
	va_end(ap);
	/* finally print NEWLINE */

	fprintf(flog,"\n");
	fclose(flog);
    }
    
    return 0;
}



