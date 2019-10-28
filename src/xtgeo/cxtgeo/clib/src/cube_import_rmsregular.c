/*
****************************************************************************************
 *
 * NAME:
 *    cube_import_rmsregular.c
 *
 * DESCRIPTION:
 *    Imports cube on RMS regular binary format. Need to scan the ASCII header
 *    first to allocate pointers in the caller, and to whta line to start
 *    on
 *
 * ARGUMENTS:
 *    line         i     What line binary data starts on
 *    ndef         o     Pointer. Number of defined cells
 *    ndefsum      o     Pointer. Number of defined cells??
 *    nx, ny, nz   i     Dimensions
 *    val_z       i/o    The pointer to the array with values
 *    vmin,vmax    o     Pointer to min/max values
 *    file         i     Filestring
 *    ierr         o     Flag for error
 *
 * RETURNS:
 *    Result vector and pointers are updated
 *
 * TODO/ISSUES/BUGS/NOTES:
 *    Note:  Cube format rotation is...
 *
 *    Note this format can have undef cells
 *
 *    RMS regular format: (max only info)
 *    Xmin/Xmax/Xinc: 5.3076300e+05 5.3229112e+05 1.2499161e+01
 *    Ymin/Ymax/Yinc: 6.7396470e+06 6.7432085e+06 1.8752056e+01
 *    Zmin/Zmax/Zinc: 1.4000000e+03 1.7000000e+03 2.0000000e+00
 *    Rotation: 41.10
 *    Nx/Ny/Nz: 311 191 151
 *    ..... Binary, 4byte float, X fastest, then Y then Z
 *    Note that this format is cell center oriented.
 *    Note also that the Xmin/Ymin is not true Xmin/Ymin, but ORIGIN!
 *    Not sure what XMAX/YMAX is .. (actually not needed!)
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


void cube_import_rmsregular (
                             int    iline,
                             int    *ndef,
                             int    *ndefsum,
                             int    nx,
                             int    ny,
                             int    nz,
                             float  *val_v,
                             double *vmin,
                             double *vmax,
                             char   *file,
                             int    *ierr
                             )
{

    /* locals */
    int i, nxyz, iok_close, not_def;
    float myvalue;
    char string[132];
    FILE *fc;
    int  swap;

    swap=x_swap_check();

    /* ierr=0 if all OK */
    *ierr=0;


    /* The Perl/py class should do a check if file exist! */
    fc=fopen(file,"rb");

    /* header is ASCII. NB remember the \n !... */
    for (i = 1; i <= iline; i++) {
        if (fgets(string, 132, fc) != NULL) logger_debug(__LINE__, "Scanning...");
   }

   *vmin=VERYLARGEFLOAT;
   *vmax=VERYSMALLFLOAT;

   nxyz=nx * ny * nz;
   not_def=0;
   *ndefsum=0;

   for (i=0; i<nxyz; i++) {

	/* read a single value at the time */

       x_fread (&myvalue,4,1,fc,__FILE__,__LINE__);

       if (swap==1) SWAP_FLOAT(myvalue);

       if (myvalue == UNDEF_CUBE_RMS) {
           not_def++;
           myvalue = UNDEF;
       }
       else{
           *ndefsum = *ndefsum + i;
       }

	/* find minimum and maximum map value */
       if (myvalue < UNDEF_LIMIT && myvalue < *vmin) *vmin=myvalue;
       if (myvalue < UNDEF_LIMIT && myvalue > *vmax) *vmax=myvalue;

       val_v[i]=myvalue;
   }

   *ndef= nxyz - not_def;

   iok_close=fclose(fc);

   if (iok_close != 0) *ierr=iok_close;

}
