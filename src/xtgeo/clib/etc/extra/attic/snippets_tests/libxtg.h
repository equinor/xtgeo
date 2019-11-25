/*
 *Prototypes for the libgtc libaray
 */

#include <stdio.h>
#include <string.h>
#include <math.h>


#define PHI 3.1415927
#define PHIHALF 1.5707963
#define PI  3.1415927
#define PIHALF  1.5707963
#define FLOATEPS 1.0E-05
#define VERYLARGEFLOAT 10E30;
#define VERYSMALLFLOAT -10E30;

/* 
 * Undefined values etc 
 */

/* 
 * ----------------------------------------------------------------------------
 * Maps. Undef values set to very high values. Map values > UNDEF_MAP_LIMIT
 * are undefined
 * ----------------------------------------------------------------------------
 */

/* general limits in XTGeo are recoemmended for all XTGeo data types! */
#define UNDEF               10E32
#define UNDEF_LIMIT         9.9E32
#define UNDEF_INT           2000000000
#define UNDEF_INT_LIMIT     1999999999


/* specific list */
#define UNDEF_MAP           10E32
#define UNDEF_INT_MAP       2000000000
#define UNDEF_MAP_LIMIT     9.9E32
#define UNDEF_INT_MAP_LIMIT 1999999999
#define UNDEF_MAP_STORM     -999.00
#define UNDEF_MAP_IRAP      9999900.0000
#define UNDEF_CUBE_RMS      -9999.00
#define UNDEF_POINT         10E32
#define UNDEF_POINT_LIMIT   9.9E32
#define UNDEF_POINT_RMS     -999.0000
#define UNDEF_POINT_IRAP    999.0000
#define LAST_POINT          -10E30
#define LAST_POINT_LIMIT    -9.9E30



#define MAXPSTACK 5000
#define ROFFSTRLEN 33
#define ECLNAMELEN 9
#define ECLTYPELEN 5
#define ECLINTEHEADLEN 240
#define ECLDOUBHEADLEN 160
#define ECLLOGIHEADLEN 80

#define  ECLNUMBLOCKLEN 1000
#define  ECLCHABLOCKLEN 105

#define UNDEF_ROFFBYTE 255
#define UNDEF_ROFFINT -999


/* 
 * ============================================================================
 * Importing maps
 * ============================================================================
 */


/*
 *-----------------------------------------------------------------------------
 * Importing Irap ASCII, Storm map, etc
 * ndef          = (O) Number of defined nodes 
 * ndefsum       = (O) sum of IB's 
 * nx, ny        = (O) nodes in X (EW) and Y (NS) dir
 * xstep, ystep  = (O) Mesh width 
 * x/ymin x/ymax = (O) Start and end of mesh (xmax=(nx-1)*xstep+xmin) 
 * zmin, zmax    = (O) Min and Max of max value
 * array         = (O) Actual map values (#: nx*ny)
 * file          = (I) Name of file
 * ierr          = (O) Error flag
 * debug         = (I) Flag applied in Verbose level
 *-----------------------------------------------------------------------------
 */

void map_import_irap_ascii (
			    int *ndef,
			    int *ndefsum,
			    int *nx, 
			    int *ny, 
			    float *xstep, 
			    float *ystep,
			    float *xmin, 
			    float *xmax, 
			    float *ymin, 
			    float *ymax, 
			    float *zmin,
			    float *zmax,
			    float *p_map_v,
			    char  *file, 
			    int *ierr,
			    int debug
			    );

void map_import_surfer_ascii (
			      int *ndef,
			      int *ndefsum,
			      int *nx, 
			      int *ny, 
			      float *xstep, 
			      float *ystep,
			      float *xmin, 
			      float *xmax, 
			      float *ymin, 
			      float *ymax, 
			      float *zmin,
			      float *zmax,
			      float *p_map_v,
			      char  *file, 
			      int *ierr,
			      int debug
			      );

void map_import_storm_binary (
			      int   *ndef,
			      int   *ndefsum,
			      int   *nx, 
			      int   *ny, 
			      float *xstep, 
			      float *ystep,
			      float *xmin, 
			      float *xmax, 
			      float *ymin, 
			      float *ymax, 
			      float *zmin, 
			      float *zmax, 
			      float *p_map_v,
			      char  *file, 
			      int   *ierr,
			      int   debug
			      );

void map_import_arcflt (
		       int   *ndef,
		       int   *ndefsum,
		       int   *nx, 
		       int   *ny, 
		       float *xstep, 
		       float *ystep,
		       float *xmin, 
		       float *xmax, 
		       float *ymin, 
		       float *ymax, 
		       float *zmin, 
		       float *zmax, 
		       float *p_map_v,
		       char  *file1, 
		       char  *file2, 
		       int   *ierr,
		       int   debug
		       );


void map_export_storm_binary (
			      int nx, 
			      int ny, 
			      float xstep, 
			      float ystep,
			      float xmin, 
			      float xmax, 
			      float ymin, 
			      float ymax, 
			      float *p_map_v,
			      char  *file, 
			      int   debug
			      ); 

void map_export_arcflt (
			int nx, 
			int ny, 
			float xstep, 
			float ystep,
			float xmin, 
			float xmax, 
			float ymin, 
			float ymax, 
			float *p_map_v,
			char  *file1, 
			char  *file2, 
			int   debug
			); 

void map_export_surfer_ascii (
			      int nx, 
			      int ny, 
			      float xstep, 
			      float ystep,
			      float xmin, 
			      float xmax, 
			      float ymin, 
			      float ymax, 
			      float *p_map_v,
			      char  *file, 
			      int   debug
			      ); 

/*
 *=============================================================================
 * Map functions
 *=============================================================================
 */


void map_set_value (
		    int nx,
		    int ny,
		    float *p_zval_v,
		    float value,
		    int debug
		    );


void map_operation_value (
			  int mode,
			  int nx,
			  int ny,
			  float *p_zval_v,
			  float value,
			  float value2,
			  float value3,
			  int debug
			  );


int map_median_filter (
		       float *map_in_v,
		       int nx,
		       int ny,
		       int nradius,
		       int mode
		       );



void map_pol_mask(
		  int nx,
		  int ny,
		  float xstep, 
		  float ystep,
		  float xmin, 
		  float ymin, 
		  float *p_map_v,
		  double *p_xp_v,
		  double *p_yp_v,
		  int   np,
		  int   debug
		  );
    

void map_chk_point_between(
			   float x,
			   float y,
			   float z,
			   
			   int nx1,
			   int ny1,
			   float xstep1, 
			   float ystep1,
			   float xmin1, 
			   float ymin1, 
			   float *p_map1_v,
			   
			   int nx2,
			   int ny2,
			   float xstep2, 
			   float ystep2,
			   float xmin2, 
			   float ymin2, 
			   float *p_map2_v,
			   
			   int   *outside,
			   float *zdiff,
			   
			   int debug
			   );

int map_get_corners_xy(
		       float x,
		       float y,
		       
		       int nx,
		       int ny,
		       float xstep, 
		       float ystep,
		       float xmin, 
		       float ymin, 
		       float *p_map_v,
		       
		       int debug
		       );


float map_get_z_from_xy(
			float x,
			float y,

			int nx,
			int ny,
			float xstep, 
			float ystep,
			float xmin, 
			float ymin, 
			float *p_map_v,
			
			int debug
			);


void map_simple_stats (
		       float *p_map_in_v,
		       int nx,
		       int ny,
		       float *zmin,
		       float *zmax,
		       float *mean,
		       int   *ndef,
		       int   *sign,
		       int   debug
		       );


void map_tilt (
	       int   nx, 
	       int   ny, 
	       float dx,
	       float dy,
	       float xstart,
	       float ystart,
	       float *p_map_v,
	       float angle,
	       float azimuth,
	       int   ierror,
	       int   debug
	       );


/* operations between to maps */
void map_operation_map (
			int   nx, 
			int   ny, 
			float *p_map1_v,
			float *p_map2_v,
			int   iop,
			float factor,
			int   debug
			); 



void map_merge_map (
			  int   nx, 
			  int   ny, 
			  float xmin,
			  float ymin,
			  float xinc,
			  float yinc,
			  float *p_map1_v,
			  float *p_map2_v,
			  double *p_xp_v,
			  double *p_yp_v,
			  int   np,
			  int   pflag,
			  int   debug
			  );



void map_wiener_from_grd3d (
			    float z,
			    int nx,
			    int ny,
			    int nz,
			    float *p_coord_v,
			    float *p_zcorn_v,
			    char  *ptype,
			    int   *p_int_v,
			    float *p_float_v,
			    int mx,
			    int my,
			    float xmin,
			    float xstep,
			    float ymin,
			    float ystep,
			    float *p_map_v,
			    int   debug
			    );

void map_slice_cube (
		     int   ncx,
		     int   ncy,
		     int   ncz,
		     float cxmin,
		     float cxinc,
		     float cymin,
		     float cyinc,
		     float czmin,
		     float czinc,
		     float *p_cubeval_v,
		     float crotation,
		     int   mx,
		     int   my,
		     float xmin,
		     float xstep,
		     float ymin,
		     float ystep,
		     float *p_zslice_v,
		     float *p_map_v, 
		     int   option,
		     int   debug
		     );
     

void map_slice_grd3d (
		      int   nx,
		      int   ny,
		      int   nz,
		      float *p_coord_v,
		      float *p_zcorn_v,
		      int   *p_actnum_v,
		      int   mx,
		      int   my,
		      float xmin,
		      float xstep,
		      float ymin,
		      float ystep,
		      float *p_zval_v, 
		      int   *p_ib_v, 
		      int   option,
		      int   debug
		      );
     
void map_sample_grd3d_prop (
			    int   nx,
			    int   ny,
			    int   nz,
			    int   type,
			    int   *p_iprop_v,
			    float *p_fprop_v,
			    int   mx,
			    int   my,
			    float *p_zval_v, 
			    int   *p_ib_v,   
			    int   debug
			    );


/*
 *=============================================================================
 * General ("x") methods...
 *=============================================================================
 */


float x_interp_map_nodes (
			 float *x_v,
			 float *y_v,
			 float *z_v,
			 float x,
			 float y,
			 int method,
			 int debug
			 );

int x_ijk2ib (
	      int i, 
	      int j, 
	      int k, 
	      int nx, 
	      int ny, 
	      int nz,
	      int ia_start
	      );

void x_ib2ijk (
	       int   ib,
	       int   *i,
	       int   *j,
	       int   *k,
	       int   nx,
	       int   ny,
	       int   nz,
	       int   ia_start
	       );


/*
 *=============================================================================
 * Polygon/points methods
 *=============================================================================
 */

void pox_import_rms (
		     double *p_xp_v,
		     double *p_yp_v,
		     double *p_zp_v,
		     char   *file,
		     int    debug
		     );


int pox_import_irap (
		     double *p_xp_v,
		     double *p_yp_v,
		     double *p_zp_v,
		     int    npoints,
		     char   *file,
		     int    debug
		     );



void pox_export_irap_ascii (
			    double *p_xp_v,
			    double *p_yp_v,
			    double *p_zp_v,
			    char   *file,
			    int    npoints,
			    int    iflag,
			    int    debug
			    );



double pox_zsum (
		 double *p_z1_v, 
		 int    np1, 
		 int    debug
		 );


int pox_copy_pox (
		  int   np, 
		  double *p_x1_v, 
		  double *p_y1_v, 
		  double *p_z1_v, 
		  double *p_x2_v, 
		  double *p_y2_v, 
		  double *p_z2_v, 
		  int   debug
		  );

int pox_operation_scalar (
			  double *p_x1_v, 
			  double *p_y1_v, 
			  double *p_z1_v, 
			  int    np1, 
			  double value, 
			  int    iop,
			  int    debug
			  );

int pox_operation_pox (
		       double *p_x1_v, 
		       double *p_y1_v, 
		       double *p_z1_v, 
		       int   np1, 
		       double *p_x2_v, 
		       double *p_y2_v, 
		       double *p_z2_v, 
		       int   np2, 
		       int   iop,
		       float tol,
		       int   debug
		       );



void pol_import_irap (
		      int    i1,
		      int    i2,
		      double  *p_xp_v,
		      double  *p_yp_v,
		      double  *p_zp_v,
		      char   *file,
		      int    debug
		      );


int pol_chk_point_inside(
			 double x,
			 double y,
			 double *p_xp_v,
			 double *p_yp_v,
			 int    np,
			 int    debug
			 );
 
int polys_chk_point_inside(
			   double x,
			   double y,
			   double *p_xp_v,
			   double *p_yp_v,
			   int    np1,
			   int    np2,
			   int    debug
			   );
 
int pol_close(
	       int     np, 
	       double *p_x_v, 
	       double *p_y_v, 
	       double *p_z_v, 
	       double dist,
	       int    option,
	       int    debug
	      );

int pol_set_entry (
		   int    i,
		   double x,
		   double y,
		   double z,
		   int    npmax,
		   double *p_x_v, 
		   double *p_y_v, 
		   double *p_z_v, 
		   int    option, 
		   int    debug
		   );


int pol_refine (
		int    np, 
		int    npmax, 
		double *p_x_v, 
		double *p_y_v, 
		double *p_z_v, 
		double dist,
		int    option,
		int    debug
		);

int pol_extend (
		int    np, 
		double *p_x_v, 
		double *p_y_v, 
		double *p_z_v, 
		double dist,
		int    mode,
		double xang,
		int    option,  /* 0: look in 3D, 1: look in 2d XY */ 
		int    debug
		);

/*
 *=============================================================================
 * Cube (regular 3D) methods
 *=============================================================================
 */

void cube_import_rmsregular (
                             int   iline,
                             int   *ndef,
                             int   *ndefsum,
                             int   nx,
                             int   ny,
                             int   nz,
                             float *val_v,
                             float *vmin,
                             float *vmax,
                             char  *file,
                             int   *ierr,
                             int   debug
                             );

void cube_export_rmsregular (
			     int   nx, 
			     int   ny,
			     int   nz,
			     float xmin, 
			     float ymin, 
			     float zmin, 
			     float xinc, 
			     float yinc, 
			     float zinc, 
			     float rotation,
			     float *val_v,
			     char  *file, 
			     int   debug
			     );

int cube_ij_from_xy(
		    int   *i,
		    int   *j,
		    float x,
		    float y,
		    float xmin,
		    float xinc,
		    float ymin,
		    float yinc,
		    int   nx,
		    int   ny,
		    float rot_azi_deg,
		    int   flag,
		    int   debug
		    );

int cube_ijk_from_xyz(
		      int   *i,
		      int   *j,
		      int   *k,
		      float x,
		      float y,
		      float z,
		      float xmin,
		      float xinc,
		      float ymin,
		      float yinc,
		      float zmin,
		      float zinc,
		      int   nx,
		      int   ny,
		      int   nz,
		      float rot_deg,
		      int   flag,
		      int   debug
		      );

float cube_value_ijk(
		     int   i,
		     int   j,
		     int   k,
		     int   nx,
		     int   ny,
		     int   nz,
		     float *p_val_v,
		     int   debug
		     );

int cube_vertical_val_list(
			   int   i,
			   int   j,
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_val_v,
			   float *p_vertical_v,
			   int   debug
			   );

/*
 *=============================================================================
 * Grid (3D) methods
 *=============================================================================
 */

void grd3d_import_roff_grid (
			    int     *num_act,
			    int     *num_subgrds,
			    float   *p_coord_v,
			    float   *p_zgrd3d_v,
			    int     *p_actnum_v,
			    int     *p_subgrd_v,
			    int     nnsub,
			    char    *filename,
			    int     debug
			    );


void grd3d_import_roff_prop (
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *prop_name,
			      int     *p_int_v,
			      float   *p_float_v,
			      char    **codenames,
			      int     *codevalues,
			      char    *filename,
			      int     debug
			      );



void grd3d_export_roff_grid (
			    int     mode,
			    int     nx,
			    int     ny,
			    int     nz,
			    int     num_subgrds,
			    int     isubgrd_to_export,
			    float   xoffset,
			    float   yoffset,
			    float   zoffset,
			    float   *p_coord_v,
			    float   *p_zgrd3d_v,
			    int     *p_actnum_v,
			    int     *p_subgrd_v,
			    char    *filename,
			    int     debug
			    );


void grd3d_export_roff_end (
			    int     mode,
			    char    *filename,
			    int     debug
			    );

void grd3d_scan_roff_bingrid (
			      int     *nx,
			      int     *ny,
			      int     *nz,
			      int     *nsubs,
			      char    *filename,
			      int     debug
			      );

int grd3d_scan_roff_binpar (
			    char    *parname,
			    char    *filename,
			    int     *ndcodes,
			    int     debug
			    );


void grd3d_export_roff_pstart (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,	
			      char    *filename,	
			      int     debug	
			      );


void grd3d_export_roff_prop (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,
			      int     num_subgrds,
			      int     isubgrd_to_export,
			      int     *p_subgrd_v,
			      char    *pname,
			      char    *ptype,
			      int     *p_int_v,
			      float   *p_float_v,
			      double  *p_double_v,
			      int     ncodes,
			      char    **codenames,
			      int     *codevalues,			      
			      char    *filename,
			      int     debug
			      );




void grd3d_import_ecl_egrid (
			     int   mode,
			     int   nx,
			     int   ny,
			     int   nz,
			     int   *num_active,
			     float *coord_v,
			     float *grd3d_v,
			     int   *actnum_v,
			     char  *filename,
			     int   debug
			     );

void grd3d_scan_ecl_grid_hd (
			     int   ftype,
			     int   *nx,
			     int   *ny,
			     int   *nz,
			     char  *filename,
			     int   debug
			     );

void grd3d_scan_ecl_egrid_hd (
			      int   ftype,
			      int   *nx,
			      int   *ny,
			      int   *nz,
			      char  *filename,
			      int   debug
			      );

void grd3d_import_ecl_grid (
			    int   ftype,
			    int   nxyz,
			    int   *num_active,
			    float *coord_v,
			    float *grd3d_v,
			    int   *actnum_v,
			    char  *filename,
			    int   debug
			    );


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
			    );

void grd3d_list_ecl_kw (
			int    ftype,
			char   *filename,
			int    debug
			);


void grd3d_strip_adouble (
			  int     nxyz,
			  int     counter,
			  double  *v1,
			  double  *v2,
			  int     debug
			  );

void grd3d_strip_afloat (
			 int     nxyz,
			 int     counter,
			 double  *v1,
			 float   *v2,
			 int     debug
			 );

void grd3d_strip_anint (
			 int     nxyz,
			 int     counter,
			 double  *v1,
			 int     *v2,
			 int     debug
			 );

/*
void grd3d_import_ecl_prop (
			     int    ftype,
			     int    ntstep,
			     int    nxyz,
			     int    *nactive,
			     char   *keywords,
			     char   *filename,
			     int    *ivector_v,
			     float  *fvector_v,
			     double *dvector_v,
			     char   **cvector_v,
			     int    *lvector_v,
			     int    debug
			     );
*/

void grd3d_export_ecl_grid (
			    int     nx,
			    int     ny,
			    int     nz,
			    float   *p_coord_v,
			    float   *p_zcorn_v,
			    int     *p_actnum_v,
			    char    *filename,
			    int     mode,
			    int     debug
			    );


void grd3d_export_ecl_pstart (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,
			      int     *p_actnum_v,
			      char    *filename,
			      int     debug
			      );

void grd3d_export_ecl_prop (
			     int     mode,
			     int     nx,
			     int     ny,
			     int     nz,
			     char    *cname,
			     char    *ctype,
			     int     *int_v,
			     float   *float_v,
			     double  *double_v,
			     char    **string_v,
			     int     *logi_v,
			     int     *p_actnum_v,
			     char    *filename,
			     int     debug
			     );

void grd3d_import_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zcorn_v,
			  int     *p_actnum_v,
			  char    *filename,
			  int     debug
			  );

void grd3d_export_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zcorn_v,
			  int     *p_actnum_v,
			  char    *filename,
			  int     debug
			  );

void grd3d_export_grdeclprop (
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *propname,
			      float   *p_fprop_v,
			      int     *p_iprop_v,
			      char    *filename,
			      int     filemode,
			      int     propmode,
			      int     debug
			      );


void grd3d_import_grdeclpar (
                             int     nx,
			     int     ny,
			     int     nz,
			     char    *prop_name,
			     float   *p_float_v,
			     char    *filename,
			     int     debug
			     );
     


void grd3d_export_flgrs(
                        int nx,
			int ny,
			int nz,
			float *p_coord_v,
			float *p_zc_v,
			int   *p_actnum_v,
			float flimit,
			char  *file,
			int   mdiv,
			int   kthrough,
			int   append,
			int   additonal,
			int   debug
                        );



void grd3d_fault_marks(
		       int nx,
		       int ny,
		       int nz,
		       float *p_coord_v,
		       float *p_zcorn_v,
		       int   *p_actnum_v,
		       int   *p_fmark_v,
		       float flimit,
		       int   debug
		       );
    
int grd3d_pol_ftrace(
		     int    nx,
		     int    ny,
		     int    nz,
		     float  *p_coord_v,
		     float  *p_zcorn_v,
		     int    *p_actnum_v,
		     float  *p_fprop_v,
		     float  *p_fprop2_v,
		     int    klayer,
		     int    nelem,
		     int    *fid,
		     float  *sfr,
		     float  *dir,
		     int    *ext,
		     int    *pri,
		     float  *xw1,
		     float  *xw2,
		     float  *yw1,
		     float  *yw2,
		     int    *np,
		     double  *p_xp_v,
		     double  *p_yp_v,
		     int    maxradius,
		     int    ntracemax,
		     int    option,
		     int    debug
		     );

int grd3d_cell_faultthrows(
			   int   i,
			   int   j,
			   int   z,
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   float throw[],
			   int   option,
			   int   debug
			   );

int grd3d_count_prop_int(
			 int   nxyz,
			 int   *p_nprop_v,
			 int   *p_actnum_v,
			 int   nval,
			 int   debug
			 );


void grd3d_convert_hybrid (
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   int   nzhyb,
			   float *p_zcornhyb_v,
			   int   *p_actnumhyb_v,
			   int   *p_num_act,
			   float toplevel,
			   float botlevel,
			   int   ndiv,
			   int   debug
			   );

void grd3d_convert_hybrid2(
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   int   nzhyb,
			   float *p_zcornhyb_v,
			   int   *p_actnumhyb_v,
			   int   *p_num_act,
			   float toplevel,
			   float botlevel,
			   int   ndiv,
			   float *p_region_v,
			   int   region,
			   int   debug
			   );

void grd3d_merge_grids (
			int   nx,
			int   ny,
			int   nz1,
			float *p_coord1_v,
			float *p_zcorn1_v,
			int   *p_actnum1_v,
			int   nz2,
			float *p_zcorn2_v,
			int   *p_actnum2_v,
			int   nznew,
			float *p_zcornnew_v,
			int   *p_actnumnew_v,
			int   *p_numnew_act,
			int   option,
			int   debug
			);
    

void grd3d_make_z_consistent (
			      int   nx,
			      int   ny,
			      int   nz,
			      float *p_zcorn_v,
			      int   *p_actnum_v,
			      float zsep,
			      int   debug
			      );

void grd3d_flip_depth (
		       int   nx,
		       int   ny,
		       int   nz,
		       float *p_coord_v,
		       float *p_zcorn_v,
		       int   debug
		       );


void grd3d_set_dz_cell (
			int   i,
			int   j,
			int   k,
			int   nx,
			int   ny,
			int   nz,
			float *p_zcorn_v,
			int   *p_actnum_v,
			float zsep,
			int   debug
			);


int grd3d_point_in_cell(
			int ibstart,
			int  kzonly,
			float x,
			float y,
			float z,
			int nx,
			int ny,
			int nz,
			float *p_coor_v,
			float *p_zcorn_v,
			int   *p_actnum_v,
			int   option,
			int   debug
			);


void grd3d_collapse_inact (
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   int   debug
			   );


void grd3d_midpoint (
		     int     i,
		     int     j,
		     int     k,
		     int     nx,
		     int     ny,
		     int     nz,
		     float   *p_coord_v,
		     float   *p_zgrd3d_v,
		     float   *x,
		     float   *y,
		     float   *z,
		     int     debug
		     );

void grd3d_cellpoint (
		      int     i,
		      int     j,
		      int     k,
		      int     nx,
		      int     ny,
		      int     nz,
		      int     ftype,
		      float   *p_coord_v,
		      float   *p_zgrd3d_v,
		      float   *x,
		      float   *y,
		      float   *z,
		      int     debug
		      );



void grd3d_make_active(
		       int     i1,
		       int     i2,
		       int     j1,
		       int     j2,
		       int     k1,
		       int     k2,
		       int     nx,
		       int     ny,
		       int     nz,
		       int     *p_actnum_v,
		       int     debug
		       );


void grd3d_inact_outs_pol(
			  int     np,
			  double   *p_xp_v,
			  double   *p_yp_v,
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zcorn_v,
			  int     *p_actnum_v,
			  int     *p_subgrd_v,
			  int     isub,
			  int     nsub,
			  int     option,
			  int     debug
			  );


void grd3d_set_prop_by_pol(
			   int    np,
			   double  *p_xp_v,
			   double  *p_yp_v,
			   int    nx,
			   int    ny,
			   int    nz,
			   float  *p_coord_v,
			   float  *p_zcorn_v,
			   int    *p_actnum_v,
			   float  *p_prop_v,
			   float  value,
			   float  ronly,
			   int    i1,
			   int    i2,
			   int    j1,
			   int    j2,
			   int    k1,
			   int    k2,
			   int    option,
			   int    debug
			   );

void grd3d_set_prop_in_pol(
			   int    np,
			   double  *p_xp_v,
			   double  *p_yp_v,
			   int    nx,
			   int    ny,
			   int    nz,
			   float  *p_coord_v,
			   float  *p_zcorn_v,
			   int    *p_actnum_v,
			   int    *p_prop_v,
			   float  value,
			   int    i1,
			   int    i2,
			   int    j1,
			   int    j2,
			   int    k1,
			   int    k2,
			   int    option,
			   int    debug
			   );

void grd3d_split_prop_by_pol(
			     int    np,
			     double  *p_xp_v,
			     double  *p_yp_v,
			     int    nx,
			     int    ny,
			     int    nz,
			     float  *p_coord_v,
			     float  *p_zcorn_v,
			     int    *p_actnum_v,
			     int    *p_prop_v,
			     float  *p_propfilter_v,
			     float  filterval,
			     int    i1,
			     int    i2,
			     int    j1,
			     int    j2,
			     int    k1,
			     int    k2,
			     int    option,
			     int    debug
			     );

void grd3d_inact_by_dz(
		       int nx,
		       int ny,
		       int nz,
		       float *p_zcorn_v,
		       int   *p_actnum_v,
		       float threshold,
		       int   flip,
		       int   debug
		       );

void grd3d_copy_prop_int(
			 int   nxyz,
			 int   *p_input_v,
			 int   *p_output_v,
			 int   debug
			 );

void grd3d_copy_prop_float(
			   int   nxyz,
			   float *p_input_v,
			   float *p_output_v,
			   int   debug
			   );


void grd3d_ifthen_prop_intint(
			      int   nx,
			      int   ny,
			      int   nz,
			      int   i1,
			      int   i2,
			      int   j1,
			      int   j2,
			      int   k1,
			      int   k2,
			      int   *p_other_v,
			      int   *p_this_v,
			      int   *p_array,
			      int   alen,
			      int   newvalue,
			      int   elsevalue,
			      int   rmin,
			      int   rmax,
			      int   debug
			      );

float grd3d_frac_prop_within_ii(
			       int   nxyz,
			       int   *p_other_v,
			       int   *p_this_v,
			       int   *p_oarray,
			       int   oalen,
			       int   *p_tarray,
			       int   talen,
			       int   debug
				);
    
void grd3d_remap_prop_g2g(
			   int   nx1,
			   int   ny1,
			   int   nz1,
			   int   nx2,
			   int   ny2,
			   int   nz2,
			   int   isub,
			   int   num_subgrds,
			   int   *p_subgrd_v,
			   char  *ptype,
			   int   *p_int1_v,
			   float *p_float1_v,
			   int   *p_int2_v,
			   float *p_float2_v,
			   int   debug
			   );

int grd3d_count_active(
		       int     nx,
		       int     ny,
		       int     nz,
		       int    *p_actnum_v,
		       int debug
		       );


void grd3d_calc_dz(
		   int nx,
		   int ny,
		   int nz,
		   float *p_zcorn_v,
		   float *p_dz_v,
		   int flip,
		   int debug
		   );

void grd3d_calc_z(
		  int nx,
		  int ny,
		  int nz,
		  float *p_zcorn_v,
		  float *p_z_v,
		  int debug
		  );

int grd3d_prop_infill1_int(
			   int nx,
			   int ny,
			   int nz,
			   int i1,
			   int i2,
			   int j1,
			   int j2,
			   int k1,
			   int k2,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   int   *p_xxx_v,
			   int   value,
			   int   debug
			   );

void grd3d_calc_cell_dip(
			 int nx,
			 int ny,
			 int nz,
			 float *p_coord_v,
			 float *p_zcorn_v,
			 float *p_dip_v,
			 int   debug
			 );

void grd3d_interp_prop_verti(
			     int   nx,
			     int   ny,
			     int   nz,
			     float *p_zcorn_v,
			     int   *p_actnum_v,
			     float *p_xxx_v,
			     float bgval,
			     float dzmax,
			     int   option,
			     int   debug
			     );

void grd3d_interp_prop_vertii(
			      int   nx,
			      int   ny,
			      int   nz,
			      int   i1,			      
			      int   i2,
			      int   j1,
			      int   j2,
			      float *p_zcorn_v,
			      int   *p_actnum_v,
			      int   *p_xxx_v,
			      int   bgval,
			      float dzmax,
			      int   option,
			      int   debug
			      );

int grd3d_getcell_by_prop(
			  int     nx,
			  int     ny,
			  int     nz,
			  int     i1,
			  int     i2,
			  int     j1,
			  int     j2,
			  int     k1,
			  int     k2,
			  float   *p_coord_v,
			  float   *p_zcorn_v,
			  int     *p_actnum_v,
			  float   *p_xxx_v,
			  float   pmin,
			  float   pmax,
			  int     criteria,
			  int     option,
			  int     debug
			  );

    
void grd3d_calc_sum_dz(
		       int nx,
		       int ny,
		       int nz,
		       float *p_zcorn_v,
		       int   *p_actnum_v,
		       float *p_sumdz_v,
		       int   flip,
		       int   debug
		       );


void grd3d_calc_abase(
		      int option,
		      int nx,
		      int ny,
		      int nz,
		      float *p_dz_v,
		      float *p_abase_v,
		      int   flip,
		      int   debug
		      );


void grd3d_print_cellinfo (
			   int   i1,
			   int   i2,
			   int   j1,
			   int   j2,
			   int   k1,
			   int   k2,
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *actnum_v,
			   int   debug
			   );
  
void grd3d_corners (
		    int     i,
		    int     j,
		    int     k,
		    int     nx,
		    int     ny,
		    int     nz,
		    float   *p_coord_v,
		    float   *p_zgrd3d_v,
		    float   corners[],
		    int     debug
		    );

void grd3d_set_propval_int(
			   int     nxyz,
			   int     *p_prop_v,
			   int     value,
			   int     debug
			   );

void grd3d_set_propval_float(
			     int     nxyz,
			     float   *p_prop_v,
			     float   value,
			     int     debug
			     );

    
void grd3d_pzcorns (
		    int     i,
		    int     j,
		    int     k,
		    int     nx,
		    int     ny,
		    int     nz,
		    float   *p_coord_v,
		    float   *p_zgrd3d_v,
		    float   *p,
		    int     *flag,
		    int     debug
		    );


void grd3d_adj_z_from_zlog (
			    int   nx,
			    int   ny,
			    int   nz,
			    float *p_coord_v,
			    float *p_zcorn_v,
			    int   *p_actnum_v,
			    int   *p_zon_v,
			    int   nval,
			    float *p_utme_v,
			    float *p_utmn_v,
			    float *p_tvds_v,
			    int   *p_zlog_v,
			    float dupper,
			    float dlower,
			    int   iflag,
			    int   debug
			    );

void grd3d_adj_z_from_map (
			   int nx,
			   int ny,
			   int nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int *p_actnum_v,
			   int mx,
			   int my,
			   float xmin,
			   float xstep,
			   float ymin,
			   float ystep,
			   float *p_grd2d_v,
			   float zconst,
			   int   iflag,
			   int   flip,
			   int   debug
			   );

void grd3d_adj_z_from_mapv2 (
			     int nx,
			     int ny,
			     int nz,
			     float *p_coord_v,
			     float *p_zcorn_v,
			     int *p_actnum_v,
			     int mx,
			     int my,
			     float xmin,
			     float xstep,
			     float ymin,
			     float ystep,
			     float *p_map_v,
			     int   iflag,
			     int   flip,
			     int   debug
			     );
void grd3d_adj_z_from_mapv3 (
			     int nx,
			     int ny,
			     int nz,
			     float *p_coord_v,
			     float *p_zcorn_v,
			     int *p_actnum_v,
			     int mx,
			     int my,
			     float xmin,
			     float xstep,
			     float ymin,
			     float ystep,
			     float *p_map_v,
			     int   iflag,
			     int   flip,
			     int   debug
			     );

/* diff */
void grd3d_diff_zcorn (
		       int nx,
		       int ny,
		       int nz,
		       float *p_coord1_v,
		       float *p_zcorn1_v,
		       int   *p_actnum1_v,
		       float *p_dz1_v,
		       float *p_dz2_v,
		       float *p_dz3_v,
		       float *p_dz4_v,
		       float *p_dz5_v,
		       float *p_dz6_v,
		       float *p_dz7_v,
		       float *p_dz8_v,
		       float *p_dzavg_v,
		       float *p_coord2_v,
		       float *p_zcorn2_v,
		       int   *p_actnum2_v,
		       int   debug
		       );


void grd3d_adj_dzcorn (
		       int nx,
		       int ny,
		       int nz,
		       float *p_zcorn_v,
		       float *p_dz1_v,
		       float *p_dz2_v,
		       float *p_dz3_v,
		       float *p_dz4_v,
		       float *p_dz5_v,
		       float *p_dz6_v,
		       float *p_dz7_v,
		       float *p_dz8_v,
		       int   debug
		       );


int grd3d_set_active(
		     int   nx,
		     int   ny,
		     int   nz,
		     int   i1,
		     int   i2,
		     int   j1,
		     int   j2,
		     int   k1,
		     int   k2,
		     int   *p_actnum_v,
		     int   debug
		     );

void grd3d_zdominance_int(
			  int   nx,
			  int   ny,
			  int   nz,
			  float *p_zcorn_v,
			  int   *p_actnum_v,
			  int   *p_xxx_v,
			  int   option,
			  int   debug
			  );


/* more general things */

int xtg_silent (int value);
char *xtg_verbose_file(char *filename);
