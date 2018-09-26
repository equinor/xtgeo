/*
 *Prototypes for the libgtc libaray
 */

#include <stdio.h>
#include <string.h>
#include <math.h>

#define PHI 3.1415927
#define PHIHALF 1.5707963
#define FLOATEPS 1.0E-05
#define VERYLARGEFLOAT 10E30;
#define VERYSMALLFLOAT -10E30;

/* 
 * Undefined values etc 
 */

#define UNDEF_MAP -9999.9900000000000
#define UNDEF_MAP_LIMIT -9999.0000000000000

#define STORM_GRD2D_UNDEF -999.00
#define MAXPSTACK 5000
#define ROFFSTRLEN 33
#define ECLNAMELEN 9
#define ECLTYPELEN 5
#define ECLINTEHEADLEN 240
#define ECLDOUBHEADLEN 160
#define ECLLOGIHEADLEN 80

#define  ECLNUMBLOCKLEN 1000
#define  ECLCHABLOCKLEN 105

#define ROFFBYTE_UNDEF 255
#define ROFFINT_UNDEF -999

#define POINTS_UNDEF -888888.0000


/* 
 * ============================================================================
 * Importing maps
 * ============================================================================
 */


/*
 *-----------------------------------------------------------------------------
 * Importing Irap ASCII map
 * ndef          = (O) Number of defined nodes 
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

void grd2d_import_irap_ascii (
			      int *ndef,
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
			      float *p_grd2d_v,
			      char  *file, 
			      int *ierr,
			      int debug
			      );


/*
 *-----------------------------------------------------------------------------
 * Import Storm binary map (regular, not rotated)
 * Arguments: see grd2d_import_irap_ascii
 *-----------------------------------------------------------------------------
 */

void grd2d_import_storm_binary (
				int   *ndef,
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
				float *p_grd2d_v,
				char  *file, 
				int   *ierr,
				int   debug
				);


/* 
 * ============================================================================
 * Exporting maps
 * ============================================================================
 */

/*
 *-----------------------------------------------------------------------------
 * Importing Irap ASCII map
 * nx, ny          = (I) Number of defined nodes in X and Y dir
 * x/y step/min/.. = (I) Geometrical settings of the map
 * p_grd2d_v       = (I) The map values
 * file            = (I) Name of file to export to
 * debug           = (I) Flag applied in Verbose level
 *-----------------------------------------------------------------------------
 */
void grd2d_export_storm_binary (
				int nx, 
				int ny, 
				float xstep, 
				float ystep,
				float xmin, 
				float xmax, 
				float ymin, 
				float ymax, 
				float *p_grd2d_v,
				char  *file, 
				int   debug
				); 


/* 
 * ============================================================================
 * Importing 3D grids and parameters
 * ============================================================================
 */

/*
 *-----------------------------------------------------------------------------
 * Importing Eclipse grid
 * ftype           = ?
 * nx, ny, nz      = (O) Number of cells X Y Z
 *
 * THIS ROUTINE IS NOT WORKING PROPERLY NOW
 *-----------------------------------------------------------------------------
 */

void grd3d_import_eclipse_grid (
				int   ftype,
				int   *nx,
				int   *ny,
				int   *nz,
				int   nxyz,
				int   *num_active,
				float *grd3d_v,
				int   *actnum_v,
				char  *filename,
				int   debug
				);

/*
 *-----------------------------------------------------------------------------
 * Importing ROFF grid (both binary and ASCII, but no BYTE-ORDER reversing
 * is implemented)
 * num_act         = (O) Number of active cells (not in use now)
 * num_subs        = (O) Number of sub_grids
 * p_coord_v       = (O) Holds the top and bottom pillar XYZ coords
 * p_zcorn_v       = (O) Holds the top 4 Z corners of each cell
 * p_actnum_v      = (O) Holds ACTNUM array
 * p_subgrd_v      = (O) Small array that holds the number of layers pr sub.
 * filename        = (I) Name of file
 * debug           = (I) A.a.
 *-----------------------------------------------------------------------------
 */

void grd3d_import_roff_grid (
			    int     *num_act,
			    int     *num_subs,
			    float   *p_coord_v,
			    float   *p_zgrd3d_v,
			    int     *p_actnum_v,
			    int     *p_subgrd_v,
			    char    *filename,
			    int     debug
			    );


void grd3d_import_roff_param (
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *param_name,
			      int     *p_int_v,
			      float   *p_float_v,
			      char    **codenames,
			      int     *codevalues,
			      char    *filename,
			      int     debug
			      );


/* scanning functions ...*/

void grd3d_scan_nxyz_roff_bin_grid (
				    int     *nx,
				    int     *ny,
				    int     *nz,
				    int     *nsubs,
				    char    *filename,
				    int     debug
				    );

int grd3d_scan_par_roff_bin (
			     char    *parname,
			     char    *filename,
			     int     *ndcodes,
			     int     debug
			     );


void grd3d_import_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zgrd3d_v,
			  int     *p_actnum_v,
			  char    *filename,
			  int     debug
			  );

void grd3d_import_eclipse_grdecl_param (
					int     nx,
					int     ny,
					int     nz,
					char    *param_name,
					float   *p_float_v,
					char    *filename,
					int     debug
					);



int grd3d_import_storm_binary (
			       int   m_action,
			       int   *nx,
			       int   *ny,
			       int   *nz,
			       int   nxyz,
			       int   *num_active,
			       float *grd3d_v,
			       int   *actnum_v,
			       char  *filename,
			       int   debug
			       ); 

/*
int grd3d_import_storm_param (
			      int    M_action,
			      int    nx,
			      int    ny,
			      int    nz,
			      int    inxyz,
			      int    *int_v,
			      double *double_v,
			      char   *filename,
			      int    debug
			      );			       

*/
void grd3d_import_eclipse_param (
				 int    ftype,
				 int    ntstep,
				 int    nxyz,
				 int    *nactive,
				 char   *keyword,
				 char   *filename,
				 int    *ivector_v,
				 float  *fvector_v,
				 double *dvector_v,
				 char   **cvector_v,
				 int    *lvector_v,
				 int    debug
				 );

/* 
 * ============================================================================
 * Exporting 3D grids and parameters
 * ============================================================================
 */


/* new 4 corner pr cell version: */
void grd3d_export_roff_grid (
			    int     mode,
			    int     nx,
			    int     ny,
			    int     nz,
			    int     num_subgrds,
			    int     isub_to_export,
			    float   xscale,
			    float   yscale,
			    float   zscale,
			    float   *p_coord_v,
			    float   *p_zgrd3d_v,
			    int     *p_actnum_v,
			    int     *p_subgrd_v,
			    char    *filename,
			    int     debug
			    );



void grd3d_export_roff_param (
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

void grd3d_export_roff_pstart (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,	
			      char    *filename,	
			      int     debug	
			      );



/* write endtag; must follow other export_grd3d_roff... routines */
void grd3d_export_roff_end (
			    int     mode,
			    char    *filename,
			    int     debug
			    );

void grd3d_export_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zgrd3d_v,
			  int     *p_actnum_v,
			  char    *filename,
			  int     debug
			  );





void grd3d_export_eclipse_grid (
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


void grd3d_export_eclipse_pstart (
				  int     mode,
				  int     nx,
				  int     ny,
				  int     nz,
				  int     *p_actnum_v,
				  char    *filename,
				  int     debug
				  );

void grd3d_export_eclipse_param (
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



/* 
 * ============================================================================
 * Pure Map functions
 * ============================================================================
 */

int grd2d_median_filter (
			 float *grd2d_in_v,
			 int nx,
			 int ny,
			 int nradius,
			 int mode
			 );

void grd2d_set_value (
		     int nx,
		     int ny,
		     float *p_zval_v,
		     float value,
		     int debug
		     );

void grd2d_operation_map (
			  int   nx, 
			  int   ny, 
			  float *p_grd2d1_v,
			  float *p_grd2d2_v,
			  int   iop,
			  int   debug
			  );


void grd2d_getcontour(
		      float z0,
		      int nx,
		      int ny,
		      float xstep, 
		      float ystep,
		      float xmin, 
		      float ymin, 
		      float *p_grd2d_v,
		      int debug
		      );


int grd2d_get_corners_xy(
			 float x,
			 float y,			 
			 int nx,
			 int ny,
			 float xstep, 
			 float ystep,
			 float xmin, 
			 float ymin, 
			 float *p_grd2d_v,			 
			 int debug
			 );

/* 
 * ============================================================================
 * Map and point functions
 * ============================================================================
 */

void grd2d_chk_point_between(
                             float x,
                             float y,
                             float z,

                             int nx1,
                             int ny1,
                             float xstep1, 
                             float ystep1,
                             float xmin1, 
                             float ymin1, 
                             float *p_grd2d1_v,
                             
                             int nx2,
                             int ny2,
                             float xstep2, 
                             float ystep2,
                             float xmin2, 
                             float ymin2, 
                             float *p_grd2d2_v,
                             
                             int   *outside,
                             float *zdiff,

                             int debug
                             );

 


/* 
 * ============================================================================
 * Pure 3D grid functions
 * ============================================================================
 */


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

void grd3d_remap_param_g2g(
			   int   nx1,
			   int   ny1,
			   int   nz1,
			   int   nx2,
			   int   ny2,
			   int   nz2,
			   int   isub,
			   int   num_subgrds,
			   int   *p_isubgrd_v,
			   char  *ptype,
			   int   *p_int1_v,
			   float *p_float1_v,
			   int   *p_int2_v,
			   float *p_float2_v,
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

void grd3d_match_zonebounds (
			     int   nx,
			     int   ny,
			     int   nz1,
			     int   nz2,
			     float *p_zcorn1_v,
			     int   *p_actnum1_v,
			     float *p_zcorn2_v,
			     int   *p_actnum2_v,
			     int   iflag,
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

void grd3d_calc_dz (
		    int nx,
		    int ny,
		    int nz,
		    float *p_grd3d_v,
		    float *p_dz_v,
		    int   debug
		    );

void grd3d_calc_sum_dz(
		      int nx,
		      int ny,
		      int nz,
		      float *p_grd3d_v,
		      int   *p_actnum_v,
		      float *p_dz_v,
		      int   debug
		      );

void grd3d_calc_abase(
		      int option,
		      int nx,
		      int ny,
		      int nz,
		      float *p_dz_v,
		      float *p_base_v,
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


void grd3d_print_cellinfo (
			   int x1,
			   int x2,
			   int y1,
			   int y2,
			   int z1,
			   int z2,
			   int nx,
			   int ny,
			   int nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *actnum_v,
			   int debug
			   );


int grd3d_point_in_cell(
			int ibstart,
			float x,
			float y,
			float z,
			int nx,
			int ny,
			int nz,
			float *p_coor_v,
			float *p_zcorn_v,
			int   debug
			);




/* 
 * ============================================================================
 * Pure point polygons ... functions
 * ============================================================================
 */

void points_import_rms (
			double *p_xp_v,
			double *p_yp_v,
			double *p_zp_v,
			char   *file,
			int    debug
			);


void points_export_irap_ascii (
			       double *p_xp_v,
			       double *p_yp_v,
			       double *p_zp_v,
			       char   *file,
			       int    debug
			       );

/* 
 * ============================================================================
 * Combined 3D grid and map functions
 * ============================================================================
 */

void grd3d_adj_z_from_map (
			   int nx,
			   int ny,
			   int nz,
			   float *p_coord_v,
			   float *p_zcorn_v,
			   int   *p_actnum_v,
			   int mx,
			   int my,
			   float xmin,
			   float xstep,
			   float ystart,
			   float ystep,
			   float *p_grd2d_v,
			   int iflag,
			   int debug
			   );


void grd2d_wiener_from_grd3d (
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
			      float *p_grd2d_v,
			      int   debug
			      );


/* 
 * ============================================================================
 * Combined 2D (map) grid and polygon functions
 * ============================================================================
 */



void grd2d_polygon_mask(
			int nx,
			int ny,
			float xstep, 
			float ystep,
			float xmin, 
			float ymin, 
			float *p_grd2d_v,
			double *p_xp_v,
			double *p_yp_v,
			int debug
			);









/* 
 * ============================================================================
 *                               OTHER STUFF
 * ============================================================================
 */


int ijk2ib (
	    int i, 
	    int j, 
	    int k, 
	    int nx, 
	    int ny, 
	    int nz,
	    int ia_offset
	    );

void ib2ijk (
	     int   ib,
	     int   *i,
	     int   *j,
	     int   *k,
	     int   nx,
	     int   ny,
	     int   nz,
	     int   ia_start
	     );


float z_interpolate_map (
			 float *x_v,
			 float *y_v,
			 float *z_v,
			 float x,
			 float y,
			 int method,
			 int debug
			 );


/* TMP - should be in libgtc_ */

int chk_point_in_cell (
		       float x,
		       float y,
		       float z,
		       float coor[],
		       int   imethod,
		       int   debug
		       );



/* 
 * ============================================================================
 *                                  THE END
 * ============================================================================
 */





