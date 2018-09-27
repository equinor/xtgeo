#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>


int xtg_speak(
	      char *,
	      int,
	      char *,
	      ...
	      );

int xtg_warn(
	      char *,
	      int,
	      char *,
	      ...
	      );

int xtg_error(
	      char *,
	      char *,
	      ...
	      );

int xtg_shout(
	      char *,
	      char *,
	      ...
	      );

int xtgverbose(int);

int x_nint(float);

int x_cmp_sort (
		const void *vp, 
		const void *vq
		);

void x_mapaxes (
		int   mode,
		float *x,
		float *y,
		const float x1,
		const float y1,
		const float x2,
		const float y2,
		const float x3,
		const float y3,
		float xmin,
		float xmax,
		float ymin,
		float ymax,
		int   option,
		int   debug
		);


void x_stretch_points( 
		      int   np,
		      float *p1_v,
		      float *p2_v,
		      int method,
		      int debug
		       );


void x_stretch_vector (
		       float *z_v,
		       int   nz,
		       int   k1,
		       int   k2,
		       float z1n,
		       float z2n,
		       int debug
		       );



int x_vector_extrapol (
		       double x1,
		       double y1,
		       double z1,
		       double *x2,
		       double *y2,
		       double *z2,
		       double dlen,
		       double xang,
		       int   debug
		       );

int x_chk_point_in_cell (
			 float x,
			 float y,
			 float z,
			 float coor[],
			 int   imethod,
			 int   debug
			 );

int x_kvpt3s (
	      float pp[], 
	      float tri[][3],
	      int   *ier
	      );

void x_kmgmps (
	       float a[][3], 
	       int l[], 
	       float prmn, 
	       int m, 
	       int n, 
	       float eps, 
	       int *ier
	       );

void x_kmsubs (
	       float x[],
	       float a[][3],
	       int m,
	       int n,
	       float b[],
	       int l[],
	       int *ier
	       );

void x_regular_geom (
                     float xmin,
                     float xinc,
                     float ymin,
                     float yinc,
                     float zmin,
                     float zinc,
                     int   nx,
                     int   ny,
                     int   nz,
                     int   i,
                     int   j,
                     int   z,
                     float rot_azi_deg,
                     float *xcenter,
                     float *ycenter,
                     float *zcenter,
                     float *corners_v,
                     int   flag,
                     int   debug
                     );

int x_vector_info2 (
		    float x1,
		    float x2,
		    float y1,
		    float y2,
		    float *vlen,
		    float *azimuth_radian,
		    float *azimuth_degrees,
		    int   debug
		    );

double x_diff_angle (
		     double ang1,
		     double ang2,
		     int    option,
		     int    debug
		     );


int x_vector_linint (
		    double x1,
		    double y1,
		    double z1,
		    double x2,
		    double y2,
		    double z2,
		    double dlen,
		    double *xn,
		    double *yn,
		    double *zn,
		    int   debug
		     );

/*
 *-----------------------------------------------------------------------------
 * Byte swapping test
 *-----------------------------------------------------------------------------
 */

int x_swap_check();

int x_byteorder(int);

/******************************************************************************
  FUNCTION: SwapEndian
  PURPOSE: Swap the byte order of a structure
  EXAMPLE: float F=123.456;; SWAP_FLOAT(F);
******************************************************************************/

#define SWAP_INT(Var)    Var = *(int*)           SwapEndian((void*)&Var, sizeof(int))
#define SWAP_SHORT(Var)  Var = *(short*)         SwapEndian((void*)&Var, sizeof(short))
#define SWAP_USHORT(Var) Var = *(unsigned short*)SwapEndian((void*)&Var, sizeof(short))
#define SWAP_LONG(Var)   Var = *(long*)          SwapEndian((void*)&Var, sizeof(long))
#define SWAP_ULONG(Var)  Var = *(unsigned long*) SwapEndian((void*)&Var, sizeof(long))
#define SWAP_FLOAT(Var)  Var = *(float*)         SwapEndian((void*)&Var, sizeof(float))
#define SWAP_DOUBLE(Var) Var = *(double*)        SwapEndian((void*)&Var, sizeof(double))

extern void *SwapEndian(void* Addr, const int Nb);





/*
 *-----------------------------------------------------------------------------
 * No-public grd3d routines connected to ROFF format
 *-----------------------------------------------------------------------------
 */

void _grd3d_imp_roff_asc_grd (
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

void _grd3d_imp_roff_bin_grd (
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


void _grd3d_imp_roff_asc_prp (
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

void _grd3d_imp_roff_bin_prp (
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

int _grd3d_roffbinstring(char *bla, FILE *fc);

float _grd3d_getfloatvalue(char *name, FILE *fc);

int _grd3d_getintvalue(char *name, FILE *fc);

void _grd3d_getfloatarray(float *array, int num, FILE *fc);

void _grd3d_getbytearray(char *array, int num, FILE *fc);

void _grd3d_getintarray(int *array, int num, FILE *fc);

void _grd3d_getchararray(char **array, int num, FILE *fc);

void _grd3d_roff_to_xtg_grid (
			      int     nx,
			      int     ny,
			      int     nz,
			      float   xoffset,
			      float   yoffset,
			      float   zoffset,
			      float   xscale,
			      float   yscale,
			      float   zscale,
			      float   *cornerlines_v,
			      char    *splitenz_v,
			      float   *zdata_v,
			      int     *num_act,
			      int     *num_subgrds,
			      float   *p_coord_v,
			      float   *p_zgrd3d_v,
			      int     *p_actnum_v,
			      int     *p_subgrd_v,
			      int     debug
			      );


/*
 *-----------------------------------------------------------------------------
 * No-public grd3d routines for other issues
 *-----------------------------------------------------------------------------
 */


int _grd3d_fnd_near_cell (
			  int     *i,
			  int     *j,
			  int     *k,
			  int     nx,
			  int     ny,
			  int     nz,
			  float   *p_coord_v,
			  float   *p_zgrd3d_v,
			  int     *p_actnum_v,
			  float   x,
			  float   y,
			  float   z,
			  int     debug
			  );

int _grd3d_fnd_cell (
		     int     *i,
		     int     *j,
		     int     *k,
		     int     nx,
		     int     ny,
		     int     nz,
		     float   *p_coord_v,
		     float   *p_zgrd3d_v,
		     int     *p_actnum_v,
		     float   x,
		     float   y,
		     float   z,
		     int     debug
		     );




int u_wri_ecl_bin_record (
			  char    *cname,
			  char    *ctype,
			  int     reclen,
			  int     *tmp_int_v,
			  float   *tmp_float_v,
			  double  *tmp_double_v,
			  char    **tmp_string_v,
			  int     *tmp_logi_v,
			  FILE    *fc,
			  int     debug
			  );

int u_read_ecl_bin_record (
			   char    *cname,
			   char    *ctype,
			   int     *reclen,
			   int max_alloc_int,
			   int max_alloc_float,
			   int max_alloc_double,
			   int max_alloc_char,
			   int max_alloc_logi,
			   int     *tmp_int_v,
			   float   *tmp_float_v,
			   double  *tmp_double_v,
			   char    **tmp_string_v,
			   int     *tmp_logi_v,
			   FILE    *fc,
			   int     debug
			   );

int u_read_ecl_asc_record (
			   char    *cname,
			   char    *ctype,
			   int     *reclen,
			   int     *tmp_int_v,
			   float   *tmp_float_v,
			   double  *tmp_double_v,
			   char    **tmp_string_v,
			   int     *tmp_logi_v,
			   FILE    *fc,
			   int     debug
			   );

int u_eightletter (
		   char *cs
		   );
