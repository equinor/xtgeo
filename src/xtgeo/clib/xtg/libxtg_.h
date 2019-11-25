#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define XTGDEBUG 0   /* this is a tmp solution */


void x_fgets(char *, int, FILE *);
void x_fread (void *, size_t , size_t , FILE *, char *, int);


int x_nint(double);


int x_cmp_sort (
    const void *vp,
    const void *vq
    );

void x_mapaxes (
    int   mode,
    double *x,
    double *y,
    const double x1,
    const double y1,
    const double x2,
    const double y2,
    const double x3,
    const double y3,
    int   option
    );


void x_stretch_points(
    int   np,
    double *p1_v,
    double *p2_v,
    int method,
    int debug
    );

void x_basicstats (
    int n,
    double undef,
    double *v,
    double *min,
    double *max,
    double *avg,
    int debug
    );

void x_basicstats2 (
    int n,
    float undef,
    float *v,
    float *min,
    float *max,
    float *avg,
    int debug
    );

void x_stretch_vector (
    double *z_v,
    int   nz,
    int   k1,
    int   k2,
    double z1n,
    double z2n,
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

int x_vector_extrapol2 (
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
    double x,
    double y,
    double z,
    double coor[],
    int   imethod,
    int   debug
    );

void x_2d_rect_corners(
    double x, double y, double xinc, double yinc,
    double rot,
    double result[8], int debug);

int x_kvpt3s (
    double pp[],
    double tri[][3],
    int   *ier
    );

void x_kmgmps (
    double a[][3],
    int l[],
    double prmn,
    int m,
    int n,
    double eps,
    int *ier
    );

void x_kmsubs (
    double x[],
    double a[][3],
    int m,
    int n,
    double b[],
    int l[],
    int *ier
    );

void x_regular_geom (
    double xmin,
    double xinc,
    double ymin,
    double yinc,
    double zmin,
    double zinc,
    int   nx,
    int   ny,
    int   nz,
    int   i,
    int   j,
    int   z,
    double rot_azi_deg,
    double *xcenter,
    double *ycenter,
    double *zcenter,
    double *corners_v,
    int   flag,
    int   debug
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

double x_vector_len3d (
    double x1,
    double x2,
    double y1,
    double y2,
    double z1,
    double z2
    );


int x_interp_cube_nodes (
    double *x_v,
    double *y_v,
    double *z_v,
    float *p_v,
    double x,
    double y,
    double z,
    float *value,
    int method,
    int debug
    );

int x_plane_normalvector(double *points_v, double *nvector, int option,
			 int debug);
int x_isect_line_plane(double *nvector, double *line_v, double *point_v,
		       int option, int debug);

double x_angle_vectors(double *avec, double *bvec, int debug);

double x_sample_z_from_xy_cell(double *cell_v, double x, double y,
			       int option, int option2, int debug);

int x_point_line_dist(double x1, double y1, double z1,
		      double x2, double y2, double z2,
		      double x3, double y3, double z3,
		      double *distance,
		      int option1, int option2, int debug);

int x_point_line_pos(double x1, double y1, double z1,
                     double x2, double y2, double z2,
                     double x3, double y3, double z3,
                     double *x, double *y, double *z,
                     double *rel,
                     int option1, int debug);

FILE *x_fopen(const char *filename, const char *mode, int debug);

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
    double  *p_coord_v,
    double  *p_zgrd3d_v,
    int     *p_actnum_v,
    int     *p_subgrd_v,
    int     nnsub,
    char    *filename,
    int     debug
    );

void _grd3d_imp_roff_bin_grd (
    int     *num_act,
    int     *num_subgrds,
    double  *p_coord_v,
    double  *p_zgrd3d_v,
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
    double   *p_double_v,
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
    double   *p_double_v,
    char    **codenames,
    int     *codevalues,
    char    *filename,
    int     debug
    );

int _grd3d_roffbinstring(char *bla, FILE *fc);

float _grd3d_getfloatvalue(char *name, FILE *fc);

int _grd3d_getintvalue(char *name, FILE *fc);

void _grd3d_getfloatarray(float *array, int num, FILE *fc);

// TODO shall be UNSIGNED char?
void _grd3d_getbytearray(char *array, int num, FILE *fc);

void _grd3d_getintarray(int *array, int num, FILE *fc);

void _grd3d_getchararray(char **array, int num, FILE *fc);

void _grd3d_getchararray1d(char *array, int num, FILE *fc);

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
    double  *p_coord_v,
    double  *p_zgrd3d_v,
    int     *p_actnum_v,
    int     *p_subgrd_v,
    int     debug
    );

/* new from sep 2016 */
int   x_roffbinstring(char *bla, FILE *fc);
float x_roffgetfloatvalue(char *name, FILE *fc);
int   x_roffgetintvalue(char *name, FILE *fc);
void  x_roffgetfloatarray(float *array, int num, FILE *fc);
void  x_roffgetbytearray(unsigned char *array, int num, FILE *fc);
void  x_roffgetintarray(int *array, int num, FILE *fc);
void  x_roffgetchararray(char *array, int num, FILE *fc);


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



int u_read_segy_bitem (
    int nc,
    int ic,
    void *ptr,
    size_t size,
    size_t nmemb,
    FILE *fc,
    FILE *fout,
    int swap,
    char *txt,
    int *nb,
    int option
    );


void u_ibm_to_float (
    int *from,
    int *to,
    int n,
    int endian,
    int swap
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



int u_eightletter (
    char *cs
    );
