/*
 * -------------------------------------------------------------------------------------
 * Python stuff SWIG (examples...; see cxtgeo.i):
 * -------------------------------------------------------------------------------------
 * int    *swig_int_out_p1,         // Value of output pointers
 * double *swig_np_dbl_aout_v1,     // *p_xx_v to update argout for numpy
 * long   n_swig_np_dbl_aout_v1,    // length of nmpy array
 *
 * char    *swig_bnd_char_10k,      // bounded characters up to 10000
 *
 */

#define _GNU_SOURCE 1

#pragma once

#include "xtg.h"
#include <stdint.h>
#include <stdio.h>

#define PI 3.14159265358979323846
#define M_PI 3.14159265358979323846
#define PIHALF 1.57079632679489661923

#define FLOATEPS 1.0E-05
#define VERYLARGEFLOAT 10E30
#define VERYSMALLFLOAT -10E30

/* better? */
#define VERYLARGEPOSITIVE 10E30
#define VERYLARGENEGATIVE -10E30

/*
 * -------------------------------------------------------------------------------------
 * Maps etc. Undef values set to very high values. Map values > UNDEF_MAP_LIMIT
 * are undefined
 * -------------------------------------------------------------------------------------
 */

/* general limits in XTGeo are recoemmended for all XTGeo data types! */
#define UNDEF 10E32
#define UNDEF_LIMIT 9.9E32
#define UNDEF_INT 2000000000
#define UNDEF_INT_LIMIT 1999999999

/* specific list */
#define UNDEF_MAP 10E32
#define UNDEF_INT_MAP 2000000000
#define UNDEF_MAP_LIMIT 9.9E32
#define UNDEF_INT_MAP_LIMIT 1999999999
#define UNDEF_MAP_STORM -999.00
#define UNDEF_MAP_IRAP 9999900.0000 /* Irap ascii */
#define UNDEF_MAP_IRAPB 1E30        /* Irap binary */
#define UNDEF_MAP_ZMAP -99999.0     /* ZMAP binary (but can vary) */
#define UNDEF_MAP_IRAPB_LIMIT 0.99E30
#define UNDEF_CUBE_RMS -9999.00
#define UNDEF_POINT 10E32
#define UNDEF_POINT_LIMIT 9.9E32
#define UNDEF_POINT_RMS -999.0000
#define UNDEF_POINT_IRAP 999.0000
#define LAST_POINT -10E30
#define LAST_POINT_LIMIT -9.9E30

#define MAXPSTACK 5000
#define ROFFSTRLEN 100
#define ECLNAMELEN 9
#define ECLTYPELEN 5
#define ECLINTEHEADLEN 240
#define ECLDOUBHEADLEN 160
#define ECLLOGIHEADLEN 80

#define ECLNUMBLOCKLEN 1000
#define ECLCHABLOCKLEN 105

#define UNDEF_ROFFBYTE 255
#define UNDEF_ROFFINT -999
#define UNDEF_ROFFFLOAT -999.0

/* for eclipse ascii export e.g. GRDECL */
#define UNDEF_ECLINT 0
#define UNDEF_ECLFLOAT 0

/* eclipse and roff binary read max keywords and dates */
#define MAXKEYWORDS 1000000
#define MAXDATES 1000

/*
 *======================================================================================
 * GENERAL XTGEO
 *======================================================================================
 */

FILE *
xtg_fopen(const char *filename, const char *mode);

FILE *
xtg_fopen_bytestream(char *swig_bytes, long swig_bytes_len, const char *mode);

int
xtg_fseek_start(FILE *fhandle);

int
xtg_fflush(FILE *fhandle);

long
xtg_ftell(FILE *fhandle);

int
xtg_fclose(FILE *fhandle);

int
xtg_get_fbuffer(FILE *fhandle, char *swig_bytes, long swig_bytes_len);

/*
 *======================================================================================
 * GENERAL FUNCTIONS
 *======================================================================================
 */

double
x_interp_map_nodes(double *x_v,
                   double *y_v,
                   double *z_v,
                   double x,
                   double y,
                   int method);

long
x_ijk2ib(long i, long j, long k, long nx, long ny, long nz, int ia_start);

long
x_ijk2ic(long i, long j, long k, long nx, long ny, long nz, int ia_start);

void
x_ib2ijk(long ib,
         int *swig_int_out_p1,  // *i
         int *swig_int_out_p2,  // *j
         int *swig_int_out_p3,  // *k
         int nx,
         int ny,
         int nz,
         int ia_start);

void
x_ic2ijk(long ic,
         int *swig_int_out_p1,  // *i
         int *swig_int_out_p2,  // *j
         int *swig_int_out_p3,  // *k
         int nx,
         int ny,
         int nz,
         int ia_start);

void
x_vector_info2(double x1,
               double x2,
               double y1,
               double y2,
               double *swig_dbl_out_p1,  // *vlen
               double *swig_dbl_out_p2,  // *xangle_radian
               double *swig_dbl_out_p3,  // *xangle_degrees
               int option);

int
x_vector_linint2(double x0,
                 double y0,
                 double z0,
                 double x1,
                 double y1,
                 double z1,
                 double dist,
                 double *swig_dbl_out_p1,  // *xr
                 double *swig_dbl_out_p2,  // *yr
                 double *swig_dbl_out_p3,  // *zr
                 int option);

double
x_diff_angle(double ang1, double ang2, int option);

double
x_avg_angles(double *swig_np_dbl_in_v1,  // *angles
             long n_swig_np_dbl_in_v1);  // nsize,

void
x_conv_int2float(int n, int *v1, float *v2);

void
x_conv_int2double(int n, int *v1, double *v2);

void
x_conv_double2float(int n, double *v1, float *v2);

double
x_rotation_conv(double ain, int aimode, int mode, int option);

double
x_tetrahedron_volume(double *swig_np_dbl_inplaceflat_v1,
                     long n_swig_np_dbl_inplaceflat_v1);

int
x_point_in_tetrahedron(double x0,
                       double y0,
                       double z0,
                       double *swig_np_dbl_inplaceflat_v1,
                       long n_swig_np_dbl_inplaceflat_v1);

int
x_point_in_hexahedron(double x0,
                      double y0,
                      double z0,
                      double *swig_np_dbl_inplaceflat_v1,
                      long n_swig_np_dbl_inplaceflat_v1,
                      int method);

double
x_vectorpair_angle3d(double *swig_np_dbl_in_v1,
                     long n_swig_np_dbl_in_v1,
                     double *swig_np_dbl_in_v2,
                     long n_swig_np_dbl_in_v2,
                     double *swig_np_dbl_in_v3,
                     long n_swig_np_dbl_in_v3,
                     int degrees,
                     int option);

int
x_minmax_cellangles_topbase(double *swig_np_dbl_in_v1,
                            long n_swig_np_dbl_in_v1,
                            double *swig_dbl_out_p1,
                            double *swig_dbl_out_p2,
                            int option,
                            int degrees);

int
x_minmax_cellangles_sides(double *swig_np_dbl_in_v1,
                          long n_swig_np_dbl_in_v1,
                          double *swig_dbl_out_p1,
                          double *swig_dbl_out_p2,
                          int degrees);

double
x_hexahedron_volume(double *corners, long ndim, int presision);

/*
 * =====================================================================================
 * surf_* for regular maps/surfaces
 * =====================================================================================
 */

/* the swig* names are for eventual typemaps signatures in the cxtgeo.i
 * file to SWIG */
int
surf_import_irap_bin(FILE *fhandle,
                     int mode,
                     int *swig_int_out_p1,         // *p_mx
                     int *swig_int_out_p2,         // *p_my
                     long *swig_lon_out_p1,        // *p_ndef
                     double *swig_dbl_out_p1,      // *p_xori
                     double *swig_dbl_out_p2,      // *p_yori
                     double *swig_dbl_out_p3,      // *p_xinc
                     double *swig_dbl_out_p4,      // *p_yinc
                     double *swig_dbl_out_p5,      // *p_rot
                     double *swig_np_dbl_aout_v1,  // *p_map_v
                     long n_swig_np_dbl_aout_v1,   // nmap
                     int option);

int
surf_import_irap_ascii(FILE *fc,
                       int mode,
                       int *swig_int_out_p1,         // *nx
                       int *swig_int_out_p2,         // *ny
                       long *swig_lon_out_p1,        // *ndef
                       double *swig_dbl_out_p1,      // *xori
                       double *swig_dbl_out_p2,      // *yori
                       double *swig_dbl_out_p3,      // *xinc
                       double *swig_dbl_out_p4,      // *yinc
                       double *swig_dbl_out_p5,      // *rot
                       double *swig_np_dbl_aout_v1,  // *p_map_v
                       long n_swig_np_dbl_aout_v1,   // nmap
                       int option);

int
surf_import_ijxyz(FILE *fc,
                  int mode,
                  int *swig_int_out_p1,         // *nx
                  int *swig_int_out_p2,         // *ny
                  long *swig_lon_out_p1,        // *ndef,
                  double *swig_dbl_out_p1,      // *xori,
                  double *swig_dbl_out_p2,      // *yori,
                  double *swig_dbl_out_p3,      // *xinc,
                  double *swig_dbl_out_p4,      // *yinc,
                  double *swig_dbl_out_p5,      // *rot,
                  int *swig_np_int_aout_v1,     // *ilines,
                  long n_swig_np_int_aout_v1,   // nilines or ncol,
                  int *swig_np_int_aout_v2,     // *xlines,
                  long n_swig_np_int_aout_v2,   // nxlines or nrow,
                  double *swig_np_dbl_aout_v1,  // *p_map_v
                  long n_swig_np_dbl_aout_v1,   // nmap = nrow*ncol
                  int *swig_int_out_p3,         // yflip
                  int option);

int
surf_import_ijxyz_tmpl(FILE *fc,
                       int *swig_np_int_in_v1,       // *ilines,
                       long n_swig_np_int_in_v1,     // nilines or ncol,
                       int *swig_np_int_in_v2,       // *xlines,
                       long n_swig_np_int_in_v2,     // nxlines or nrow,
                       double *swig_np_dbl_aout_v1,  // *p_map_v
                       long n_swig_np_dbl_aout_v1,   // nmap = nrow*ncol
                       int option);

void
surf_import_petromod_bin(FILE *fhandle,
                         int mode,
                         float undef,
                         char *swig_bnd_char_10k,  // *dsc
                         int mx,
                         int my,
                         double *swig_np_dbl_aout_v1,  // *p_map_v
                         long n_swig_np_dbl_aout_v1);  // nmap

int
surf_export_irap_bin(FILE *fc,
                     int mx,
                     int my,
                     double xori,
                     double yori,
                     double xinc,
                     double yinc,
                     double rot,
                     double *swig_np_dbl_in_v1,  // *p_map_v,
                     long n_swig_np_dbl_in_v1,   // mxy
                     int option);

int
surf_export_irap_ascii(FILE *fc,
                       int mx,
                       int my,
                       double xori,
                       double yori,
                       double xinc,
                       double yinc,
                       double rot,
                       double *swig_np_dbl_in_v1,  // *p_map_v
                       long n_swig_np_dbl_in_v1,   // mxy
                       int option);

int
surf_export_zmap_ascii(FILE *fc,
                       int mx,
                       int my,
                       double xori,
                       double yori,
                       double xinc,
                       double yinc,
                       double *swig_np_dbl_in_v1,  // *p_map_v
                       long n_swig_np_dbl_in_v1,   // mxy
                       double zmin,
                       double zmax,
                       int option);

int
surf_export_storm_bin(FILE *fc,
                      int mx,
                      int my,
                      double xori,
                      double yori,
                      double xinc,
                      double yinc,
                      double *swig_np_dbl_in_v1,  // *p_map_v
                      long n_swig_np_dbl_in_v1,   // mxy
                      double zmin,
                      double zmax,
                      int option);

void
surf_export_petromod_bin(FILE *fc,
                         char *dsc,
                         double *swig_np_dbl_in_v1,  // *surfzv
                         long n_swig_np_dbl_in_v1);  // nsurf

int
surf_export_ijxyz(FILE *fc,
                  int mx,
                  int my,
                  double xori,
                  double yori,
                  double xinc,
                  double yinc,
                  double rot,
                  int yflip,
                  int *swig_np_int_in_v1,     // ilines
                  long n_swig_np_int_in_v1,   // ncol
                  int *swig_np_int_in_v2,     // xlines
                  long n_swig_np_int_in_v2,   // nrow
                  double *swig_np_dbl_in_v1,  // *p_map_v
                  long n_swig_np_dbl_in_v1,   // mxy
                  int option);

int
surf_zminmax(int nx, int ny, double *p_map_v, double *zmin, double *zmax);

int
surf_xyz_from_ij(int i,
                 int j,
                 double *swig_dbl_out_p1,  //  *x
                 double *swig_dbl_out_p2,  //  *y
                 double *swig_dbl_out_p3,  //  *z
                 double xori,
                 double xinc,
                 double yori,
                 double yinc,
                 int nx,
                 int ny,
                 int yflip,
                 double rot_deg,
                 double *swig_np_dbl_in_v1,  // *p_map_v,
                 long n_swig_np_dbl_in_v1,   // nn
                 int flag);

int
surf_xyori_from_ij(int i,
                   int j,
                   double x,
                   double y,
                   double *swig_dbl_out_p1,  //  *xori
                   double xinc,
                   double *swig_dbl_out_p2,  //  *yori
                   double yinc,
                   int nx,
                   int ny,
                   int yflip,
                   double rot_deg,
                   int flag);

double
surf_get_z_from_ij(int ic,
                   int jc,
                   double x,
                   double y,
                   int nx,
                   int ny,
                   double xinc,
                   double yinc,
                   double xori,
                   double yori,
                   double *p_map_v,
                   int option);

double
surf_get_z_from_xy(double x,
                   double y,
                   int nx,
                   int ny,
                   double xori,
                   double yori,
                   double xinc,
                   double yinc,
                   int yflip,
                   double rot_deg,
                   double *swig_np_dbl_in_v1,  // *p_map_v
                   long n_swig_np_dbl_in_v1,   // nn
                   int option);

int
surf_get_zv_from_xyv(double *swig_np_dbl_in_v1,  // *xv
                     long n_swig_np_dbl_in_v1,
                     double *swig_np_dbl_in_v2,  // *yv
                     long n_swig_np_dbl_in_v2,
                     double *swig_np_dbl_inplace_v1,  // *zv
                     long n_swig_np_dbl_inplace_v1,
                     int nx,
                     int ny,
                     double xori,
                     double yori,
                     double xinc,
                     double yinc,
                     int yflip,
                     double rot_deg,
                     double *swig_np_dbl_in_v3,  // *p_map_v,
                     long n_swig_np_dbl_in_v3,
                     int option);

int
surf_xy_as_values(double xori,
                  double xinc,
                  double yori,
                  double yinc,
                  int nx,
                  int ny,
                  double rot_deg,
                  double *swig_np_dbl_aout_v1,  // *p_x_v
                  long n_swig_np_dbl_aout_v1,   // nn1
                  double *swig_np_dbl_aout_v2,  // *p_y_v
                  long n_swig_np_dbl_aout_v2,   // nn2
                  int flag);

int
surf_slice_grd3d(int mcol,
                 int mrow,
                 double xori,
                 double xinc,
                 double yori,
                 double yinc,
                 double rotation,
                 int yflip,
                 double *swig_np_dbl_in_v1,  // *p_zslice_v
                 long n_swig_np_dbl_in_v1,
                 double *swig_np_dbl_aout_v1,  // *p_map_v to update argout
                 long n_swig_np_dbl_aout_v1,
                 int ncol,
                 int nrow,
                 int nlay,

                 double *swig_np_dbl_in_v2,  // *coordsv,
                 long n_swig_np_dbl_in_v2,   // ncoord,
                 double *swig_np_dbl_in_v3,  // *zcornsv,
                 long n_swig_np_dbl_in_v3,   // nzcorn,
                 int *swig_np_int_in_v1,     // *actnumsv,
                 long n_swig_np_int_in_v1,   // nactnum,

                 double *p_prop_v,
                 int buffer);

int
surf_resample(int nx1,
              int ny1,
              double xori1,
              double xinc1,
              double yori1,
              double yinc1,
              int yflip1,
              double rota1,
              double *swig_np_dbl_inplaceflat_v1,
              long n_swig_np_dbl_inplaceflat_v1,
              int nx2,
              int ny2,
              double xori2,
              double xinc2,
              double yori2,
              double yinc2,
              int yflip2,
              double rota2,
              double *swig_np_dbl_inplaceflat_v2,
              long n_swig_np_dbl_inplaceflat_v2,
              int option);

int
surf_get_dist_values(double xori,
                     double xinc,
                     double yori,
                     double yinc,
                     int nx,
                     int ny,
                     double rot_deg,
                     double x0,
                     double y0,
                     double azimuth,
                     double *swig_np_dbl_inplace_v1,  // *p_map_v INPLACE
                     long n_swig_np_dbl_inplace_v1,
                     int flag);

int
surf_slice_cube(int ncx,
                int ncy,
                int ncz,
                double cxori,
                double cxinc,
                double cyori,
                double cyinc,
                double czori,
                double czinc,
                double crotation,
                int yflip,
                float *swig_np_flt_in_v1,  // *p_cubeval_v
                long n_swig_np_flt_in_v1,
                int mx,
                int my,
                double xori,
                double xinc,
                double yori,
                double yinc,
                int mapflip,
                double mrotation,
                double *swig_np_dbl_in_v1,  // *p_zslice_v
                long n_swig_np_dbl_in_v1,
                double *swig_np_dbl_aout_v1,  // *p_map_v to update argout
                long n_swig_np_dbl_aout_v1,
                int option1,
                int option2);

int
surf_slice_cube_v3(int ncol,
                   int nrow,
                   int nlay,
                   double czori,
                   double czinc,
                   float *swig_np_flt_inplaceflat_v1,
                   long n_swig_np_flt_inplaceflat_v1,
                   double *swig_np_dbl_inplaceflat_v1,
                   long n_swig_np_dbl_inplaceflat_v1,
                   double *swig_np_dbl_inplaceflat_v2,
                   long n_swig_np_dbl_inplaceflat_v2,
                   mbool *swig_np_boo_inplaceflat_v1,  // *maskv,
                   long n_swig_np_boo_inplaceflat_v1,
                   int optnearest,
                   int optmask);

int
surf_stack_slice_cube(int ncol,
                      int nrow,
                      int nlay,
                      int nstack,
                      double czori,
                      double czinc,
                      float *cubevalsv,
                      double **stack,
                      mbool **rmask,
                      int optnearest,
                      int optmask);

int
surf_slice_cube_window(int ncx,
                       int ncy,
                       int ncz,
                       double cxori,
                       double cxinc,
                       double cyori,
                       double cyinc,
                       double czori,
                       double czinc,
                       double crotation,
                       int yflip,
                       float *swig_np_flt_in_v1,  // *p_cubeval_v
                       long n_swig_np_flt_in_v1,  // ncube
                       int mx,
                       int my,
                       double xori,
                       double xinc,
                       double yori,
                       double yinc,
                       int mapflip,
                       double mrotation,
                       double *swig_np_dbl_in_v1,  // *p_map_v
                       long n_swig_np_dbl_in_v1,   // nmap
                       double zincr,
                       int nzincr,
                       double *swig_np_dbl_aout_v1,  // *p_attrs_v: update
                       long n_swig_np_dbl_aout_v1,   // nattrsmap
                       int nattr,
                       int option1,
                       int option2);

// WIP
int
surf_cube_attr_intv(int ncol,
                    int nrow,
                    int nlay,
                    double czori,
                    double czinc,
                    float *swig_np_flt_inplaceflat_v1,
                    long n_swig_np_flt_inplaceflat_v1,
                    double *swig_np_dbl_inplaceflat_v1,
                    long n_swig_np_dbl_inplaceflat_v1,
                    double *swig_np_dbl_inplaceflat_v2,
                    long n_swig_np_dbl_inplaceflat_v2,
                    mbool *swig_np_boo_inplaceflat_v1,  // *maskv1,
                    long n_swig_np_boo_inplaceflat_v1,
                    mbool *swig_np_boo_inplaceflat_v2,  // *maskv2,
                    long n_swig_np_boo_inplaceflat_v2,
                    double slicezinc,
                    int ndiv,
                    int ndivdisc,
                    double *swig_np_dbl_inplaceflat_v3,
                    long n_swig_np_dbl_inplaceflat_v3,
                    int optnearest,
                    int optmask,
                    int optprogress,
                    double maskthreshold,
                    int optsum);

void
surf_sample_grd3d_lay(int nx,
                      int ny,
                      int nz,
                      double *swig_np_dbl_in_v1,  // *coordsv,
                      long n_swig_np_dbl_in_v1,   // ncoord,
                      double *swig_np_dbl_in_v2,  // *zcornsv,
                      long n_swig_np_dbl_in_v2,   // nzcorn,
                      int *swig_np_int_in_v1,     // *actnumsv,
                      long n_swig_np_int_in_v1,   // nactnum,
                      int klayer,
                      int mx,
                      int my,
                      double xori,
                      double xstep,
                      double yori,
                      double ystep,
                      double rotation,
                      double *swig_np_dbl_inplace_v1,
                      long n_swig_np_dbl_inplace_v1,
                      double *swig_np_dbl_inplace_v2,
                      long n_swig_np_dbl_inplace_v2,
                      double *swig_np_dbl_inplace_v3,
                      long n_swig_np_dbl_inplace_v3,
                      int option);

int
surf_setval_poly(double xori,
                 double xinc,
                 double yori,
                 double yinc,
                 int ncol,
                 int nrow,
                 int yflip,
                 double rot_deg,
                 double *swig_np_dbl_inplace_v1,  // *p_map_v
                 long n_swig_np_dbl_inplace_v1,   // nmap
                 double *swig_np_dbl_in_v1,       // *p_xp_v,
                 long n_swig_np_dbl_in_v1,        // npolx
                 double *swig_np_dbl_in_v2,       // *p_yp_v,
                 long n_swig_np_dbl_in_v2,        // npoly
                 double value,
                 int flag);
/*
 *======================================================================================
 * POLYGON/POINTS
 *======================================================================================
 */

int
pol_chk_point_inside(double x, double y, double *p_xp_v, double *p_yp_v, int np);

int
pol_do_points_inside(double *swig_np_dbl_in_v1,  // xpoi
                     long n_swig_np_dbl_in_v1,
                     double *swig_np_dbl_in_v2,  // ypoi
                     long n_swig_np_dbl_in_v2,
                     double *swig_np_dbl_inplace_v1,  // zpoi
                     long n_swig_np_dbl_inplace_v1,
                     double *swig_np_dbl_in_v3,  // xpol
                     long n_swig_np_dbl_in_v3,
                     double *swig_np_dbl_in_v4,  // ypol
                     long n_swig_np_dbl_in_v4,
                     double new_value,
                     int option,
                     int inside);

int
polys_chk_point_inside(double x,
                       double y,
                       double *p_xp_v,
                       double *p_yp_v,
                       int np1,
                       int np2);

int
pol_geometrics(double *swig_np_dbl_in_v1,    // *xv
               long n_swig_np_dbl_in_v1,     // nxv
               double *swig_np_dbl_in_v2,    // *yv
               long n_swig_np_dbl_in_v2,     // nyv
               double *swig_np_dbl_in_v3,    // *zv
               long n_swig_np_dbl_in_v3,     // nzv
               double *swig_np_dbl_aout_v1,  // *tv
               long n_swig_np_dbl_aout_v1,   // ntv
               double *swig_np_dbl_aout_v2,  // *dtv
               long n_swig_np_dbl_aout_v2,   // ndtv
               double *swig_np_dbl_aout_v3,  // *hv
               long n_swig_np_dbl_aout_v3,   // nhv
               double *swig_np_dbl_aout_v4,  // *dhv
               long n_swig_np_dbl_aout_v4);  // ndhv

/*
 *======================================================================================
 * CUBE (REGULAR 3D)
 *======================================================================================
 */

/* sucu_* is common for surf and cube: */

int
sucu_ij_from_xy(int *i,
                int *j,
                double *rx,
                double *ry,
                double x,
                double y,
                double xori,
                double xinc,
                double yori,
                double yinc,
                int nx,
                int ny,
                int yflip,
                double rot_azi_deg,
                int flag);

void
cube_scan_segy_hdr(char *file,
                   int *gn_bitsheader,
                   int *gn_formatcode,
                   float *gf_segyformat,
                   int *gn_samplespertrace,
                   int *gn_measuresystem,
                   int option,
                   char *outfile);

int
cube_import_storm(int nx,
                  int ny,
                  int nz,
                  char *file,
                  int lstart,
                  float *swig_np_flt_aout_v1,  // *p_cube_v
                  long n_swig_np_flt_aout_v1,  // nxyz
                  int option);

void
cube_import_segy(char *file,
                 int hbitoffset,
                 int formatcode,
                 float segyformat,
                 int samplespertrace,
                 int *nx,
                 int *ny,
                 int *nz,
                 float *p_val_v,
                 double *xori,
                 double *xinc,
                 double *yori,
                 double *yinc,
                 double *zori,
                 double *zinc,
                 double *rotation,
                 int *yflip,
                 int *zflip,
                 double *minval,
                 double *maxval,
                 int optscan,
                 int option,
                 char *outfile);

void
cube_import_rmsregular(int iline,
                       int *ndef,
                       int *ndefsum,
                       int nx,
                       int ny,
                       int nz,
                       float *val_v,
                       double *vmin,
                       double *vmax,
                       char *file,
                       int *ierr);

int
cube_export_segy(char *sfile,
                 int nx,
                 int ny,
                 int nz,
                 float *swig_np_flt_in_v1,  // cube_v
                 long n_swig_np_flt_in_v1,  // n total nx*ny*nz
                 double xori,
                 double xinc,
                 double yori,
                 double yinc,
                 double zori,
                 double zinc,
                 double rotation,
                 int yflip,
                 int zflip,
                 int *ilinesp,
                 int *xlinesp,
                 int *tracidp,
                 int option);

int
cube_export_rmsregular(int nx,
                       int ny,
                       int nz,
                       double xmin,
                       double ymin,
                       double zmin,
                       double xinc,
                       double yinc,
                       double zinc,
                       double rotation,
                       int yflip,
                       float *swig_np_flt_in_v1,  // *cubevalsv
                       long n_swig_np_flt_in_v1,  // n total nx*ny*nz
                       char *file);

int
cube_coord_val_ijk(int i,
                   int j,
                   int k,
                   int nx,
                   int ny,
                   int nz,
                   double xori,
                   double xinc,
                   double yori,
                   double yinc,
                   double zori,
                   double zinc,
                   double rot_deg,
                   int yflip,
                   float *p_val_v,
                   double *x,
                   double *y,
                   double *z,
                   float *value,
                   int option);

int
cube_xy_from_ij(int i,
                int j,
                double *swig_dbl_out_p1,  //  *x
                double *swig_dbl_out_p2,  //  *y
                double xori,
                double xinc,
                double yori,
                double yinc,
                int nx,
                int ny,
                int yflip,
                double rot_azi_deg,
                int flag);

int
cube_ijk_from_xyz(int *i,
                  int *j,
                  int *k,
                  double *rx,
                  double *ry,
                  double *rz,
                  double x,
                  double y,
                  double z,
                  double xori,
                  double xinc,
                  double yori,
                  double yinc,
                  double zori,
                  double zinc,
                  int nx,
                  int ny,
                  int nz,
                  double rot_deg,
                  int yflip,
                  int flag);

int
cube_value_ijk(int i,
               int j,
               int k,
               int nx,
               int ny,
               int nz,
               float *p_val_v,
               float *value);

int
cube_value_xyz_cell(double x,
                    double y,
                    double z,
                    double xori,
                    double xinc,
                    double yori,
                    double yinc,
                    double zori,
                    double zinc,
                    double rot_deg,
                    int yflip,
                    int nx,
                    int ny,
                    int nz,
                    float *p_val_v,
                    float *value,
                    int option);

int
cube_value_xyz_interp(double x,
                      double y,
                      double z,
                      double xori,
                      double xinc,
                      double yori,
                      double yinc,
                      double zori,
                      double zinc,
                      double rot_deg,
                      int yflip,
                      int nx,
                      int ny,
                      int nz,
                      float *p_val_v,
                      float *value,
                      int option);

int
cube_vertical_val_list(int i,
                       int j,
                       int nx,
                       int ny,
                       int nz,
                       float *p_val_v,
                       float *p_vertical_v);

int
cube_resample_cube(int ncx1,
                   int ncy1,
                   int ncz1,
                   double cxori1,
                   double cxinc1,
                   double cyori1,
                   double cyinc1,
                   double czori1,
                   double czinc1,
                   double crotation1,
                   int yflip1,
                   float *swig_np_flt_inplace_v1,  // *p_cubeval1_v,
                   long n_swig_np_flt_inplace_v1,  // ncube1,
                   int ncx2,
                   int ncy2,
                   int ncz2,
                   double cxori2,
                   double cxinc2,
                   double cyori2,
                   double cyinc2,
                   double czori2,
                   double czinc2,
                   double crotation2,
                   int yflip2,
                   float *swig_np_flt_in_v1,  // *p_cubeval2_v,
                   long n_swig_np_flt_in_v1,  // ncube2,
                   int option1,
                   int option2,
                   float ovalue);

int
cube_get_randomline(double *swig_np_dbl_in_v1,  // *xvec,
                    long n_swig_np_dbl_in_v1,   // nxvec,
                    double *swig_np_dbl_in_v2,  // *yvec,
                    long n_swig_np_dbl_in_v2,   // nyvec,
                    double zmin,
                    double zmax,
                    int nzsam,
                    double xori,
                    double xinc,
                    double yori,
                    double yinc,
                    double zori,
                    double zinc,
                    double rot_deg,
                    int yflip,
                    int nx,
                    int ny,
                    int nz,
                    float *swig_np_flt_in_v1,     // *p_val_v
                    long n_swig_np_flt_in_v1,     // ncube
                    double *swig_np_dbl_aout_v1,  // *values
                    long n_swig_np_dbl_aout_v1,   // nvalues
                    int option);
/*
 *======================================================================================
 * GRID (3D) CORNERPOINTS
 * FIXHD: codefix needed
 *======================================================================================
 */

typedef double (*metric)(const double,
                         const double,
                         const double,
                         const double,
                         const double,
                         const double);
double
euclid_length(const double x1,
              const double y1,
              const double z1,
              const double x2,
              const double y2,
              const double z2);

double
horizontal_length(const double x1,
                  const double y1,
                  const double z1,
                  const double x2,
                  const double y2,
                  const double z2);

double
east_west_vertical_length(const double x1,
                          const double y1,
                          const double z1,
                          const double x2,
                          const double y2,
                          const double z2);

double
north_south_vertical_length(const double x1,
                            const double y1,
                            const double z1,
                            const double x2,
                            const double y2,
                            const double z2);

double
x_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2);

double
y_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2);

double
z_projection(const double x1,
             const double y1,
             const double z1,
             const double x2,
             const double y2,
             const double z2);

int
grdcp3d_calc_dx(int nx,
                int ny,
                int nz,
                double *swig_np_dbl_in_v1,       // *coordsv,
                long n_swig_np_dbl_in_v1,        // ncoord,
                double *swig_np_dbl_in_v2,       // *zcornsv,
                long n_swig_np_dbl_in_v2,        // nzcorn,
                double *swig_np_dbl_inplace_v1,  // *dx,
                long n_swig_np_dbl_inplace_v1,   // ntot,
                metric m);
int
grdcp3d_calc_dy(int nx,
                int ny,
                int nz,
                double *swig_np_dbl_in_v1,       // *coordsv,
                long n_swig_np_dbl_in_v1,        // ncoord,
                double *swig_np_dbl_in_v2,       // *zcornsv,
                long n_swig_np_dbl_in_v2,        // nzcorn,
                double *swig_np_dbl_inplace_v1,  // *dy,
                long n_swig_np_dbl_inplace_v1,   // ntot,
                metric m);

int
grdcp3d_calc_dz(int nx,
                int ny,
                int nz,
                double *swig_np_dbl_in_v1,       // *coordsv,
                long n_swig_np_dbl_in_v1,        // ncoord,
                double *swig_np_dbl_in_v2,       // *zcornsv,
                long n_swig_np_dbl_in_v2,        // nzcorn,
                double *swig_np_dbl_inplace_v1,  // *dz,
                long n_swig_np_dbl_inplace_v1,   // ntot,
                metric m);

void
grdcp3d_calc_xyz(long ncol,
               long nrow,
               long nlay,

               double *swig_np_dbl_in_v1, // *coordsv,
               long n_swig_np_dbl_in_v1,  // ncoord,
               float *swig_np_flt_in_v1,  // *zcornsv,
               long n_swig_np_flt_in_v1,  // nzcorn,
               int *swig_np_int_in_v1,    // *actnumsv,
               long n_swig_np_int_in_v1,  // nactnum,

               int option,

               double *swig_np_dbl_aout_v1,  // xarr
               long n_swig_np_dbl_aout_v1,
               double *swig_np_dbl_aout_v2,  // yarr
               long n_swig_np_dbl_aout_v2,
               double *swig_np_dbl_aout_v3,  // zarr
               long n_swig_np_dbl_aout_v3);

long
grd3d_scan_roffbinary(FILE *fc,
                      int *swig_int_out_p1,       // *swap,
                      char *swig_out_char_msize,  // tagletters
                      int nswig_out_char_msize,
                      int *rectypes,
                      long *reclengths,
                      long *recstarts,
                      long maxkw);

void
grd3d_conv_roxapi_grid(int nx,
                       int ny,
                       int nz,
                       long nxyz,

                       int *swig_np_int_in_v1,     // *cact
                       long n_swig_np_int_in_v1,   // ncact
                       double *swig_np_dbl_in_v1,  // *crds
                       long n_swig_np_dbl_in_v1,   // ncrds

                       double *swig_np_dbl_inplace_v1,  // *coordsv
                       long n_swig_np_dbl_inplace_v1,   // ncoord
                       double *swig_np_dbl_inplace_v2,  // *zcornsv
                       long n_swig_np_dbl_inplace_v2,   // nzcorn
                       int *swig_np_int_inplace_v1,     // *actnumsv
                       long n_swig_np_int_inplace_v1    // nactnum
);

long
grd3d_scan_eclbinary(FILE *fc,
                     char *swig_out_char_msize,  // keywords (as one long string)
                     int nswig_out_char_msize,
                     int *swig_np_int_aout_v1,  // rectype
                     long n_swig_np_int_aout_v1,
                     long *swig_np_lng_aout_v1,  // reclengths
                     long n_swig_np_lng_aout_v1,
                     long *swig_np_lng_aout_v2,  // recstarts
                     long n_swig_np_lng_aout_v2);

int
grd3d_read_eclrecord(FILE *fc,
                     long recstart,
                     int rectype,
                     int *swig_np_int_inplace_v1,
                     long n_swig_np_int_inplace_v1,
                     float *swig_np_flt_inplace_v1,
                     long n_swig_np_flt_inplace_v1,
                     double *swig_np_dbl_inplace_v1,
                     long n_swig_np_dbl_inplace_v1);

int
grd3d_roff2xtgeo_splitenz(int nz,
                          float zoffset,
                          float zscale,
                          char *swig_bytes,               // *splitenz
                          long swig_bytes_len,            // nsplitenz
                          float *swig_np_flt_inplace_v1,  // *zdata
                          long n_swig_np_flt_inplace_v1,  // nzdata
                          float *swig_np_flt_inplace_v2,  // *zcornsv
                          long n_swig_np_flt_inplace_v2   // nzcorn
);

void
grd3d_zcorn_convert(int nx, int ny, int nz, float *zcorn, double *zcornsv, int option);

int
grd3d_ecl_tsteps(FILE *fc, int *seqnums, int *day, int *mon, int *year, int nmax);

int
grd3d_conv_grid_roxapi(int ncol,
                       int nrow,
                       int nlay,

                       double *swig_np_dbl_in_v1,  // *coordsv
                       long n_swig_np_dbl_in_v1,   // ncoord
                       double *swig_np_dbl_in_v2,  // *zcornsv
                       long n_swig_np_dbl_in_v2,   // nzcorn
                       int *swig_np_int_in_v1,     // *actnumsv
                       long n_swig_np_int_in_v1,   // nact

                       double *swig_np_dbl_aout_v1,  // *tpillars
                       long n_swig_np_dbl_aout_v1,   // ntpillars
                       double *swig_np_dbl_aout_v2,  // *bpillars
                       long n_swig_np_dbl_aout_v2,   // nbpillars
                       double *swig_np_dbl_aout_v3,  // *zcorners
                       long n_swig_np_dbl_aout_v3    // nzcorners
);

int
grd3d_copy(int ncol,
           int nrow,
           int nlay,
           double *p_coord1_v,
           double *p_zcorn1_v,
           int *p_actnum1_v,
           double *p_coord2_v,
           double *p_zcorn2_v,
           int *p_actnum2_v,
           int iflag);

int
grd3d_crop_geometry(int nx,
                    int ny,
                    int nz,

                    double *swig_np_dbl_in_v1,  // *p_coord1_v
                    long n_swig_np_dbl_in_v1,   // ncoord1
                    double *swig_np_dbl_in_v2,  // *p_zcorn1_v
                    long n_swig_np_dbl_in_v2,   // nzcorn1
                    int *swig_np_int_in_v1,     // *p_actnum1_v
                    long n_swig_np_int_in_v1,   // nact1

                    double *swig_np_dbl_inplace_v1,  // *p_coord2_v
                    long n_swig_np_dbl_inplace_v1,   // ncoord2
                    double *swig_np_dbl_inplace_v2,  // *p_zcorn2_v
                    long n_swig_np_dbl_inplace_v2,   // nzcorn2
                    int *swig_np_int_inplace_v1,     // *p_actnum2_v
                    long n_swig_np_int_inplace_v1,   // nact2

                    int ic1,
                    int ic2,
                    int jc1,
                    int jc2,
                    int kc1,
                    int kc2,
                    int *nactive,
                    int iflag);

int
grd3d_reduce_onelayer(int nx,
                      int ny,
                      int nz,
                      double *swig_np_dbl_in_v1,       // *p_zcorn1_v
                      long n_swig_np_dbl_in_v1,        // nzcornin1
                      double *swig_np_dbl_inplace_v1,  // *p_zcorn2_v
                      long n_swig_np_dbl_inplace_v1,   // nzcornin2
                      int *swig_np_int_in_v1,          // *p_actnum1_v
                      long n_swig_np_int_in_v1,        // nact1
                      int *swig_np_int_inplace_v1,     // *p_actnum2_v
                      long n_swig_np_int_inplace_v1,   // nact2
                      int *nactive,
                      int iflag);

int
grd3d_get_lay_slice(int nx,
                    int ny,
                    int nz,

                    double *swig_np_dbl_in_v1,  // *coordsv
                    long n_swig_np_dbl_in_v1,   // ncoord
                    double *swig_np_dbl_in_v2,  // *zcornsv
                    long n_swig_np_dbl_in_v2,   // nzcorn
                    int *swig_np_int_in_v1,     // *actnumsv
                    long n_swig_np_int_in_v1,   // nact

                    int kslice,
                    int koption,
                    int actonly,

                    double *swig_np_dbl_aout_v1,  // *slicev
                    long n_swig_np_dbl_aout_v1,   // nslicev,
                    long *swig_np_lng_aout_v1,    // *ibv,
                    long n_swig_np_lng_aout_v1    // nibv
);

int
grd3d_refine_vert(int nx,
                  int ny,
                  int nz,

                  double *swig_np_dbl_in_v2,  // *zcornsv
                  long n_swig_np_dbl_in_v2,   // nzcorn
                  int *swig_np_int_in_v1,     // *actnumsv
                  long n_swig_np_int_in_v1,   // nact

                  int nzref,

                  double *swig_np_dbl_inplace_v2,  // *p_zcornref_v
                  long n_swig_np_dbl_inplace_v2,   // nzcornref
                  int *swig_np_int_inplace_v1,     // *p_actnumref_v
                  long n_swig_np_int_inplace_v1,   // nactref

                  int *rfac);

void
grd3d_convert_hybrid(int nx,
                     int ny,
                     int nz,

                     double *swig_np_dbl_in_v1,  // *coordsv
                     long n_swig_np_dbl_in_v1,   // ncoord
                     double *swig_np_dbl_in_v2,  // *zcornsv
                     long n_swig_np_dbl_in_v2,   // nzcorn
                     int *swig_np_int_in_v1,     // *actnumsv
                     long n_swig_np_int_in_v1,   // nact

                     int nzhyb,

                     double *swig_np_dbl_inplace_v1,  // *p_zcornhyb_v
                     long n_swig_np_dbl_inplace_v1,   // nzcornhybin
                     int *swig_np_int_inplace_v1,     // *p_actnumhyb_v
                     long n_swig_np_int_inplace_v1,   // nacthybin

                     double toplevel,
                     double botlevel,
                     int ndiv,

                     int *swig_np_int_in_v2,    // *p_region_v
                     long n_swig_np_int_in_v2,  // nreg

                     int region);

void
grd3d_make_z_consistent(int nx,
                        int ny,
                        int nz,
                        double *swig_np_dbl_inplace_v1,  // *zcornsv
                        long n_swig_np_dbl_inplace_v1,   // nzcorn
                        double zsep);

int
grd3d_translate(int nx,
                int ny,
                int nz,
                int xflip,
                int yflip,
                int zflip,
                double xshift,
                double yshift,
                double zshift,
                double *swig_np_dbl_inplace_v1,  // *coordsv
                long n_swig_np_dbl_inplace_v1,   // ncoord
                double *swig_np_dbl_inplace_v2,  // *zcornsv
                long n_swig_np_dbl_inplace_v2);  // nzcorn

int
grd3d_reverse_jrows(int nx,
                    int ny,
                    int nz,
                    double *swig_np_dbl_inplace_v1,  // *coordsv
                    long n_swig_np_dbl_inplace_v1,   // ncoord
                    double *swig_np_dbl_inplace_v2,  // *zcornsv
                    long n_swig_np_dbl_inplace_v2,   // nzcorn
                    int *swig_np_int_inplace_v1,     // *actnumsv
                    long n_swig_np_int_inplace_v1);  // nact

int
grd3d_point_val_crange(double x,
                       double y,
                       double z,
                       int nx,
                       int ny,
                       int nz,
                       double *p_coor_v,
                       double *zcornsv,
                       int *actnumsv,
                       double *p_val_v,
                       double *value,
                       int imin,
                       int imax,
                       int jmin,
                       int jmax,
                       int kmin,
                       int kmax,
                       long *ibs,
                       int option);

long
grd3d_point_in_cell(long ibstart,
                    int kzonly,
                    double x,
                    double y,
                    double z,
                    int nx,
                    int ny,
                    int nz,
                    double *p_coor_v,
                    double *zcornsv,
                    int *actnumsv,
                    int maxrad,
                    int sflag,
                    int *nradsearch,
                    int option);

int
grd3d_points_ijk_cells(double *swig_np_dbl_in_v1,  // *xvec
                       long n_swig_np_dbl_in_v1,   // nxvec
                       double *swig_np_dbl_in_v2,  // *yvec
                       long n_swig_np_dbl_in_v2,   // nyvec
                       double *swig_np_dbl_in_v3,  // *zvec
                       long n_swig_np_dbl_in_v3,   // nzvec

                       int mcol,
                       int mrow,
                       double xori,
                       double yori,
                       double xinc,
                       double yinc,
                       double rotation,
                       int yflip,
                       double *maptopi,
                       double *maptopj,
                       double *mapbasi,
                       double *mapbasj,

                       int nx,
                       int ny,
                       int nz,
                       double *swig_np_dbl_in_v4,  // *coordsv
                       long n_swig_np_dbl_in_v4,   // ncoordin
                       double *swig_np_dbl_in_v5,  // *p_zcoord_v
                       long n_swig_np_dbl_in_v5,   // nzcornin
                       int *swig_np_int_in_v1,     // *actnumsv
                       long n_swig_np_int_in_v1,   // nactin

                       double *swig_np_dbl_in_v6,  // *p_zcoordone_v
                       long n_swig_np_dbl_in_v6,   // nzcornonein

                       int actnumoption,

                       int *swig_np_int_aout_v1,     // *ivec
                       long n_swig_np_int_aout_v1,   // nivec
                       int *swig_np_int_aout_v2,     // *jvec
                       long n_swig_np_int_aout_v2,   // njvec
                       int *swig_np_int_aout_v3,     // *kvec
                       long n_swig_np_int_aout_v3);  // nkvec

int
grd3d_get_randomline(double *swig_np_dbl_in_v1,  // *xvec,
                     long n_swig_np_dbl_in_v1,   // nxvec,
                     double *swig_np_dbl_in_v2,  // *yvec,
                     long n_swig_np_dbl_in_v2,   // nyvec,
                     double zmin,
                     double zmax,
                     int nzsam,

                     int mcol,
                     int mrow,
                     double xori,
                     double yori,
                     double xinc,
                     double yinc,
                     double rotation,
                     int yflip,
                     double *maptopi,
                     double *maptopj,
                     double *mapbasi,
                     double *mapbasj,

                     int nx,
                     int ny,
                     int nz,

                     double *swig_np_dbl_in_v3,  // *coordsv
                     long n_swig_np_dbl_in_v3,   // ncoordin
                     double *swig_np_dbl_in_v4,  // *p_zcoord_v
                     long n_swig_np_dbl_in_v4,   // nzcornin
                     int *swig_np_int_in_v1,     // *actnumsv
                     long n_swig_np_int_in_v1,   // nactin

                     double *p_val_v,

                     double *swig_np_dbl_in_v5,  // *p_zcoordone_v
                     long n_swig_np_dbl_in_v5,   // nzcornonein
                     int *swig_np_int_in_v2,     // *p_actnumone_v
                     long n_swig_np_int_in_v2,   // nactonein

                     double *swig_np_dbl_aout_v1,  // *values
                     long n_swig_np_dbl_aout_v1);  // nvalues

void
grd3d_collapse_inact(int nx,
                     int ny,
                     int nz,
                     double *swig_np_dbl_inplace_v1,  // *zcornsv
                     long n_swig_np_dbl_inplace_v1,   // nzcorn
                     int *swig_np_int_inplace_v1,     // *actnumsv
                     long n_swig_np_int_inplace_v1    // nactnum
);

void
grdcp3d_midpoint(long i,
               long j,
               long k,
               long ncol,
               long nrow,
               long nlay,
               double *coordsv,
               long ncoord,
               float *zcornsv,
               long nzcorn,
               double *x,
               double *y,
               double *z);

void
grd3d_midpoint(int i,
               int j,
               int k,
               int nx,
               int ny,
               int nz,
               double *coordsv,
               long ncoord,
               double *zcornsv,
               long nzcorn,
               double *x,
               double *y,
               double *z);

int
grd3d_inact_outside_pol(double *swig_np_dbl_in_v1,  // polygons X
                        long n_swig_np_dbl_in_v1,   // N
                        double *swig_np_dbl_in_v2,  // polygons Y
                        long n_swig_np_dbl_in_v2,   // N
                        int nx,
                        int ny,
                        int nz,
                        double *swig_np_dbl_in_v3,      // *coordsv
                        long n_swig_np_dbl_in_v3,       // ncoordin
                        double *swig_np_dbl_in_v4,      // *zcornsv
                        long n_swig_np_dbl_in_v4,       // nzcornin
                        int *swig_np_int_inplace_v1,    // *actnumsv
                        long n_swig_np_int_inplace_v1,  // nact
                        int k1,
                        int k2,
                        int force_close,
                        int option);

int
grd3d_setval_poly(double *swig_np_dbl_in_v1,  // polygons X
                  long n_swig_np_dbl_in_v1,   // N
                  double *swig_np_dbl_in_v2,  // polygons Y
                  long n_swig_np_dbl_in_v2,   // N

                  int nx,
                  int ny,
                  int nz,

                  double *swig_np_dbl_in_v3,  // *coordsv
                  long n_swig_np_dbl_in_v3,   // ncoordin
                  double *swig_np_dbl_in_v4,  // *zcornsv
                  long n_swig_np_dbl_in_v4,   // nzcornin
                  int *swig_np_int_in_v1,     // *actnumsv
                  long n_swig_np_int_in_v1,   // nactin

                  double *p_val_v,
                  double value);

int
grd3d_geometrics(int nx,
                 int ny,
                 int nz,

                 double *swig_np_dbl_in_v1,  // *coordsv
                 long n_swig_np_dbl_in_v1,   // ncoord
                 double *swig_np_dbl_in_v2,  // *zcornsv
                 long n_swig_np_dbl_in_v2,   // nzcorn
                 int *swig_np_int_in_v1,     // *actnumsv
                 long n_swig_np_int_in_v1,   // nact

                 double *xori,
                 double *yori,
                 double *zori,
                 double *xmin,
                 double *xmax,
                 double *ymin,
                 double *ymax,
                 double *zmin,
                 double *zmax,
                 double *rotation,
                 double *dx,
                 double *dy,
                 double *dz,
                 int option1,
                 int option2);

int
grd3d_check_cell_splits(int ncol,
                        int nrow,
                        int nlay,
                        double *coordsv,
                        double *zcornsv,
                        long ib1,
                        long ib2);

int
grd3d_adj_cells(int ncol,
                int nrow,
                int nlay,
                double *swig_np_dbl_in_v1,      // *coordsv
                long n_swig_np_dbl_in_v1,       // ncoordin
                double *swig_np_dbl_in_v2,      // *zcornsv
                long n_swig_np_dbl_in_v2,       // nzcornin
                int *swig_np_int_inplace_v1,    // *actnumsv
                long n_swig_np_int_inplace_v1,  // nact
                int *p_prop1,
                long nprop1,
                int val1,
                int val2,
                int *p_prop2,
                long nprop2,
                int iflag1,
                int iflag2);

void
grd3d_corners(int i,
              int j,
              int k,
              int nx,
              int ny,
              int nz,

              double *swig_np_dbl_in_v1,  // *coordsv
              long n_swig_np_dbl_in_v1,   // ncoordin
              double *swig_np_dbl_in_v2,  // *p_zcoord_v
              long n_swig_np_dbl_in_v2,   // nzcornin

              double corners[]);

double
grd3d_zminmax(int i, int j, int k, int nx, int ny, int nz, double *zcornsv, int option);

void
grd3d_get_all_corners(int nx,
                      int ny,
                      int nz,

                      double *swig_np_dbl_in_v1,  // *coordsv
                      long n_swig_np_dbl_in_v1,   // ncoordin
                      double *swig_np_dbl_in_v2,  // *p_zcoord_v
                      long n_swig_np_dbl_in_v2,   // nzcornin
                      int *swig_np_int_in_v1,     // *actnumsv
                      long n_swig_np_int_in_v1,   // nactin

                      double *x1,
                      double *y1,
                      double *z1,
                      double *x2,
                      double *y2,
                      double *z2,
                      double *x3,
                      double *y3,
                      double *z3,
                      double *x4,
                      double *y4,
                      double *z4,
                      double *x5,
                      double *y5,
                      double *z5,
                      double *x6,
                      double *y6,
                      double *z6,
                      double *x7,
                      double *y7,
                      double *z7,
                      double *x8,
                      double *y8,
                      double *z8,
                      int option);

int
grd3d_well_ijk(int nx,
               int ny,
               int nz,

               double *swig_np_dbl_in_v1,  // *coordsv
               long n_swig_np_dbl_in_v1,   // ncoordin
               double *swig_np_dbl_in_v2,  // *p_zcoord_v
               long n_swig_np_dbl_in_v2,   // nzcornin
               int *swig_np_int_in_v1,     // *actnumsv
               long n_swig_np_int_in_v1,   // nactin

               double *swig_np_dbl_in_v3,  // *p_zcoord_onelay_v
               long n_swig_np_dbl_in_v3,   // nzcornonein
               int *swig_np_int_in_v2,     // *p_actnum_onelay_v
               long n_swig_np_int_in_v2,   // nactonein

               int nval,
               double *p_utme_v,
               double *p_utmn_v,
               double *p_tvds_v,
               int *ivector,
               int *jvector,
               int *kvector,
               int iflag);

/*
 *======================================================================================
 * New format spec for corner point 3D grid, grdcp3d_*. Cf xtgformat = 2 in grid class
 * Also, all dimenstions nrow, ncol etc shall be long
 *======================================================================================
 */

void
grdcp3d_from_cube(int ncol,
                  int nrow,
                  int nlay,
                  double *swig_np_dbl_inplaceflat_v1,  // *coordsv,
                  long n_swig_np_dbl_inplaceflat_v1,   // ncoord,
                  float *swig_np_flt_inplaceflat_v1,   // *zcornsv,
                  long n_swig_np_flt_inplaceflat_v1,   // nzcorn,
                  int *swig_np_int_inplaceflat_v1,     // *actnumsv,
                  long n_swig_np_int_inplaceflat_v1,   // nactnum,
                  double xori,
                  double yori,
                  double zori,
                  double xinc,
                  double yinc,
                  double zinc,
                  double rotation,
                  int yflip,
                  int option);

void
grdcp3d_export_xtgeo_grid(long ncol,
                          long nrow,
                          long nlay,
                          double *swig_np_dbl_inplaceflat_v1,  // coordsv
                          long n_swig_np_dbl_inplaceflat_v1,   // ncoord
                          float *swig_np_flt_inplaceflat_v1,   // zcornsv (float)
                          long n_swig_np_flt_inplaceflat_v1,   // nzcorn
                          int *swig_np_int_inplaceflat_v2,     // actnumsv
                          long n_swig_np_int_inplaceflat_v2,   // nact
                          char *metadata,
                          FILE *fc);

int
grdcp3d_import_xtgeo_grid(int mode,
                          long *swig_lon_out_p1,
                          long *swig_lon_out_p2,
                          long *swig_lon_out_p3,
                          double *swig_np_dbl_inplaceflat_v1,  // coordsv
                          long n_swig_np_dbl_inplaceflat_v1,   // ncoord
                          float *swig_np_flt_inplaceflat_v1,   // zcornsv (float)
                          long n_swig_np_flt_inplaceflat_v1,   // nzcorn
                          int *swig_np_int_inplaceflat_v2,     // actnumsv
                          long n_swig_np_int_inplaceflat_v2,   // nact
                          char *swig_bnd_char_100k,            // metadata
                          FILE *fc);

int
grd3cp3d_xtgformat1to2_geom(long ncol,
                            long nrow,
                            long nlay,
                            double *swig_np_dbl_inplaceflat_v1,  // coordsv1
                            long n_swig_np_dbl_inplaceflat_v1,
                            double *swig_np_dbl_inplaceflat_v2,  // coordsv2
                            long n_swig_np_dbl_inplaceflat_v2,
                            double *swig_np_dbl_inplaceflat_v3,  // zcornsv1
                            long n_swig_np_dbl_inplaceflat_v3,
                            float *swig_np_flt_inplaceflat_v1,  // zcornsv2 (float)
                            long n_swig_np_flt_inplaceflat_v1,
                            int *swig_np_int_inplaceflat_v1,  // actnumsv1
                            long n_swig_np_int_inplaceflat_v1,
                            int *swig_np_int_inplaceflat_v2,  // actnumsv2
                            long n_swig_np_int_inplaceflat_v2);

int
grd3cp3d_xtgformat2to1_geom(long ncol,
                            long nrow,
                            long nlay,
                            double *swig_np_dbl_inplaceflat_v1,  // coordsv1
                            long n_swig_np_dbl_inplaceflat_v1,
                            double *swig_np_dbl_inplaceflat_v2,  // coordsv2
                            long n_swig_np_dbl_inplaceflat_v2,
                            double *swig_np_dbl_inplaceflat_v3,  // zcornsv1
                            long n_swig_np_dbl_inplaceflat_v3,
                            float *swig_np_flt_inplaceflat_v1,  // zcornsv2 (float)
                            long n_swig_np_flt_inplaceflat_v1,
                            int *swig_np_int_inplaceflat_v1,  // actnumsv1
                            long n_swig_np_int_inplaceflat_v1,
                            int *swig_np_int_inplaceflat_v2,  // actnumsv2
                            long n_swig_np_int_inplaceflat_v2);

void
grdcp3d_process_edges(long ncol,
                      long nrow,
                      long nlay,
                      float *swig_np_flt_inplaceflat_v1,
                      long n_swig_np_flt_inplaceflat_v1);

void
grdcp3d_corners(long ic,
                long jc,
                long kc,
                long ncol,
                long nrow,
                long nlay,
                double *swig_np_dbl_inplaceflat_v1,  // coordsv
                long n_swig_np_dbl_inplaceflat_v1,
                float *swig_np_flt_inplaceflat_v1,  // zcornsv
                long n_swig_np_flt_inplaceflat_v1,
                double corners[]);

long
grdcp3d_get_vtk_esg_geometry_data(long ncol,
                                  long nrow,
                                  long nlay,

                                  double *swig_np_dbl_inplaceflat_v1,  // coordsv
                                  long n_swig_np_dbl_inplaceflat_v1,
                                  float *swig_np_flt_inplaceflat_v1,   // zcornsv
                                  long n_swig_np_flt_inplaceflat_v1,

                                  double *swig_np_dbl_aout_v1,         // output vertex arr
                                  long n_swig_np_dbl_aout_v1,          // allocated length of vertex arr
                                  long *swig_np_lng_aout_v1,           // hex connectivity array (out)
                                  long n_swig_np_lng_aout_v1);         // allocated length of hex conn arr

void
grdcp3d_get_vtk_grid_arrays(long ncol,
                            long nrow,
                            long nlay,

                            double *swig_np_dbl_inplaceflat_v1,  // coordsv
                            long n_swig_np_dbl_inplaceflat_v1,
                            float *swig_np_flt_inplaceflat_v1,  // zcornsv
                            long n_swig_np_flt_inplaceflat_v1,

                            double *swig_np_dbl_aout_v1,  // xarr
                            long n_swig_np_dbl_aout_v1,
                            double *swig_np_dbl_aout_v2,  // yarr
                            long n_swig_np_dbl_aout_v2,
                            double *swig_np_dbl_aout_v3,  // zarr
                            long n_swig_np_dbl_aout_v3);

int
grdcp3d_conv_grid_roxapi(long ncol,
                         long nrow,
                         long nlay,

                         double *swig_np_dbl_inplaceflat_v1,  // coordsv
                         long n_swig_np_dbl_inplaceflat_v1,   // ncoord
                         float *swig_np_flt_inplaceflat_v1,   // zcornsv
                         long n_swig_np_flt_inplaceflat_v1,   // nzcorn

                         double *swig_np_dbl_aout_v1,  // tpillars
                         long n_swig_np_dbl_aout_v1,   // ntpillars
                         double *swig_np_dbl_aout_v2,  // bpillars
                         long n_swig_np_dbl_aout_v2,   // nbpillars
                         double *swig_np_dbl_aout_v3,  // zcornsv
                         long n_swig_np_dbl_aout_v3);  // nzcorners

void
grdcp3d_quality_indicators(long ncol,
                           long nrow,
                           long nlay,
                           double *swig_np_dbl_inplaceflat_v1,  // coordsv,
                           long n_swig_np_dbl_inplaceflat_v1,   // ncoordin,
                           float *swig_np_flt_inplaceflat_v1,   // zcornsv,
                           long n_swig_np_flt_inplaceflat_v1,   // nzcorn,
                           int *swig_np_int_inplaceflat_v1,     // actnumsv,
                           long n_swig_np_int_inplaceflat_v1,   // nactnum
                           float *swig_np_flt_inplaceflat_v2,   // fresults
                           long n_swig_np_flt_inplaceflat_v2);  // nactnum
void
grdcp3d_cellvol(long ncol,
                long nrow,
                long nlay,
                double *swig_np_dbl_inplaceflat_v1,  // coordsv,
                long n_swig_np_dbl_inplaceflat_v1,   // ncoordin,
                float *swig_np_flt_inplaceflat_v1,   // zcornsv,
                long n_swig_np_flt_inplaceflat_v1,   // nzcorn,
                int *swig_np_int_inplaceflat_v1,     // actnumsv,
                long n_swig_np_int_inplaceflat_v1,   // nactnum
                double *swig_np_dbl_inplaceflat_v2,  // cellvolsv
                long n_swig_np_dbl_inplaceflat_v2,   // ncell
                int precision,
                int option);

/*
 *======================================================================================
 * WELL spesific
 *======================================================================================
 */

int
well_geometrics(int np,
                double *xv,
                double *yv,
                double *zv,
                double *md,
                double *incl,
                double *az,
                int option);

int
well_trunc_parallel(double *swig_np_dbl_inplace_v1,  // *xv1,
                    long n_swig_np_dbl_inplace_v1,   // nx1
                    double *swig_np_dbl_inplace_v2,  // *yv1,
                    long n_swig_np_dbl_inplace_v2,   // ny1
                    double *swig_np_dbl_inplace_v3,  // *zv1,
                    long n_swig_np_dbl_inplace_v3,   // nz1
                    double *swig_np_dbl_in_v1,       // *xv2,
                    long n_swig_np_dbl_in_v1,        // nx2
                    double *swig_np_dbl_in_v2,       // *yv2,
                    long n_swig_np_dbl_in_v2,        // ny2
                    double *swig_np_dbl_in_v3,       // *yv2,
                    long n_swig_np_dbl_in_v3,        // ny2
                    double xtol,
                    double ytol,
                    double ztol,
                    double itol,
                    double atol,
                    int option);

int
well_surf_picks(double *swig_np_dbl_in_v1,  //*xv
                long n_swig_np_dbl_in_v1,   // nxv
                double *swig_np_dbl_in_v2,  //*yv
                long n_swig_np_dbl_in_v2,   // nyv
                double *swig_np_dbl_in_v3,  //*zv
                long n_swig_np_dbl_in_v3,   // nzv
                double *swig_np_dbl_in_v4,  //*mdv
                long n_swig_np_dbl_in_v4,   // nmdv

                int ncol,
                int nrow,
                double xori,
                double yori,
                double xinc,
                double yinc,
                int yflip,
                double rot,
                double *swig_np_dbl_in_v5,  // *surfv
                long n_swig_np_dbl_in_v5,   // nsurf,

                double *swig_np_dbl_aout_v1,  //*xoutv,
                long n_swig_np_dbl_aout_v1,
                double *swig_np_dbl_aout_v2,  //*youtv,
                long n_swig_np_dbl_aout_v2,
                double *swig_np_dbl_aout_v3,  //*zoutv,
                long n_swig_np_dbl_aout_v3,
                double *swig_np_dbl_aout_v4,  //*mdoutv,
                long n_swig_np_dbl_aout_v4,
                int *swig_np_int_aout_v1,  //*zoutv,
                long n_swig_np_int_aout_v1);

int
well_mask_shoulder(double *swig_np_dbl_inplaceflat_v1,  // lvec
                   long n_swig_np_dbl_inplaceflat_v1,   // nvec
                   int *swig_np_int_inplaceflat_v1,     // inlog
                   long n_swig_np_int_inplaceflat_v1,   // ninlog
                   int *swig_np_int_inplaceflat_v2,     // mask
                   long n_swig_np_int_inplaceflat_v2,   // nmask
                   double distance);

void
throw_exception(char *msg);

void
clear_exception();

char *
check_exception();
