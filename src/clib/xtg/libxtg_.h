#pragma once
#include "xtg.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FORTRANRECLEN 4000 /* Max record length of Fortran files */

void
x_fgets(char *, int, FILE *);

void
x_fread(void *, size_t, size_t, FILE *);

int
x_nint(double);

void
x_free(int num, ...);

double **
x_allocate_2d_double(int n1, int n2);

void
x_free_2d_double(double **ptr_array);

float **
x_allocate_2d_float(int n1, int n2);

void
x_free_2d_float(float **ptr_array);

int **
x_allocate_2d_int(int n1, int n2);

void
x_free_2d_int(int **ptr_array);

mbool **
x_allocate_2d_mbool(int n1, int n2);

void
x_free_2d_mbool(mbool **ptr_array);

int
x_cmp_sort(const void *vp, const void *vq);

void
x_mapaxes(int mode,
          double *x,
          double *y,
          const double x1,
          const double y1,
          const double x2,
          const double y2,
          const double x3,
          const double y3,
          int option);

int
x_verify_vectorlengths(int nx,
                       int ny,
                       int nz,
                       long ncoord,
                       long nzcorn,
                       long *ntot,
                       int ntlen);

void
x_basicstats(int n, double undef, double *v, double *min, double *max, double *avg);

void
x_basicstats2(int n, float undef, float *v, float *min, float *max, float *avg);

int
x_chk_point_in_cell(double x, double y, double z, double coor[], int imethod);

int
x_chk_point_in_hexahedron(double x, double y, double z, double *coor, int flip);

void
x_2d_rect_corners(double x,
                  double y,
                  double xinc,
                  double yinc,
                  double rot,
                  double result[8]);

int
x_kvpt3s(double pp[], double tri[][3], int *ier);

void
x_kmgmps(double a[][3], int l[], double prmn, int m, int n, double eps, int *ier);

void
x_kmsubs(double x[], double a[][3], int m, int n, double b[], int l[], int *ier);

int
x_vector_linint(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double dlen,
                double *xn,
                double *yn,
                double *zn);

double
x_vector_linint1d(double dval, double *dist, double *vals, int nval, int option);

double
x_vector_linint3(double x0, double x1, double x2, double y0, double y2);

int
x_linint3d(double *p0, double *p1, double zp, double *xp, double *yp);

double
x_vector_len3d(double x1, double x2, double y1, double y2, double z1, double z2);

double
x_vector_len3dx(double x1, double y1, double z1, double x2, double y2, double z2);

int
x_interp_cube_nodes(double *x_v,
                    double *y_v,
                    double *z_v,
                    float *p_v,
                    double x,
                    double y,
                    double z,
                    float *value,
                    int method);

int
x_plane_normalvector(double *points_v, double *nvector, int option);
int
x_isect_line_plane(double *nvector, double *line_v, double *point_v, int option);

double
x_angle_vectors(double *avec, double *bvec);

double
x_sample_z_from_xy_cell(double *cell_v, double x, double y, int option, int option2);

int
x_point_line_dist(double x1,
                  double y1,
                  double z1,
                  double x2,
                  double y2,
                  double z2,
                  double x3,
                  double y3,
                  double z3,
                  double *distance,
                  int option1,
                  int option2);

int
x_point_line_pos(double x1,
                 double y1,
                 double z1,
                 double x2,
                 double y2,
                 double z2,
                 double x3,
                 double y3,
                 double z3,
                 double *x,
                 double *y,
                 double *z,
                 double *rel,
                 int option1);

FILE *
x_fopen(const char *filename, const char *mode);

/*
 *-----------------------------------------------------------------------------
 * Byte swapping test
 *-----------------------------------------------------------------------------
 */

int
x_swap_check();

int
x_byteorder(int);

/******************************************************************************
  FUNCTION: SwapEndian
  PURPOSE: Swap the byte order of a structure
  EXAMPLE: float F=123.456;; SWAP_FLOAT(F);
******************************************************************************/

#define SWAP_INT(Var) Var = *(int *)SwapEndian((void *)&Var, sizeof(int))
#define SWAP_SHORT(Var) Var = *(short *)SwapEndian((void *)&Var, sizeof(short))
#define SWAP_USHORT(Var)                                                               \
    Var = *(unsigned short *)SwapEndian((void *)&Var, sizeof(short))
#define SWAP_LONG(Var) Var = *(long *)SwapEndian((void *)&Var, sizeof(long))
#define SWAP_ULONG(Var) Var = *(unsigned long *)SwapEndian((void *)&Var, sizeof(long))
#define SWAP_FLOAT(Var) Var = *(float *)SwapEndian((void *)&Var, sizeof(float))
#define SWAP_DOUBLE(Var) Var = *(double *)SwapEndian((void *)&Var, sizeof(double))

extern void *
SwapEndian(void *Addr, const int Nb);

/*
 *--------------------------------------------------------------------------------------
 * No-public grd3d routines for other issues
 *--------------------------------------------------------------------------------------
 */

int
u_read_segy_bitem(int nc,
                  int ic,
                  void *ptr,
                  size_t size,
                  size_t nmemb,
                  FILE *fc,
                  FILE *fout,
                  int swap,
                  char *txt,
                  int *nb,
                  int option);

void
u_ibm_to_float(int *from, int *to, int n, int endian, int swap);
