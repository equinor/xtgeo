#ifndef COMMON_H_
#define COMMON_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define FORTRANRECLEN 4000 /* Max record length of Fortran files */
#define XTGFORMAT1 1
#define XTGFORMAT2 2

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

#ifdef __cplusplus
extern "C"
{
#endif  // __cplusplus

    void x_fread(void *, size_t, size_t, FILE *);

    double **x_allocate_2d_double(int n1, int n2);

    void x_free_2d_double(double **ptr_array);

    bool **x_allocate_2d_bool(int n1, int n2);

    void x_free_2d_bool(bool **ptr_array);

    int x_verify_vectorlengths(long ncol,
                               long nrow,
                               long nlay,
                               long ncoord,
                               long nzcorn,
                               long *ntot,
                               int ntotlen,
                               int format);

    int x_chk_point_in_cell(double x, double y, double z, double coor[], int imethod);

    void x_2d_rect_corners(double x,
                           double y,
                           double xinc,
                           double yinc,
                           double rot,
                           double result[8]);

    int x_kvpt3s(double pp[], double tri[][3], int *ier);

    void
    x_kmgmps(double a[][3], int l[], double prmn, int m, int n, double eps, int *ier);

    void
    x_kmsubs(double x[], double a[][3], int m, int n, double b[], int l[], int *ier);

    int x_vector_linint(double x1,
                        double y1,
                        double z1,
                        double x2,
                        double y2,
                        double z2,
                        double dlen,
                        double *xn,
                        double *yn,
                        double *zn);

    double x_vector_linint1d(double dval,
                             double *dist,
                             double *vals,
                             int nval,
                             int option);

    double x_vector_linint3(double x0, double x1, double x2, double y0, double y2);

    int x_linint3d(double *p0, double *p1, double zp, double *xp, double *yp);

    double x_vector_len3d(double x1,
                          double x2,
                          double y1,
                          double y2,
                          double z1,
                          double z2);

    double x_vector_len3dx(double x1,
                           double y1,
                           double z1,
                           double x2,
                           double y2,
                           double z2);

    int x_interp_cube_nodes(double *x_v,
                            double *y_v,
                            double *z_v,
                            float *p_v,
                            double x,
                            double y,
                            double z,
                            float *value,
                            int method);

    int x_plane_normalvector(double *points_v, double *nvector, int option);

    int x_isect_line_plane(double *nvector,
                           double *line_v,
                           double *point_v,
                           int option);

    double x_angle_vectors(double *avec, double *bvec);

    double x_sample_z_from_xy_cell(double *cell_v,
                                   double x,
                                   double y,
                                   int option,
                                   int option2);

    int x_point_line_dist(double x1,
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

    int x_point_line_pos(double x1,
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

    /*
     *-----------------------------------------------------------------------------
     * Byte swapping test
     *-----------------------------------------------------------------------------
     */
    inline int x_swap_check()
    {

        long num = 1;
        void *ptr = &num;
        return *(char *)ptr;
    }

    extern void *SwapEndian(void *Addr, const int Nb);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // COMMON_H_
