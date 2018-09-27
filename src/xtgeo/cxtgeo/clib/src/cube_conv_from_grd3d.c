/*
 * ############################################################################
 * cube_conv_from_grd3d.c
 *
 * Take a general grid with one prop and convert it to cube format. If the
 * grid in grd3d format is fairly regular, the distortion shall be minimal,
 * otherwise it may be severe (logical perhaps...).
 *
 * ############################################################################
 * ToDo:
 * -
 * ############################################################################
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

int cube_conv_from_grd3d(
			 int    nx,
			 int    ny,
			 int    nz,
			 double *p_coord_v,
			 double *p_zcorn_v,
			 int    *p_actnum_v,
			 double *p_prop_v,
			 double *xori,
			 double *xinc,
			 double *yori,
			 double *yinc,
			 double *zori,
			 double *zinc,
			 double *rotation,
			 float  *p_cube_val,
			 double *cube_val_min,
			 double *cube_val_max,
			 int    option,
			 int    debug
			 )
{
    /* locals */
    char      s[24]="cube_conv_from_grd3d";
    double    vxori, vxmin, vxmax, vyori, vymin, vymax, vzori, vzmin, vzmax;
    double    vrotation, vdx, vdy, vdz;
    float     vpropmin, vpropmax, vpropavg;
    int       ier, option1, option2;
    long      nn;

    xtgverbose(debug);
    xtg_speak(s,2,"Entering routine <%s>",s);

    /* compute avg dx, dy, dz */

    option1 = 0; /* work with all cells; inactive cells also */
    option2 = 1; /* use cell _center_ metrics */

    ier = grd3d_geometrics( nx, ny, nz, p_coord_v, p_zcorn_v, p_actnum_v,
			    &vxori, &vyori, &vzori, &vxmin, &vxmax,
			    &vymin, &vymax, &vzmin, &vzmax,
			    &vrotation, &vdx, &vdy, &vdz,
			    option1, option2, debug );

    /* ier is an indication of "regular quality" in degrees, rotation
       in deg from X as normal math (anti clock)*/



    /* return the requested values */
    *xori = vxori;
    *xinc = vdx;

    *yori = vyori;
    *yinc = vdy;

    *zori = vzori;
    *zinc = vdz;

    *rotation = vrotation;

   /* values (property) */

    nn = nx * ny * nz;
    grd3d_copy_prop_doflo(nn, p_prop_v, p_cube_val, debug );

    /* min max of values (basicstats2 takes a float array instead of double)*/
    x_basicstats2( nn, (float)UNDEF, p_cube_val, &vpropmin, &vpropmax,
                   &vpropavg, debug );

    *cube_val_min = (double)vpropmin;
    *cube_val_max = (double)vpropmax;


    return 1;
}
