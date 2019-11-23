#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>



int u_read_ecl_bin_record (
			   char    *cname,
			   char    *ctype,
			   int     *nact,
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
			   int     *nact,
			   int     *tmp_int_v,
			   float   *tmp_float_v,
			   double  *tmp_double_v,
			   char    **tmp_string_v,
			   int     *tmp_logi_v,
			   FILE    *fc,
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


int u_ecl_eightletter (
		       char *cs
		       );


int gtc_msg(
	    int,
	    int,
	    char *,
	    ...
	    );

int gtc_speak(
	      char *,
	      int,
	      char *,
	      ...
	      );

int gtc_warn(
	      char *,
	      int,
	      char *,
	      ...
	      );

int gtc_error(
	      char *,
	      char *,
	      ...
	      );

int gtc_verbose(int);

int cmp_sort (
	      const void *vp, 
	      const void *vq
	      );





void import_roff_ascii_grid (
			     int     *num_act,
			     int     *num_subgrds,
			     float   *p_coord_v,
			     float   *p_zgrd3d_v,
			     int     *p_actnum_v,
			     int     *p_subgrd_v,
			     char    *filename,
			     int     debug
			     );

void import_roff_binary_grid (
			      int     *num_act,
			      int     *num_subgrds,
			      float   *p_coord_v,
			      float   *p_zgrd3d_v,
			      int     *p_actnum_v,
			      int     *p_subgrd_v,
			      char    *filename,
			      int     debug
			      );




void import_roff_ascii_par (
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


void import_roff_binary_par (
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

void roff_to_gtc_grid (
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
int chk_point_in_cell (
		       float x,
		       float y,
		       float z,
		       float coor[],
		       int   imethod,
		       int   debug
		       );

*/



int chk_point_in_polygon (
			  double x,
			  double y,
			  double *p_xp_v,
			  double *p_yp_v,
			  int    np1,
			  int    np2,
			  int    debug
			  );



int kvpt3s (
	    float pp[], 
	    float tri[][3],
	    int   *ier
	    ); 


void kmgmps (
	     float a[][3], 
	     int l[], 
	     float prmn, 
	     int m, 
	     int n, 
	     float eps, 
	     int *ier
	     );
void kmsubs (
	     float x[],
	     float a[][3],
	     int m,
	     int n,
	     float b[],
	     int l[],
	     int *ier
	     );

int roffbinstring(char *bla, FILE *fc);

float getfloatvalue     (char *name, FILE *fc);
int   getintvalue       (char *name, FILE *fc);

void getfloatarray  (float *array, int num, FILE *fc);
void getbytearray   (char *array, int num, FILE *fc);
void getintarray    (int *array, int num, FILE *fc);
void getchararray   (char **array, int num, FILE *fc);

