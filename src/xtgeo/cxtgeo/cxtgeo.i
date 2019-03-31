%module "cxtgeo"

%{
  #define SWIG_FILE_WITH_INIT
  #include "clib/src/libxtg.h"
  #include "clib/src/libxtg_.h"
  #include "clib/src/swap_endian.h"
%}

%include typemaps.i


%include cpointer.i
%include carrays.i
%include cstring.i

 /* output strings, bounded with \0 and a max length */
%cstring_bounded_output(char *swig_bnd_char_10k, 10000);
%cstring_bounded_output(char *swig_bnd_char_100k, 100000);
%cstring_bounded_output(char *swig_bnd_char_1m, 1000000);


/* Create some functions for working with "int *" etc */
%pointer_functions(int, intpointer);
%pointer_functions(long, longpointer)
%pointer_functions(float, floatpointer)
%pointer_functions(double, doublepointer);
%pointer_functions(char, charpointer)

%array_functions(int, intarray)
%array_functions(long, longarray)
%array_functions(float, floatarray)
%array_functions(double, doublearray)
%array_functions(char, chararray)


//=============================================================================
// Numpy tranforms
//=============================================================================
%include "numpy.i"
%numpy_typemaps(double, NPY_DOUBLE, long)
%numpy_typemaps(float, NPY_FLOAT, long)
%numpy_typemaps(int, NPY_INT, long)
%init %{
import_array();
%}


//=============================================================================
// Magic typemaps (cf libxtg.h)
//=============================================================================
%apply double *OUTPUT { double *swig_dbl_out_p1 };
%apply double *OUTPUT { double *swig_dbl_out_p2 };
%apply double *OUTPUT { double *swig_dbl_out_p3 };
%apply double *OUTPUT { double *swig_dbl_out_p4 };
%apply double *OUTPUT { double *swig_dbl_out_p5 };
%apply double *OUTPUT { double *swig_dbl_out_p6 };
%apply double *OUTPUT { double *swig_dbl_out_p7 };
%apply double *OUTPUT { double *swig_dbl_out_p8 };
%apply double *OUTPUT { double *swig_dbl_out_p9 };

%apply int *OUTPUT { int *swig_int_out_p1 };
%apply int *OUTPUT { int *swig_int_out_p2 };
%apply int *OUTPUT { int *swig_int_out_p3 };

%apply long *OUTPUT { long *swig_lon_out_p1 };

%apply int *INOUT { int *swig_int_inout_p1 };
%apply int *INOUT { int *swig_int_inout_p2 };


// numpies (1D)
// IN int32 no 1
%apply (int* IN_ARRAY1, long DIM1) {(int *swig_np_int_in_v1,
                                     long n_swig_np_int_in_v1)};
// IN int32 no 2
%apply (int* IN_ARRAY1, long DIM1) {(int *swig_np_int_in_v2,
                                     long n_swig_np_int_in_v2)};
// IN float32 no 1
%apply (float* IN_ARRAY1, long DIM1) {(float *swig_np_flt_in_v1,
                                       long n_swig_np_flt_in_v1)};
// IN float32 no 2
%apply (float* IN_ARRAY1, long DIM1) {(float *swig_np_flt_in_v1,
                                       long n_swig_np_flt_in_v1)};

// IN float64 / double no 1
%apply (double* IN_ARRAY1, long DIM1) {(double *swig_np_dbl_in_v1,
                                        long n_swig_np_dbl_in_v1)};
// IN float64 / double no 2
%apply (double* IN_ARRAY1, long DIM1) {(double *swig_np_dbl_in_v2,
                                        long n_swig_np_dbl_in_v2)};

// IN float64 / double no 3
%apply (double* IN_ARRAY1, long DIM1) {(double *swig_np_dbl_in_v3,
                                        long n_swig_np_dbl_in_v3)};

// IN float64 / double no 4
%apply (double* IN_ARRAY1, long DIM1) {(double *swig_np_dbl_in_v4,
                                        long n_swig_np_dbl_in_v4)};

// INPLACE int no 1
%apply (int* INPLACE_ARRAY1, long DIM1) {(int *swig_np_int_inplace_v1,
                                          long n_swig_np_int_inplace_v1)};

// INPLACE float no 1
%apply (float* INPLACE_ARRAY1, long DIM1) {(float *swig_np_flt_inplace_v1,
                                            long n_swig_np_flt_inplace_v1)};

// INPLACE float64 / double no 1
%apply (double* INPLACE_ARRAY1, long DIM1) {(double *swig_np_dbl_inplace_v1,
                                             long n_swig_np_dbl_inplace_v1)};
// INPLACE float64 / double no 2
%apply (double* INPLACE_ARRAY1, long DIM1) {(double *swig_np_dbl_inplace_v2,
                                             long n_swig_np_dbl_inplace_v2)};
// INPLACE float64 / double no 3
%apply (double* INPLACE_ARRAY1, long DIM1) {(double *swig_np_dbl_inplace_v3,
                                             long n_swig_np_dbl_inplace_v3)};

// ARGOUT float64 / double no 1
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double *swig_np_dbl_aout_v1,
                                            long n_swig_np_dbl_aout_v1)};
// ARGOUT float64 / double no 2
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double *swig_np_dbl_aout_v2,
                                            long n_swig_np_dbl_aout_v2)};
// ARGOUT float64 / double no 3
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double *swig_np_dbl_aout_v3,
                                            long n_swig_np_dbl_aout_v3)};
// ARGOUT float64 / double no 4
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double *swig_np_dbl_aout_v4,
                                            long n_swig_np_dbl_aout_v4)};

// ARGOUT float32 / no 1
%apply (float* ARGOUT_ARRAY1, long DIM1) {(float *swig_np_flt_aout_v1,
                                           long n_swig_np_flt_aout_v1)};
// ARGOUT float32 / no 2
%apply (float* ARGOUT_ARRAY1, long DIM1) {(float *swig_np_flt_aout_v2,
                                           long n_swig_np_flt_aout_v2)};

// ARGOUT int no 1
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int *swig_np_int_aout_v1,
                                         long n_swig_np_int_aout_v1)};
// ARGOUT int no 2
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int *swig_np_int_aout_v2,
                                         long n_swig_np_int_aout_v2)};
// ARGOUT int no 3
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int *swig_np_int_aout_v3,
                                         long n_swig_np_int_aout_v3)};
// ARGOUT int no 4
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int *swig_np_int_aout_v4,
                                         long n_swig_np_int_aout_v4)};
// ARGOUT int no 5
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int *swig_np_int_aout_v5,
                                         long n_swig_np_int_aout_v5)};


//=============================================================================
// Inline tranforms
//=============================================================================
// double version
%apply (double* ARGOUT_ARRAY1, long DIM1) {(double* np, long len)};
%inline %{
    /* DOUBLE: convert carray to numpy array (copy from carray to numpy)*/
    void swig_carr_to_numpy_1d(double* np, long len, double *carr) {
	long i;
	for (i=0;i<len;i++) {
	    np[i] = carr[i];
	}
    }
    %}

// float 32 version
%apply (float* ARGOUT_ARRAY1, long DIM1) {(float* npf, long lenf)};
%inline %{
    /* FLOAT: convert carray to numpy array (copy from carray to numpy)*/
    void swig_carr_to_numpy_f1d(float* npf, long lenf, float *carrf) {
	long i;
	for (i=0;i<lenf;i++) {
	    npf[i] = carrf[i];
	}
    }
    %}

// integer version
%apply (int* ARGOUT_ARRAY1, long DIM1) {(int* npi, long nnlen)};
%inline %{
    /* INT: convert carray to numpy array (copy from carray to numpy)*/
    void swig_carr_to_numpy_i1d(int* npi, long nnlen, int *carri) {
       long i;
	for (i=0;i<nnlen;i++) {
	    npi[i] = carri[i];
	    /* if (i==0) printf("Item 0 <%d>\n", npi[i]); */
	    /* if (i==100) printf("Item 100 <%d>\n", npi[i]); */
	    /* if (i==(nnlen-1)) printf("Item %d <%d>\n", nnlen-1, npi[i]); */
	}
    }
    %}

// ============================================================================

// double
%apply (double* IN_ARRAY1, long DIM1) {(double* npinput, long len2)};
%inline %{
    /* copy numpy array values to existing carray, ie update carray*/
    void swig_numpy_to_carr_1d(double* npinput, long len2, double *cxarr) {
	long i;
	for (i=0;i<len2;i++) {
	    cxarr[i] = npinput[i];
	}
    }
    %}

// float
%apply (float* IN_ARRAY1, long DIM1) {(float* npinputf, long len2f)};
%inline %{
    /* copy numpy array values to existing carray, ie update carray*/
    void swig_numpy_to_carr_f1d(float* npinputf, long len2f, float *cxarrf) {
	long i;
	for (i=0;i<len2f;i++) {
	    cxarrf[i] = npinputf[i];
	}
    }
    %}


//int
%apply (int* IN_ARRAY1, long DIM1) {(int* npinputi, long len2i)};
%inline %{
    /* copy numpy array values to existing carray, ie update carray*/
    void swig_numpy_to_carr_i1d(int* npinputi, long len2i, int *cxarri) {
	long i;
	for (i=0;i<len2i;i++) {
	    cxarri[i] = npinputi[i];
	}
    }
    %}

//=============================================================================
//The rest of the XTGeo functions:
//Really, this is the lazy way of doing it!

%include "clib/src/libxtg.h";
