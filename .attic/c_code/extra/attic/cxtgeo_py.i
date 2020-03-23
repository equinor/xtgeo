%module "cxtgeo"
%{
#include "libxtg.h"
#include "libxtg_.h"
#include "swap_endian.h"
%}


//%include typemaps.i
//%include pointer.i

// this should replace pointer.i:

// give routines such as $i=new_intarray(100); delete_intarray($i)??;
// intpointer_setitem($i, 40, 202); $ival=intpointer_getitem($i,40);
%include "carrays.i"
%array_functions(int, intarray)
%array_functions(float, floatarray)
%array_functions(double, doublearray)
%array_functions(char, chararray)

// give routines such as $i=new_intpointer(); $j=copy_intpointer(44)??; delete_intpointer($i)
// intpointer_assign($i, 40); $ival=intpointer_value($i);
%include "cpointer.i"
%pointer_functions(int, intpointer)
%pointer_functions(float, floatpointer)
%pointer_functions(double, doublepointer)
%pointer_functions(char, charpointer)


// Really, this is the lazy way of doing it!
%include "libxtg.h";

