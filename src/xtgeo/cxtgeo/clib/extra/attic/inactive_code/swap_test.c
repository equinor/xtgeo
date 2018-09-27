
/* "To determine what your machine is, you can do (in the C programming
   language):If you get 1 as a result in B, your machine is 
   little endian, if you get 0 it's big endian. The PC is little 
   endian, the SGI machines and Macs are big endian." 

   Picked up from Internet by JCR
   22-APR-2003
*/

#include "libxtg_.h"

int swap_test ()
{

  long L=1; 
  void *Ptr=&L; 
  char B=*(char*)Ptr;
  int  i;

  i=B;

  return i;
}
