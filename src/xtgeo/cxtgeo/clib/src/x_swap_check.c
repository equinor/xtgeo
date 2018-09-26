
/* "To determine what your machine is, you can do (in the C programming
   language):If you get 1 as a result in B, your machine is 
   little endian, if you get 0 it's big endian. The PC is little 
   endian, the SGI machines and Macs are big endian." 

   Picked up from Internet by JCR
   22-APR-2003
*/

#include "libxtg_.h"

int x_swap_check ()
{

  long L=1; 
  void *Ptr=&L; 
  char B=*(char*)Ptr;
  int  i;

  i=B;

  return i;
}

/* used in ROFF files. Values:
 *
 * i = -1 or lower: query byteroder status
 * i =  0 Machine is big endian, import is bigendian (no swap)
 * i =  1 Machine is little endian, import is little endian (no swap)
 * i =  2 Machine is little endian, import file is big endian (swap needed)
 * i =  3 Machine is big endian, import file is little endian (swap needed)
 *
 */

int x_byteorder (int i)
{
    static int byteorder;

    if (i>-1) {
	byteorder=i;
    }
    return byteorder;
}

