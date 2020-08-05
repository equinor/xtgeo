// Notice: I, J, K are "one based" as in common cell counting

#include "libxtg.h"
#include "libxtg_.h"

long
x_ijk2ib(int i, int j, int k, int nx, int ny, int nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        return -2;
    }

    long ib = ((long)k - 1) * (long)nx * (long)ny;
    ib = ib + ((long)j - 1) * (long)nx;
    ib = ib + i;

    if (ia_start == 0)
        ib--;

    return ib;
}

/* c order counting, where K is looping fastest, them J, then I */
long
x_ijk2ic(int i, int j, int k, int nx, int ny, int nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        return -2;
    }


    long ic = ((long)i - 1) * (long)nz * (long)ny;
    ic = ic + ((long)j - 1) * nz;
    ic = ic + (long)k;

    if (ia_start == 0)
        ic--;

    return ic;
}
