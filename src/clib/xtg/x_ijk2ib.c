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

    long ib = (k - 1) * nx * ny;
    ib = ib + (j - 1) * nx;
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

    long ic = (i - 1) * nz * ny;
    ic = ic + (j - 1) * nz;
    ic = ic + k;

    if (ia_start == 0)
        ic--;

    return ic;
}
