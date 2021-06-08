// Notice: I, J, K are "one based" as in common cell counting

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

long
x_ijk2ib(long i, long j, long k, long nx, long ny, long nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        logger_error(LI, FI, FU, "i: %ld, nx: %ld, j: %ld, ny: %ld, k: %ld, nz: %ld\n",
                     i, nx, j, ny, k, nz);
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        logger_error(LI, FI, FU, "i: %ld, j: %ld, k: %ld\n", i, j, k);
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
x_ijk2ic(long i, long j, long k, long nx, long ny, long nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        logger_error(LI, FI, FU, "i: %ld, nx: %ld, j: %ld, ny: %ld, k: %ld, nz: %ld\n",
                     i, nx, j, ny, k, nz);
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        logger_error(LI, FI, FU, "i: %ld, j: %ld, k: %ld\n", i, j, k);
        return -2;
    }

    long ic = (i - 1) * nz * ny;
    ic = ic + (j - 1) * nz;
    ic = ic + k;

    if (ia_start == 0)
        ic--;

    return ic;
}
