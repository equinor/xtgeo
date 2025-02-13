/*
****************************************************************************************
*
* NAME:
*    surf_import_irap_ascii.cpp
*
* AUTHOR(S):
*
*
* DESCRIPTION:
*    Import a map on Irap ascii format.
*
* ARGUMENTS:
*    fd             i     File handle
*    mx             i     Map dimension X (I)
*    my             i     Map dimension Y (J)
*    xori           i     X origin coordinate
*    yori           i     Y origin coordinate
*    xinc           i     X increment
*    yinc           i     Y increment
*    rot            i     Rotation (degrees, from X axis, anti-clock)
*    p_surf_v       i     1D pointer to map/surface values pointer array
*    option         i     0: read only dimensions (for memory alloc), 1 all
*
* RETURNS:
*    Function: 0: upon success. If problems <> 0:
*
* TODO/ISSUES/BUGS:
*    Issue: The surf_* routines in XTGeo will include rotation, and origins
*           (not xmin etc ) and steps are used to define the map extent.
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/
#include <charconv>
#include <stdio.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "logger.h"

template<typename T, typename... U>
int
_read_headers(FILE *fd, int args_read, T &&arg, U &&...args)
{
    static thread_local char input[100];
    fscanf(fd, " %s", input);
    
    auto [ptr, ec] = std::from_chars(input, input + 100, arg);
    if (ec != std::errc{})
        return args_read;
    ++args_read;

    if constexpr (sizeof...(args) > 0)
        args_read = _read_headers(fd, args_read, args...);

    return args_read;
}

template<typename... T>
int
read_headers(FILE *fd, T &&...args)
{
    int args_read = 0;
    return _read_headers(fd, args_read, args...);
}

int
surf_import_irap_ascii(FILE *fd,
                       int mode,
                       int *nx,
                       int *ny,
                       long *ndef,
                       double *xori,
                       double *yori,
                       double *xinc,
                       double *yinc,
                       double *rot,
                       double *p_map_v,
                       long nmap,
                       int option)
{

    /* locals*/
    int idum, ib, ic, i, j, k, iok;
    long ncount;

    double value, ddum;

    ncount = 0;

    fseek(fd, 0, SEEK_SET);

    /* read header */
    iok = read_headers(fd, idum, *ny, *xinc, *yinc, *xori, ddum, *yori, ddum, *nx, *rot,
                       ddum, ddum, idum, idum, idum, idum, idum, idum, idum);

    if (iok < 19) {
        logger_error(LI, FI, FU,
                     "Something went wrong with Irap ASCII import. Report as BUG");
        logger_error(LI, FI, FU, "IOK is %d", iok);
        return -1;
    }

    if (*rot < 0.0)
        *rot = *rot + 360.0;

    if (mode == 0) {
        return EXIT_SUCCESS;
    }

    /* read values */
    for (ib = 0; ib < nmap; ib++) {
        static thread_local char input[100];
        fscanf(fd, " %s", input);
        auto [ptr, ec] = std::from_chars(input, input + 100, value);

        if (value == UNDEF_MAP_IRAP) {
            value = UNDEF_MAP;
        } else {
            value = float(value);
            ncount++;
        }

        /* convert to C order */
        x_ib2ijk(ib, &i, &j, &k, *nx, *ny, 1, 0);
        ic = x_ijk2ic(i, j, 1, *nx, *ny, 1, 0);
        if (ic < 0) {
            throw_exception("Convert to c order resulted in index outside boundary in "
                            "surf_import_irap_ascii");
            return EXIT_FAILURE;
        }

        p_map_v[ic] = value;
    }

    *ndef = ncount;

    return EXIT_SUCCESS;
}
