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
#include <scn/scan.h>
#include <xtgeo/xtgeo.h>
#include "logger.h"

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
    fseek(fd, 0, SEEK_SET);
    auto result =
      scn::scan<int, int, double, double, double, double, double, double, int, double,
                double, double, int, int, int, int, int, int, int>(
        fd, "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}");
    if (!result) {
        logger_error(LI, FI, FU,
                     "Something went wrong with Irap ASCII import. Report as BUG");
        logger_error(LI, FI, FU, "Error: %s", result.error().msg());
        return -1;
    }
    std::tie(std::ignore, *ny, *xinc, *yinc, *xori, std::ignore, *yori, std::ignore,
             *nx, *rot, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore,
             std::ignore, std::ignore, std::ignore, std::ignore) = result->values();

    if (*rot < 0.0)
        *rot = *rot + 360.0;

    if (mode == 0)
        return EXIT_SUCCESS;

    if (*nx < 0 || *ny < 0 || *nx * *ny > nmap) {
        logger_error(LI, FI, FU,
                     "Incorrect dimension encountered while importing Irap ASCII");
        logger_error(LI, FI, FU, "nx: %d, ny: %d, nmap: %d");
        return -1;
    }

    /* read values */
    long ncount = 0;
    for (int i = 0; i < nmap; i++) {
        auto parsed_value = scn::scan<double>(fd, "{}");
        if (!result) {
            logger_error(LI, FI, FU, "Failed to read values during Irap ASCII import.");
            logger_error(LI, FI, FU, "Error: %s", result.error().msg());
            return -1;
        }

        double value = parsed_value.value().value();

        if (value == UNDEF_MAP_IRAP) {
            value = UNDEF_MAP;
        } else {
            value = float(value);
            ++ncount;
        }

        // convert to C order (column major to row major order)
        int ic = i / *nx + (i % *nx) * *ny;
        p_map_v[ic] = value;
    }

    *ndef = ncount;

    return EXIT_SUCCESS;
}
