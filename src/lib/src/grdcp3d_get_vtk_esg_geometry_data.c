/*
****************************************************************************************
*
* NAME:
*    grdcp3d_get_vtk_esg_geometry_data.c
*
* DESCRIPTION:
*    Get geometry data that is suitable as input to VTK's vtkExplicitStructuredGrid.
*    See: https://vtk.org/doc/nightly/html/classvtkExplicitStructuredGrid.html#details
*
*    Basically what we have to do is build an unstructured grid representation where
*    all grid cells are represented as hexahedrons with explicit connectivities.
*    The connectivity table refers into the accompanying vertex table.
*
*    In VTK, cells order increases in I fastest, then J, then K.
*
*    This function also tries to remove/weld duplicate vertices, but this is a WIP.
*
* ARGUMENTS:
*    ncol, nrow, nlay     i     Dimensions
*    coordsv              i     Coordinate vector (xtgeo fmt)
*    zcornsv              i     ZCORN vector (xtgeo fmt)
*    vert_arr             o     Will be populated with vertex coordinates (XYZ, XYZ,
*etc) conn_arr             o     Will be populated with the cell connectivities
*
* RETURNS:
*    The actual number of XYZ vertices written into vert_arr.
*    Also updates output arrays vert_arr and conn_arr.
*
* TODO/ISSUES/BUGS:
*    - verify/enhance removal of duplicate nodes
*    - optimize the way cell corners are accessed
*    - reduce memory usage in GeoMaker helper class
*
* LICENCE:
*    CF XTGeo's LICENSE
***************************************************************************************
*/
#include <stdlib.h>
#include <string.h>

#include <xtgeo/xtgeo.h>

// Forward declarations of static helpers
static struct GeoMaker *
gm_create(long cellCountI,
          long cellCountJ,
          long cellCountK,
          double *targetVertexBuffer,
          long *targetConnBuffer);
static void
gm_destroy(struct GeoMaker *gm);
static void
gm_addVertex(struct GeoMaker *gm, long i, long j, long k, double v[3]);
static long
gm_vertexCount(const struct GeoMaker *gm);

long
grdcp3d_get_vtk_esg_geometry_data(long ncol,
                                  long nrow,
                                  long nlay,

                                  double *coordsv,
                                  long ncoordin,
                                  float *zcornsv,
                                  long nlaycornin,

                                  double *vert_arr,
                                  long n_vert_arr,
                                  long *conn_arr,
                                  long n_conn_arr)
{
    const long cellCount = ncol * nrow * nlay;
    const long maxVertexCount = 8 * cellCount;
    const long expectedConnCount = 8 * cellCount;

    if (3 * maxVertexCount > n_vert_arr) {
        throw_exception("Allocated size of vertex array is too small");
        return 0;
    }
    if (expectedConnCount != n_conn_arr) {
        throw_exception("Allocated size of connectivity array is too small");
        return 0;
    }

    //        7---------6                   6---------7
    //       /|        /|                  /|        /|
    //      / |       / |                 / |       / |
    //     4---------5  |                4---------5  |
    //     |  3------|--2                |  2------|--3
    //     | /       | /                 | /       | /
    //     |/        |/                  |/        |/
    //     0---------1                   0---------1
    //         VTK                          XTGeo
    //
    //  VTK hex cell definition from:
    //  https://kitware.github.io/vtk-examples/site/VTKBook/05Chapter5/#54-cell-types

    struct GeoMaker *gm = gm_create(ncol, nrow, nlay, vert_arr, conn_arr);
    double crs[24];
    for (long k = 0; k < nlay; k++) {
        for (long j = 0; j < nrow; j++) {
            for (long i = 0; i < ncol; i++) {
                grdcp3d_corners(i, j, k, ncol, nrow, nlay, coordsv, 0, zcornsv, 0, crs);

                gm_addVertex(gm, i, j, k, &crs[0 * 3]);
                gm_addVertex(gm, i + 1, j, k, &crs[1 * 3]);
                gm_addVertex(gm, i + 1, j + 1, k, &crs[3 * 3]);
                gm_addVertex(gm, i, j + 1, k, &crs[2 * 3]);
                gm_addVertex(gm, i, j, k + 1, &crs[4 * 3]);
                gm_addVertex(gm, i + 1, j, k + 1, &crs[5 * 3]);
                gm_addVertex(gm, i + 1, j + 1, k + 1, &crs[7 * 3]);
                gm_addVertex(gm, i, j + 1, k + 1, &crs[6 * 3]);
            }
        }
    }

    const long finalVertexCount = gm_vertexCount(gm);

    gm_destroy(gm);
    gm = NULL;

    return finalVertexCount;
}

// General helpers
// ====================================================================================

// ------------------------------------------------------------------------------------
static long
coordsEqual(double a[3], double b[3])
{
    return (a[0] == b[0] && a[1] == b[1] && a[2] == b[2]);
}

// The Geomaker "class"
// ====================================================================================
typedef struct GeoMaker
{
    long vertexCountI;
    long vertexCountJ;
    long vertexCountK;

    long *globToOutputVertIdx;

    double *vertexArr;
    long numVerticesAdded;
    long *connArr;
    long numConnAdded;
} GeoMaker;

// ------------------------------------------------------------------------------------
static GeoMaker *
gm_create(long cellCountI,
          long cellCountJ,
          long cellCountK,
          double *targetVertexBuffer,
          long *targetConnBuffer)
{
    GeoMaker *gm = (GeoMaker *)malloc(sizeof(GeoMaker));

    gm->vertexCountI = cellCountI + 1;
    gm->vertexCountJ = cellCountJ + 1;
    gm->vertexCountK = cellCountK + 1;

    const long maxGlobVertexIdx =
      gm->vertexCountI * gm->vertexCountJ * gm->vertexCountK;
    gm->globToOutputVertIdx = malloc(8 * maxGlobVertexIdx * sizeof(long));
    for (int i = 0; i < 8 * maxGlobVertexIdx; i++) {
        gm->globToOutputVertIdx[i] = -1;
    }

    gm->vertexArr = targetVertexBuffer;
    gm->numVerticesAdded = 0;
    gm->connArr = targetConnBuffer;
    gm->numConnAdded = 0;

    return gm;
}

// ------------------------------------------------------------------------------------
static void
gm_destroy(GeoMaker *gm)
{
    if (gm) {
        free(gm->globToOutputVertIdx);
        free(gm);
    }
}

// ------------------------------------------------------------------------------------
static long
gm_vertexIJKToIdx(const GeoMaker *gm, long i, long j, long k)
{
    return k * (gm->vertexCountI * gm->vertexCountJ) + j * gm->vertexCountI + i;
}

// ------------------------------------------------------------------------------------
static long
gm_findVertex(const GeoMaker *gm, long i, long j, long k, double v[3])
{
    const long globalVertexIdx = gm_vertexIJKToIdx(gm, i, j, k);

    for (int offset = 0; offset < 8; offset++) {
        const long mappedIdx = gm->globToOutputVertIdx[8 * globalVertexIdx + offset];
        if (mappedIdx < 0) {
            return -1;
        }

        if (coordsEqual(v, &gm->vertexArr[3 * mappedIdx])) {
            return mappedIdx;
        }
    }

    return -1;
}

// ------------------------------------------------------------------------------------
static long
gm_findVertexAlongIJ(const GeoMaker *gm, long i, long j, long k, double v[3])
{
    if (k > 0) {
        const long mappedIdx = gm_findVertex(gm, i, j, k - 1, v);
        if (mappedIdx >= 0) {
            return mappedIdx;
        }
    }

    if (k < gm->vertexCountK - 1) {
        const long mappedIdx = gm_findVertex(gm, i, j, k + 1, v);
        if (mappedIdx >= 0) {
            return mappedIdx;
        }
    }

    return -1;
}

// ------------------------------------------------------------------------------------
static void
gm_addVertex(GeoMaker *gm, long i, long j, long k, double v[3])
{
    const long globalVertexIdx = gm_vertexIJKToIdx(gm, i, j, k);

    long mappedIdx = -1;
    for (int offset = 0; offset < 8; offset++) {
        mappedIdx = gm->globToOutputVertIdx[8 * globalVertexIdx + offset];
        if (mappedIdx < 0) {
            // Take a peek above and below
            mappedIdx = gm_findVertexAlongIJ(gm, i, j, k, v);
            if (mappedIdx >= 0) {
                gm->globToOutputVertIdx[8 * globalVertexIdx + offset] = mappedIdx;
                gm->connArr[gm->numConnAdded] = mappedIdx;
                gm->numConnAdded++;
                return;
            }

            mappedIdx = gm->numVerticesAdded;

            memcpy(&gm->vertexArr[3 * mappedIdx], v, 3 * sizeof(double));
            gm->numVerticesAdded++;

            gm->globToOutputVertIdx[8 * globalVertexIdx + offset] = mappedIdx;

            break;
        }

        if (coordsEqual(v, &gm->vertexArr[3 * mappedIdx])) {
            break;
        }
    }

    gm->connArr[gm->numConnAdded] = mappedIdx;
    gm->numConnAdded++;
}

// ------------------------------------------------------------------------------------
static long
gm_vertexCount(const GeoMaker *gm)
{
    return gm->numVerticesAdded;
}
