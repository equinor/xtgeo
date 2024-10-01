#include <pybind11/stl.h>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/xtgeo.h>

namespace xtgeo::geometry {

/*
 * Estimate the volume of a hexahedron i.e. a cornerpoint cell. This is a
 * nonunique entity, but it is approximated by computing two different ways of
 * top/base splitting and average those.
 *
 *    6          7
 *     2        3
 *      |------|   Example of split along
 *      |    / |   0 - 3 at top and 4 - 7
 *      |   /  |   at base. Alternative is
 *      | /    |   1 - 2 at top and 5 - 6
 *      |------|   at base
 *     0        1
 *    4          5
 *
 * Note however... this fails if the cell is concave and convace cells needs
 * special attention
 *
 * @param corners A vector of doubles, length 24
 * @param precision The precision to calculate the volume to
 * @return The volume of the hexahadron
 */
double
hexahedron_volume(const std::vector<double> &corners, const int precision)
{
    // Avoid cells that collapsed in some way
    if (hexahedron_dz(corners) < std::numeric_limits<double>::epsilon()) {
        return 0.0;
    }

    double crn[8][3]{};
    auto idx = 0;
    for (auto i = 0; i < 8; i++) {
        for (auto j = 0; j < 3; j++) {
            crn[i][j] = corners[idx++];
        }
    }

    double vol = 0.0;
    double tetrahedron[12]{};
    for (auto i = 1; i <= precision; i++) {
        double tetra_vol = 0.0;
        for (auto j = 0; j < 6; j++) {
            idx = 0;
            for (auto k = 0; k < 4; k++) {
                auto tetra_idx = TETRAHEDRON_VERTICES[i - 1][j][k];
                tetrahedron[idx++] = crn[tetra_idx][0];
                tetrahedron[idx++] = crn[tetra_idx][1];
                tetrahedron[idx++] = crn[tetra_idx][2];
            }
            tetra_vol += x_tetrahedron_volume(tetrahedron, 12);
        }
        vol = (vol * (i - 1) + tetra_vol) / i;
    }
    return vol;
}

}  // namespace xtgeo::geometry
