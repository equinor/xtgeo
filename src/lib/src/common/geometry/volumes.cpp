#include <pybind11/stl.h>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>

namespace xtgeo::geometry {

using grid3d::CellCorners;

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
 * @param corners A CellCorners instance
 * @param precision The precision to calculate the volume to
 * @return The volume of the hexahadron
 */
double
hexahedron_volume(const CellCorners &corners, const int precision)
{
    // Avoid cells that collapsed in some way
    if (hexahedron_dz(corners) < numerics::EPSILON) {
        return 0.0;
    }

    std::array<std::array<double, 3>, 8> crn = {
        { { corners.upper_sw.x, corners.upper_sw.y, corners.upper_sw.z },
          { corners.upper_se.x, corners.upper_se.y, corners.upper_se.z },
          { corners.upper_nw.x, corners.upper_nw.y, corners.upper_nw.z },
          { corners.upper_ne.x, corners.upper_ne.y, corners.upper_ne.z },
          { corners.lower_sw.x, corners.lower_sw.y, corners.lower_sw.z },
          { corners.lower_se.x, corners.lower_se.y, corners.lower_se.z },
          { corners.lower_nw.x, corners.lower_nw.y, corners.lower_nw.z },
          { corners.lower_ne.x, corners.lower_ne.y, corners.lower_ne.z } }
    };

    double vol = 0.0;
    double tetrahedron[12]{};
    for (auto i = 1; i <= precision; i++) {
        double tetra_vol = 0.0;
        for (auto j = 0; j < 6; j++) {
            auto idx = 0;
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
