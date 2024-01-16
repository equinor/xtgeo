#include <cstddef>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Compute bulk volume of cells in a grid. Tests shows that this is very close
 * to what RMS will compute; almost identical
 *
 * @param ncol Grid dimensions ncol/nx
 * @param nrow Grid dimensions nrow/ny
 * @param nlay Grid dimensions nlay/nz
 * @param coordsv Grid Z coordinates vector
 * @param zcornsv Grid Z corners vector
 * @param actumsv Active cells vector
 * @param asmaked Process grid cells as masked
 * @return An array containing the volume of every cell
 */
py::array_t<double>
grid_cell_volumes(const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const py::array_t<int> &actnumsv,
                  const int precision,
                  const bool asmasked = false)
{
    pybind11::array_t<double> cellvols({ ncol, nrow, nlay });
    auto cellvols_ = cellvols.mutable_data();
    auto actnumsv_ = actnumsv.data();

    for (auto i = 0; i < ncol; i++) {
        for (auto j = 0; j < nrow; j++) {
            for (auto k = 0; k < nlay; k++) {
                auto idx = i * nrow * nlay + j * nlay + k;
                if (asmasked && actnumsv_[idx] == 0) {
                    cellvols_[idx] = UNDEF;
                    continue;
                }
                auto corners =
                  grid3d::cell_corners(i, j, k, ncol, nrow, nlay, coordsv, zcornsv);
                cellvols_[idx] = geometry::hexahedron_volume(corners, precision);
            }
        }
    }
    return cellvols;
}

}  // namespace xtgeo::grid3d
