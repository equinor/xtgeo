#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
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

/*
 * Get cell centers for a grid.
 *
 * @param ncol Grid dimensions ncol/nx
 * @param nrow Grid dimensions nrow/ny
 * @param nlay Grid dimensions nlay/nz
 * @param coordsv Grid Z coordinates vector
 * @param zcornsv Grid Z corners vector
 * @param actumsv Active cells vector
 * @param asmasked Process grid cells as masked (return NaN for inactive cells)
 * @return Arrays with the X, Y, Z coordinates of the cell centers
 */
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
grid_cell_centers(const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const py::array_t<int> &actnumsv,
                  const bool asmasked = false)
{
    pybind11::array_t<double> xmid({ ncol, nrow, nlay });
    pybind11::array_t<double> ymid({ ncol, nrow, nlay });
    pybind11::array_t<double> zmid({ ncol, nrow, nlay });
    auto xmid_ = xmid.mutable_unchecked<3>();
    auto ymid_ = ymid.mutable_unchecked<3>();
    auto zmid_ = zmid.mutable_unchecked<3>();
    auto actnumsv_ = actnumsv.unchecked<3>();

    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
            for (size_t k = 0; k < nlay; k++) {
                size_t idx = i * nrow * nlay + j * nlay + k;
                if (asmasked && actnumsv_(i, j, k) == 0) {
                    xmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    ymid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    zmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                auto cr =
                  grid3d::cell_corners(i, j, k, ncol, nrow, nlay, coordsv, zcornsv);

                xmid_(i, j, k) = 0.125 * (cr[0] + cr[3] + cr[6] + cr[9] + cr[12] +
                                          cr[15] + cr[18] + cr[21]);
                ymid_(i, j, k) = 0.125 * (cr[1] + cr[4] + cr[7] + cr[10] + cr[13] +
                                          cr[16] + cr[19] + cr[22]);
                zmid_(i, j, k) = 0.125 * (cr[2] + cr[5] + cr[8] + cr[11] + cr[14] +
                                          cr[17] + cr[20] + cr[23]);
            }
        }
    }
    return std::make_tuple(xmid, ymid, zmid);
}

/*
 * Compute cell height above ffl (free fluid level), as input to water saturation. Will
 * return hbot, htop, hmid (bottom of cell, top of cell, midpoint), but compute method
 * depends on option: 1: cell center above ffl, 2: cell corners above ffl
 *
 * @param ncol Grid dimensions ncol/nx
 * @param nrow Grid dimensions nrow/ny
 * @param nlay Grid dimensions nlay/nz
 * @param coordsv Grid Z coordinates vector
 * @param zcornsv Grid Z corners vector
 * @param actumsv Active cells vector
 * @param ffl Free fluid level per cell
 * @param option 1: Use cell centers, 2 use cell corners
 * @return 3 arrays, top, bot, mid; all delta heights above ffl
 */

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
grid_height_above_ffl(const size_t ncol,
                      const size_t nrow,
                      const size_t nlay,
                      const py::array_t<double> &coordsv,
                      const py::array_t<float> &zcornsv,
                      const py::array_t<int> &actnumsv,
                      const py::array_t<float> &ffl,
                      const size_t option)
{
    pybind11::array_t<double> htop({ ncol, nrow, nlay });
    pybind11::array_t<double> hbot({ ncol, nrow, nlay });
    pybind11::array_t<double> hmid({ ncol, nrow, nlay });
    auto htop_ = htop.mutable_data();
    auto hbot_ = hbot.mutable_data();
    auto hmid_ = hmid.mutable_data();
    auto actnumsv_ = actnumsv.data();
    auto ffl_ = ffl.data();

    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
            for (size_t k = 0; k < nlay; k++) {
                size_t idx = i * nrow * nlay + j * nlay + k;
                if (actnumsv_[idx] == 0) {
                    htop_[idx] = UNDEF;
                    hbot_[idx] = UNDEF;
                    hmid_[idx] = UNDEF;
                    continue;
                }
                auto cr =
                  grid3d::cell_corners(i, j, k, ncol, nrow, nlay, coordsv, zcornsv);
                if (option == 1) {
                    htop_[idx] = ffl_[idx] - 0.25 * (cr[2] + cr[5] + cr[8] + cr[11]);
                    hbot_[idx] = ffl_[idx] - 0.25 * (cr[14] + cr[17] + cr[20] + cr[23]);
                } else if (option == 2) {
                    double upper = cr[2];
                    for (size_t indx = 5; indx <= 11; indx += 3) {
                        upper = std::min(upper, cr[indx]);
                    }
                    htop_[idx] = ffl_[idx] - upper;

                    double lower = cr[14];
                    for (size_t indx = 17; indx <= 23; indx += 3) {
                        lower = std::max(lower, cr[indx]);
                    }
                    hbot_[idx] = ffl_[idx] - lower;
                }
                htop_[idx] = std::max(htop_[idx], 0.0);
                hbot_[idx] = std::max(hbot_[idx], 0.0);
                hmid_[idx] = 0.5 * (htop_[idx] + hbot_[idx]);
            }
        }
    }
    return std::make_tuple(htop, hbot, hmid);
}

}  // namespace xtgeo::grid3d
