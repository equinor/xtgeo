#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Compute bulk volume of cells in a grid. Tests shows that this is very close
 * to what RMS will compute; almost identical
 *
 * @param grd Grid struct
 * @param precision The precision to calculate the volume to
 * @param asmasked Process grid cells as masked (bool)
 * @return An array containing the volume of every cell
 */
py::array_t<double>
get_cell_volumes(const Grid &grd, const int precision, const bool asmasked)
{
    pybind11::array_t<double> cellvols({ grd.ncol, grd.nrow, grd.nlay });
    auto cellvols_ = cellvols.mutable_data();
    auto actnumsv_ = grd.actnumsv.data();

    for (auto i = 0; i < grd.ncol; i++) {
        for (auto j = 0; j < grd.nrow; j++) {
            for (auto k = 0; k < grd.nlay; k++) {
                auto idx = i * grd.nrow * grd.nlay + j * grd.nlay + k;
                if (asmasked && actnumsv_[idx] == 0) {
                    cellvols_[idx] = numerics::UNDEF_XTGEO;
                    continue;
                }
                auto crn = grid3d::get_cell_corners_from_ijk(grd, i, j, k);
                cellvols_[idx] = geometry::hexahedron_volume(crn, precision);
            }
        }
    }
    return cellvols;
}

/*
 * Get cell centers for a grid.
 *
 * @param grd Grid struct
 * @param asmasked Process grid cells as masked (return NaN for inactive cells)
 * @return Arrays with the X, Y, Z coordinates of the cell centers
 */
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_cell_centers(const Grid &grd, const bool asmasked)
{
    pybind11::array_t<double> xmid({ grd.ncol, grd.nrow, grd.nlay });
    pybind11::array_t<double> ymid({ grd.ncol, grd.nrow, grd.nlay });
    pybind11::array_t<double> zmid({ grd.ncol, grd.nrow, grd.nlay });
    auto xmid_ = xmid.mutable_unchecked<3>();
    auto ymid_ = ymid.mutable_unchecked<3>();
    auto zmid_ = zmid.mutable_unchecked<3>();
    auto actnumsv_ = grd.actnumsv.unchecked<3>();

    for (size_t i = 0; i < grd.ncol; i++) {
        for (size_t j = 0; j < grd.nrow; j++) {
            for (size_t k = 0; k < grd.nlay; k++) {
                if (asmasked && actnumsv_(i, j, k) == 0) {
                    xmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    ymid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    zmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                auto crn = grid3d::get_cell_corners_from_ijk(grd, i, j, k);

                xmid_(i, j, k) =
                  0.125 *
                  (crn.upper_sw.x + crn.upper_se.x + crn.upper_nw.x + crn.upper_ne.x +
                   crn.lower_sw.x + crn.lower_se.x + crn.lower_nw.x + crn.lower_ne.x);
                ymid_(i, j, k) =
                  0.125 *
                  (crn.upper_sw.y + crn.upper_se.y + crn.upper_nw.y + crn.upper_ne.y +
                   crn.lower_sw.y + crn.lower_se.y + crn.lower_nw.y + crn.lower_ne.y);
                zmid_(i, j, k) =
                  0.125 *
                  (crn.upper_sw.z + crn.upper_se.z + crn.upper_nw.z + crn.upper_ne.z +
                   crn.lower_sw.z + crn.lower_se.z + crn.lower_nw.z + crn.lower_ne.z);
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
 * @param grd Grid struct
 * @param ffl Free fluid level per cell
 * @param option 1: Use cell centers, 2 use cell corners
 * @return 3 arrays, top, bot, mid; all delta heights above ffl
 */

std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
get_height_above_ffl(const Grid &grd,
                     const py::array_t<float> &ffl,
                     const size_t option)
{
    pybind11::array_t<double> htop({ grd.ncol, grd.nrow, grd.nlay });
    pybind11::array_t<double> hbot({ grd.ncol, grd.nrow, grd.nlay });
    pybind11::array_t<double> hmid({ grd.ncol, grd.nrow, grd.nlay });
    auto htop_ = htop.mutable_data();
    auto hbot_ = hbot.mutable_data();
    auto hmid_ = hmid.mutable_data();
    auto actnumsv_ = grd.actnumsv.data();
    auto ffl_ = ffl.data();

    for (size_t i = 0; i < grd.ncol; i++) {
        for (size_t j = 0; j < grd.nrow; j++) {
            for (size_t k = 0; k < grd.nlay; k++) {
                size_t idx = i * grd.nrow * grd.nlay + j * grd.nlay + k;
                if (actnumsv_[idx] == 0) {
                    htop_[idx] = numerics::UNDEF_XTGEO;
                    hbot_[idx] = numerics::UNDEF_XTGEO;
                    hmid_[idx] = numerics::UNDEF_XTGEO;
                    continue;
                }
                auto corners = grid3d::get_cell_corners_from_ijk(grd, i, j, k);
                if (option == 1) {
                    htop_[idx] =
                      ffl_[idx] - 0.25 * (corners.upper_sw.z + corners.upper_se.z +
                                          corners.upper_nw.z + corners.upper_ne.z);
                    hbot_[idx] =
                      ffl_[idx] - 0.25 * (corners.lower_sw.z + corners.lower_se.z +
                                          corners.lower_nw.z + corners.lower_ne.z);
                } else if (option == 2) {
                    double upper = corners.upper_sw.z;
                    upper = std::min(upper, corners.upper_se.z);
                    upper = std::min(upper, corners.upper_nw.z);
                    upper = std::min(upper, corners.upper_ne.z);
                    htop_[idx] = ffl_[idx] - upper;

                    double lower = corners.lower_sw.z;
                    lower = std::max(lower, corners.lower_se.z);
                    lower = std::max(lower, corners.lower_nw.z);
                    lower = std::max(lower, corners.lower_ne.z);
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

/* Compute a 3D grid from a cube specification


*/
std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
create_grid_from_cube(const cube::Cube &cube,
                      const bool use_cell_center,
                      const int flip)
{

    // logging here, more for demonstration than anything else
    auto &logger =
      xtgeo::logging::LoggerManager::get("xtgeo.grid3d.create_grid_from_cube");

    logger.debug("Creating grid from cube with dimensions {} {} {} and cube "
                 "xori/yori/zori: {:6.2f} {:6.2f} {:6.2f}",
                 cube.ncol, cube.nrow, cube.nlay, cube.xori, cube.yori, cube.zori);

    // Define the shape of the arrays
    std::vector<size_t> coordsv_shape = { cube.ncol + 1, cube.nrow + 1, 6 };
    std::vector<size_t> zcornsv_shape = { cube.ncol + 1, cube.nrow + 1, cube.nlay + 1,
                                          4 };
    std::vector<size_t> actnumsv_shape = { cube.ncol, cube.nrow, cube.nlay };

    // Create the arrays with the defined shapes
    py::array_t<double> coordsv(coordsv_shape);
    py::array_t<float> zcornsv(zcornsv_shape);
    py::array_t<int8_t> actnumsv(actnumsv_shape);

    auto coordsv_ = coordsv.mutable_data();
    auto zcornsv_ = zcornsv.mutable_data();
    auto actnumsv_ = actnumsv.mutable_data();

    // fill actnum with 1
    std::fill(actnumsv_, actnumsv_ + cube.ncol * cube.nrow * cube.nlay, 1);

    double apply_xori = cube.xori;
    double apply_yori = cube.yori;
    double apply_zori = cube.zori;
    // if we are using cell center, we need to adjust the origin
    if (use_cell_center) {
        auto res = geometry::find_rect_corners_from_center(
          cube.xori, cube.yori, cube.xinc, cube.yinc, cube.rotation);
        if (flip == 1) {
            apply_xori = res[6];
            apply_yori = res[7];
        } else {
            apply_xori = res[0];
            apply_yori = res[1];
        }
        apply_zori = cube.zori - 0.5 * cube.zinc;
    }

    // initialize a temporary RegularSurface
    xtgeo::regsurf::RegularSurface rsurf{ cube.ncol,    cube.nrow, apply_xori,
                                          apply_yori,   cube.xinc, cube.yinc,
                                          cube.rotation };

    // coordinates
    size_t ibc = 0;
    for (size_t i = 0; i < cube.ncol + 1; i++) {
        for (size_t j = 0; j < cube.nrow + 1; j++) {
            xyz::Point p = xtgeo::regsurf::get_xy_from_ij(rsurf, i, j, flip);
            coordsv_[ibc++] = p.x;
            coordsv_[ibc++] = p.y;
            coordsv_[ibc++] = apply_zori;
            coordsv_[ibc++] = p.x;
            coordsv_[ibc++] = p.y;
            coordsv_[ibc++] = apply_zori + cube.zinc * (cube.nlay + 1);
        }
    }

    // zcorners
    size_t ibz = 0;
    for (size_t i = 0; i < cube.ncol + 1; i++) {
        for (size_t j = 0; j < cube.nrow + 1; j++) {
            float zlevel = apply_zori;
            for (size_t k = 0; k < cube.nlay + 1; k++) {
                for (size_t nn = 0; nn < 4; nn++)
                    zcornsv_[ibz++] = zlevel;

                zlevel += cube.zinc;
            }
        }
    }

    return std::make_tuple(coordsv, zcornsv, actnumsv);
}

std::tuple<py::array_t<float>, py::array_t<int8_t>>
refine_vertically(const Grid &grid_cpp, const py::array_t<uint8_t> refinement_per_layer)
{
    auto actnumsv_ = grid_cpp.actnumsv.unchecked<3>();
    auto zcornsv_ = grid_cpp.zcornsv.unchecked<4>();
    auto refinement_data = refinement_per_layer.unchecked<1>();

    // logging here, more for demonstration than anything else
    auto &logger = xtgeo::logging::LoggerManager::get("refine_vertically");

    size_t total_refinement = 0;

    for (size_t i = 0; i < refinement_data.shape(0); i++) {
        total_refinement += refinement_data(i);
    }

    // Shape
    std::vector<size_t> zcorn_shape = { grid_cpp.ncol + 1, grid_cpp.nrow + 1,
                                        total_refinement + 1, 4 };
    py::array_t<float> zcornref(zcorn_shape);
    py::array_t<int8_t> actnumref({ grid_cpp.ncol, grid_cpp.nrow, total_refinement });
    auto zcornref_ = zcornref.mutable_unchecked<4>();
    auto actnumref_ = actnumref.mutable_unchecked<3>();

    logger.debug("Refine vertically from {} layers to {} layers", grid_cpp.nlay,
                 total_refinement);

    for (size_t i = 0; i <= grid_cpp.ncol; i++)
        for (size_t j = 0; j <= grid_cpp.nrow; j++) {
            size_t kk = 0; /* refined grid K counter */
            for (size_t k = 0; k < grid_cpp.nlay; k++) {
                int rfactor = refinement_data[k];
                /* look at each pillar in each cell, find top */
                /* and bottom, and divide */
                for (size_t ic = 0; ic < 4; ic++) {
                    double ztop = zcornsv_(i, j, k, ic);
                    double zbot = zcornsv_(i, j, k + 1, ic);

                    /* now divide and assign to new zcorn for refined: */
                    double rdz = (zbot - ztop) / (double)rfactor;

                    if (rdz < -1 * FLOATEPS) {
                        logger.error("STOP! Negative cell thickness found at %d %d %d",
                                     i + 1, j + 1, k + 1);
                        throw std::runtime_error(
                          "Negative cell thickness found during vertical refinement");
                    }
                    /* now assign corners for the refined grid: */
                    for (size_t kr = 0; kr < rfactor; kr++) {
                        if (i < grid_cpp.ncol && j < grid_cpp.nrow)
                            actnumref_(i, j, kk + kr) = actnumsv_(i, j, k);

                        zcornref_(i, j, kk + kr, ic) = ztop + kr * rdz;
                        zcornref_(i, j, kk + kr + 1, ic) = ztop + (kr + 1) * rdz;
                    }
                }
                kk = kk + rfactor;
            }
        }
    return std::make_tuple(zcornref, actnumref);
}

}  // namespace xtgeo::grid3d
