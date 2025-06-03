#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

#ifdef __linux__
#include <omp.h>
#endif

#define FLOATEPS 1.0e-05

namespace py = pybind11;

namespace xtgeo::grid3d {

void
Grid::compute_cell_corners()
{
    auto &logger = xtgeo::logging::LoggerManager::get("Grid::compute_cell_corners");
    cell_corners_cache.resize(ncol * nrow * nlay);
    logger.info("Computing cell corners for grid ({}, {}, {})", ncol, nrow, nlay);

    // clang-format off
    #ifdef __linux__
      #pragma omp parallel for collapse(3)
    #endif
    // clang-format on
    for (size_t i = 0; i < ncol; ++i) {
        for (size_t j = 0; j < nrow; ++j) {
            for (size_t k = 0; k < nlay; ++k) {
                cell_corners_cache[i * nrow * nlay + j * nlay + k] =
                  get_cell_corners_from_ijk(*this, i, j, k);
            }
        }
    }
    logger.info("Computing cell corners for grid - DONE", ncol, nrow, nlay);
}

void
Grid::ensure_cell_corners_cache() const
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("Grid::ensure_cell_corners_cache");
    if (cell_corners_cache.size() != ncol * nrow * nlay) {
        logger.info("Cell corners cache is empty, computing cell corners");
        logger.info("Current size: {}", cell_corners_cache.size());

        // const_cast is needed because we're modifying a mutable member in a const
        // method
        const_cast<Grid *>(this)->compute_cell_corners();
    }
}
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
get_cell_volumes(const Grid &grd,
                 geometry::HexVolumePrecision precision,
                 const bool asmasked)
{
    py::array_t<double> cellvols({ grd.ncol, grd.nrow, grd.nlay });
    auto cellvols_ = cellvols.mutable_data();
    auto actnumsv_ = grd.actnumsv.data();

    size_t ncol = grd.ncol;
    size_t nrow = grd.nrow;
    size_t nlay = grd.nlay;

    grd.ensure_cell_corners_cache();

    // clang-format off
    #ifdef __linux__
      #pragma omp parallel for collapse(3)
    #endif
    // clang-format on

    for (auto i = 0; i < ncol; i++) {
        for (auto j = 0; j < nrow; j++) {
            for (auto k = 0; k < nlay; k++) {
                auto idx = i * nrow * nlay + j * nlay + k;
                if (asmasked && actnumsv_[idx] == 0) {
                    cellvols_[idx] = numerics::UNDEF_XTGEO;
                    continue;
                }
                auto crn = grd.cell_corners_cache[i * nrow * nlay + j * nlay + k];
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

    grd.ensure_cell_corners_cache();

    for (size_t i = 0; i < grd.ncol; i++) {
        for (size_t j = 0; j < grd.nrow; j++) {
            for (size_t k = 0; k < grd.nlay; k++) {
                if (asmasked && actnumsv_(i, j, k) == 0) {
                    xmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    ymid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    zmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                auto crn =
                  grd.cell_corners_cache[i * grd.nrow * grd.nlay + j * grd.nlay + k];

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
 * Compute cell height above ffl (free fluid level), as input to water saturation.
 * Will return hbot, htop, hmid (bottom of cell, top of cell, midpoint), but compute
 * method depends on option: 1: cell center above ffl, 2: cell corners above ffl
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

    grd.ensure_cell_corners_cache();
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
                auto corners =
                  grd.cell_corners_cache[i * grd.nrow * grd.nlay + j * grd.nlay + k];
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

/** @brief Get bounding box for 3D grid
 * @param Grid input grid
 * @return A tuple of two points, minimum values and maximum values
 */

std::tuple<xyz::Point, xyz::Point>
get_bounding_box(const Grid &grid)
{
    // Initialize min and max values
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::lowest();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::lowest();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::lowest();

    Grid onelayer_grid = extract_onelayer_grid(grid);
    onelayer_grid.ensure_cell_corners_cache();

    // Step 2: Loop through all cells and find min/max values
    for (size_t i = 0; i < grid.ncol; i++) {
        for (size_t j = 0; j < grid.nrow; j++) {
            // Get the corners of the cell
            CellCorners corners =
              onelayer_grid
                .cell_corners_cache[i * onelayer_grid.nrow * onelayer_grid.nlay +
                                    j * onelayer_grid.nlay + 0];

            // Loop through the 8 corners to find min/max values
            std::array<xyz::Point, 8> corner_points = {
                corners.upper_sw, corners.upper_se, corners.upper_nw, corners.upper_ne,
                corners.lower_sw, corners.lower_se, corners.lower_nw, corners.lower_ne
            };

            for (const auto &point : corner_points) {
                xmin = std::min(xmin, point.x);
                xmax = std::max(xmax, point.x);
                ymin = std::min(ymin, point.y);
                ymax = std::max(ymax, point.y);
                zmin = std::min(zmin, point.z);
                zmax = std::max(zmax, point.z);
            }
        }
    }

    // Create points to return
    xyz::Point minv = { xmin, ymin, zmin };
    xyz::Point maxv = { xmax, ymax, zmax };

    return std::make_tuple(minv, maxv);
}

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_columns(const Grid &grid_cpp, const py::array_t<uint8_t> refinement)
{
    auto &logger = xtgeo::logging::LoggerManager::get("refine_columns");
    auto actnumsv_ = grid_cpp.actnumsv.unchecked<3>();
    auto coordsv_ = grid_cpp.coordsv.unchecked<3>();
    auto zcornsv_ = grid_cpp.zcornsv.unchecked<4>();

    const size_t ncol = grid_cpp.ncol;
    const size_t nrow = grid_cpp.nrow;
    const size_t nlay = grid_cpp.nlay;

    const auto refinement_data = refinement.data();
    const size_t ncol_ref =
      std::accumulate(refinement_data, refinement_data + grid_cpp.ncol, size_t(0));

    py::array_t<double> coordref(std::vector<size_t>{ ncol_ref + 1, nrow + 1, 6 });
    py::array_t<float> zcornref(
      std::vector<size_t>{ ncol_ref + 1, nrow + 1, nlay + 1, 4 });
    py::array_t<int8_t> actnumref({ ncol_ref, nrow, nlay });

    auto coordref_ = coordref.mutable_unchecked<3>();
    auto zcornref_ = zcornref.mutable_unchecked<4>();
    auto actnumref_ = actnumref.mutable_unchecked<3>();
    logger.debug("Refine grid from {}x{}x{} to {}x{}x{} layers", ncol, nrow, nlay,
                 ncol_ref, nrow, nlay);

    size_t ii = 0;  // Refine grid i counter
    for (size_t i = 0; i < ncol; i++) {
        size_t ref_factor = refinement_data[i];
        for (size_t j = 0; j < nrow + 1; j++) {
            // Calculate COORD
            // Interpolate coordinates (top and bottom)
            for (size_t ic = 0; ic < 2; ic++) {
                double x1 = coordsv_(i, j, 3 * ic);
                double y1 = coordsv_(i, j, 3 * ic + 1);
                double z1 = coordsv_(i, j, 3 * ic + 2);

                double x2 = coordsv_(i + 1, j, 3 * ic);
                double y2 = coordsv_(i + 1, j, 3 * ic + 1);
                double z2 = coordsv_(i + 1, j, 3 * ic + 2);

                for (int ir = 0; ir <= ref_factor; ir++) {
                    if (i > 0 && ir == 0)
                        continue;  // Skip re-calculate previous point

                    if (ir == 0) {
                        coordref_(ii + ir, j, 3 * ic) = x1;
                        coordref_(ii + ir, j, 3 * ic + 1) = y1;
                        coordref_(ii + ir, j, 3 * ic + 2) = z1;
                    } else if (ir == ref_factor) {
                        coordref_(ii + ir, j, 3 * ic) = x2;
                        coordref_(ii + ir, j, 3 * ic + 1) = y2;
                        coordref_(ii + ir, j, 3 * ic + 2) = z2;
                    } else {
                        auto pt = numerics::lerp3d(x1, y1, z1, x2, y2, z2,
                                                   double(ir) / ref_factor);
                        coordref_(ii + ir, j, 3 * ic) = pt.x;
                        coordref_(ii + ir, j, 3 * ic + 1) = pt.y;
                        coordref_(ii + ir, j, 3 * ic + 2) = pt.z;
                    }
                }
            }

            // Calculate ZCORN
            for (size_t k = 0; k < nlay + 1; k++) {
                double pillar[4] = { // South pilar
                                     zcornsv_(i, j, k, 1), zcornsv_(i + 1, j, k, 0),
                                     // North pillar
                                     zcornsv_(i, j, k, 3), zcornsv_(i + 1, j, k, 2)
                };
                for (int ir = 0; ir <= ref_factor; ir++) {
                    if (i > 0 && ir == 0)
                        continue;  // Skip re-calculate previous point

                    if (ir == 0) {
                        for (size_t ic = 0; ic < 4; ic++)
                            zcornref_(ii + ir, j, k, ic) = zcornsv_(i, j, k, ic);
                    } else {

                        double fr = double(ir) / ref_factor;
                        double z_south =
                          geometry::generic::lerp(pillar[0], pillar[1], fr);
                        double z_north =
                          geometry::generic::lerp(pillar[2], pillar[3], fr);
                        if (ir == ref_factor) {
                            zcornref_(ii + ir, j, k, 0) = z_south;
                            zcornref_(ii + ir, j, k, 1) = zcornsv_(i + 1, j, k, 1);
                            zcornref_(ii + ir, j, k, 2) = z_north;
                            zcornref_(ii + ir, j, k, 3) = zcornsv_(i + 1, j, k, 3);
                        } else {
                            zcornref_(ii + ir, j, k, 0) = z_south;
                            zcornref_(ii + ir, j, k, 1) = z_south;
                            zcornref_(ii + ir, j, k, 2) = z_north;
                            zcornref_(ii + ir, j, k, 3) = z_north;
                        }
                    }
                }
                if (j < nrow && k < nlay)
                    // Refine actnum
                    for (int ir = 0; ir < ref_factor; ++ir)
                        actnumref_(ii + ir, j, k) = actnumsv_(i, j, k);
            }
        }
        ii += ref_factor;
    }
    return std::make_tuple(coordref, zcornref, actnumref);
}

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_rows(const Grid &grid_cpp, const py::array_t<uint8_t> refinement)
{
    auto &logger = xtgeo::logging::LoggerManager::get("refine_rows");
    auto actnumsv_ = grid_cpp.actnumsv.unchecked<3>();
    auto coordsv_ = grid_cpp.coordsv.unchecked<3>();
    auto zcornsv_ = grid_cpp.zcornsv.unchecked<4>();

    const size_t ncol = grid_cpp.ncol;
    const size_t nrow = grid_cpp.nrow;
    const size_t nlay = grid_cpp.nlay;

    const auto refinement_data = refinement.data();
    const size_t nrow_ref =
      std::accumulate(refinement_data, refinement_data + grid_cpp.nrow, size_t(0));

    py::array_t<double> coordref(std::vector<size_t>{ ncol + 1, nrow_ref + 1, 6 });
    py::array_t<float> zcornref(
      std::vector<size_t>{ ncol + 1, nrow_ref + 1, nlay + 1, 4 });
    py::array_t<int8_t> actnumref({ ncol, nrow_ref, nlay });

    auto coordref_ = coordref.mutable_unchecked<3>();
    auto zcornref_ = zcornref.mutable_unchecked<4>();
    auto actnumref_ = actnumref.mutable_unchecked<3>();
    logger.debug("Refine grid from {}x{}x{} to {}x{}x{} layers", ncol, nrow, nlay, ncol,
                 nrow_ref, nlay);

    for (size_t i = 0; i < ncol + 1; i++) {
        size_t jj = 0;  // Refine grid j counter
        for (size_t j = 0; j < nrow; j++) {
            size_t ref_factor = refinement_data[j];
            // Calculate COORD
            for (size_t ic = 0; ic < 2; ic++) {
                double x1 = coordsv_(i, j, 3 * ic);
                double y1 = coordsv_(i, j, 3 * ic + 1);
                double z1 = coordsv_(i, j, 3 * ic + 2);

                double x2 = coordsv_(i, j + 1, 3 * ic);
                double y2 = coordsv_(i, j + 1, 3 * ic + 1);
                double z2 = coordsv_(i, j + 1, 3 * ic + 2);

                for (size_t jr = 0; jr <= ref_factor; jr++) {
                    if (j > 0 && jr == 0)
                        continue;  // Skip re-calculate previous point
                    if (jr == 0) {
                        coordref_(i, jj + jr, 3 * ic) = x1;
                        coordref_(i, jj + jr, 3 * ic + 1) = y1;
                        coordref_(i, jj + jr, 3 * ic + 2) = z1;
                    } else if (jr == ref_factor) {
                        coordref_(i, jj + jr, 3 * ic) = x2;
                        coordref_(i, jj + jr, 3 * ic + 1) = y2;
                        coordref_(i, jj + jr, 3 * ic + 2) = z2;
                    } else {
                        auto pt = numerics::lerp3d(x1, y1, z1, x2, y2, z2,
                                                   double(jr) / ref_factor);
                        coordref_(i, jj + jr, 3 * ic) = pt.x;
                        coordref_(i, jj + jr, 3 * ic + 1) = pt.y;
                        coordref_(i, jj + jr, 3 * ic + 2) = pt.z;
                    }
                }
            }

            // Calculate ZCORN
            for (size_t k = 0; k < grid_cpp.nlay; k++) {
                for (size_t idx_vert = 0; idx_vert < 2;
                     idx_vert++)  // Vertical loop, Top - Bottom
                {
                    double pillar[4] = { // West pilar
                                         zcornsv_(i, j, k + idx_vert, 2),
                                         zcornsv_(i, j + 1, k + idx_vert, 0),
                                         // East pillar
                                         zcornsv_(i, j, k + idx_vert, 3),
                                         zcornsv_(i, j + 1, k + idx_vert, 1)
                    };
                    for (int jr = 0; jr <= ref_factor; jr++) {
                        if (j > 0 && jr == 0)
                            continue;  // Skip re-calculate previous point
                        if (jr == 0) {
                            for (size_t ic = 0; ic < 4; ic++)
                                zcornref_(i, jj + jr, k + idx_vert, ic) =
                                  zcornsv_(i, j, k + idx_vert, ic);
                        } else if (jr == ref_factor) {
                            for (size_t ic = 0; ic < 4; ic++)
                                zcornref_(i, jj + jr, k + idx_vert, ic) =
                                  zcornsv_(i, j + 1, k + idx_vert, ic);
                        } else {
                            double fr = double(jr) / ref_factor;
                            double z_west =
                              geometry::generic::lerp(pillar[0], pillar[1], fr);
                            double z_east =
                              geometry::generic::lerp(pillar[2], pillar[3], fr);
                            zcornref_(i, jj + jr, k + idx_vert, 0) =
                              zcornref_(i, jj + jr, k + idx_vert, 2) = z_west;
                            zcornref_(i, jj + jr, k + idx_vert, 1) =
                              zcornref_(i, jj + jr, k + idx_vert, 3) = z_east;
                        }
                    }
                }
                if (i < ncol)
                    // Refine actnum
                    for (int jr = 0; jr < ref_factor; ++jr)
                        actnumref_(i, jj + jr, k) = actnumsv_(i, j, k);
            }
            jj += ref_factor;
        }
    }
    return std::make_tuple(coordref, zcornref, actnumref);
}

}  // namespace xtgeo::grid3d
