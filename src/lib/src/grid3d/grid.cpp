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

#ifdef XTGEO_USE_OPENMP
#include <omp.h>
#endif

#define FLOATEPS 1.0e-05

namespace py = pybind11;

namespace xtgeo::grid3d {

// =====================================================================================
// Currently private member functions
// =====================================================================================
void
Grid::impl_compute_cell_corners() const
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("Grid::impl_compute_cell_corners");
    logger.debug("Grid instance ID: {}", static_cast<const void *>(this));

    logger.debug("Computing cell corners for grid ({}, {}, {})", m_ncol, m_nrow,
                 m_nlay);
    m_cell_corners_cache.resize(m_ncol * m_nrow * m_nlay);
    logger.debug("Cell corners cache size: {}", m_cell_corners_cache.size());

    // clang-format off
    #ifdef XTGEO_USE_OPENMP
      #pragma omp parallel for collapse(3)
    #endif
    // clang-format on
    for (int i = 0; i < static_cast<int>(m_ncol); ++i) {  // int due to MSVC OpenMP..
        for (int j = 0; j < static_cast<int>(m_nrow); ++j) {
            for (int k = 0; k < static_cast<int>(m_nlay); ++k) {
                size_t idx = i * m_nrow * m_nlay + j * m_nlay + k;
                m_cell_corners_cache[idx] = get_cell_corners_from_ijk(*this, i, j, k);
            }
        }
    }
    m_cell_corners_computed = true;
    logger.debug("Cell corners computed for grid ({}, {}, {})", m_ncol, m_nrow, m_nlay);
}

void
Grid::impl_compute_bounding_box() const
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("Grid::impl_compute_bounding_box");
    logger.debug("Computing bbox for grid ({}, {}, {})", m_ncol, m_nrow, m_nlay);
    logger.debug("Grid instance ID: {}", static_cast<const void *>(this));

    if (!m_cell_corners_computed) {
        logger.debug("Cell corners not computed, calling compute_cell_corners()");
        impl_compute_cell_corners();
    }

    // Initialize with first point
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();
    double max_z = std::numeric_limits<double>::lowest();

    // Process only first and last layers
    for (size_t i = 0; i < m_ncol; ++i) {
        for (size_t j = 0; j < m_nrow; ++j) {
            // Process first layer (k=0)
            size_t idx_first = i * m_nrow * m_nlay + j * m_nlay + 0;
            const auto &cell_first = m_cell_corners_cache[idx_first];

            const auto points_first = { &cell_first.upper_sw, &cell_first.upper_se,
                                        &cell_first.upper_nw, &cell_first.upper_ne,
                                        &cell_first.lower_sw, &cell_first.lower_se,
                                        &cell_first.lower_nw, &cell_first.lower_ne };

            for (const auto &point : points_first) {
                min_x = std::min(min_x, point->x());
                min_y = std::min(min_y, point->y());
                min_z = std::min(min_z, point->z());
                max_x = std::max(max_x, point->x());
                max_y = std::max(max_y, point->y());
                max_z = std::max(max_z, point->z());
            }

            // Process last layer (k=m_nlay-1)
            if (m_nlay > 1) {  // Only if there's more than one layer
                size_t idx_last = i * m_nrow * m_nlay + j * m_nlay + (m_nlay - 1);
                const auto &cell_last = m_cell_corners_cache[idx_last];

                const auto points_last = { &cell_last.upper_sw, &cell_last.upper_se,
                                           &cell_last.upper_nw, &cell_last.upper_ne,
                                           &cell_last.lower_sw, &cell_last.lower_se,
                                           &cell_last.lower_nw, &cell_last.lower_ne };

                for (const auto &point : points_last) {
                    min_x = std::min(min_x, point->x());
                    min_y = std::min(min_y, point->y());
                    min_z = std::min(min_z, point->z());
                    max_x = std::max(max_x, point->x());
                    max_y = std::max(max_y, point->y());
                    max_z = std::max(max_z, point->z());
                }
            }
        }
    }

    m_min_point = xyz::Point(min_x, min_y, min_z);
    m_max_point = xyz::Point(max_x, max_y, max_z);
    logger.debug("Bounding box computed: min ({}, {}, {}), max ({}, {}, {})",
                 m_min_point.x(), m_min_point.y(), m_min_point.z(), m_max_point.x(),
                 m_max_point.y(), m_max_point.z());
    m_bounding_box_computed = true;
}
// =====================================================================================
// Public functions
// =====================================================================================

/*
 * Compute bulk volume of cells in a grid. Tests show that this is very close
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
    py::array_t<double> cellvols({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    auto cellvols_ = cellvols.mutable_data();
    auto actnumsv_ = grd.get_actnumsv().data();

    auto &logger = xtgeo::logging::LoggerManager::get("get_cell_volumes");
    logger.debug("Computing cell volumes for grid ({}, {}, {})", grd.get_ncol(),
                 grd.get_nrow(), grd.get_nlay());

    size_t ncol = grd.get_ncol();
    size_t nrow = grd.get_nrow();
    size_t nlay = grd.get_nlay();

    // ensure precomputing the cache outside the OMP loop to avoid race conditions
    grd.get_cell_corners_cache();

    logger.debug("Start loop computing cell volumes}");
    // clang-format off
    #ifdef XTGEO_USE_OPENMP
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
                auto crn = grd.get_cell_corners_cache()[i * nrow * nlay + j * nlay + k];
                cellvols_[idx] = geometry::hexahedron_volume(crn, precision);
            }
        }
    }
    logger.debug("Cell volumes computed for grid ({}, {}, {})", grd.get_ncol(),
                 grd.get_nrow(), grd.get_nlay());
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
    pybind11::array_t<double> xmid({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    pybind11::array_t<double> ymid({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    pybind11::array_t<double> zmid({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    auto xmid_ = xmid.mutable_unchecked<3>();
    auto ymid_ = ymid.mutable_unchecked<3>();
    auto zmid_ = zmid.mutable_unchecked<3>();
    auto actnumsv_ = grd.get_actnumsv().unchecked<3>();

    for (size_t i = 0; i < grd.get_ncol(); i++) {
        for (size_t j = 0; j < grd.get_nrow(); j++) {
            for (size_t k = 0; k < grd.get_nlay(); k++) {
                if (asmasked && actnumsv_(i, j, k) == 0) {
                    xmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    ymid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    zmid_(i, j, k) = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                auto crn =
                  grd.get_cell_corners_cache()[i * grd.get_nrow() * grd.get_nlay() +
                                               j * grd.get_nlay() + k];

                xmid_(i, j, k) =
                  0.125 * (crn.upper_sw.x() + crn.upper_se.x() + crn.upper_nw.x() +
                           crn.upper_ne.x() + crn.lower_sw.x() + crn.lower_se.x() +
                           crn.lower_nw.x() + crn.lower_ne.x());
                ymid_(i, j, k) =
                  0.125 * (crn.upper_sw.y() + crn.upper_se.y() + crn.upper_nw.y() +
                           crn.upper_ne.y() + crn.lower_sw.y() + crn.lower_se.y() +
                           crn.lower_nw.y() + crn.lower_ne.y());
                zmid_(i, j, k) =
                  0.125 * (crn.upper_sw.z() + crn.upper_se.z() + crn.upper_nw.z() +
                           crn.upper_ne.z() + crn.lower_sw.z() + crn.lower_se.z() +
                           crn.lower_nw.z() + crn.lower_ne.z());
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
    pybind11::array_t<double> htop({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    pybind11::array_t<double> hbot({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    pybind11::array_t<double> hmid({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    auto htop_ = htop.mutable_data();
    auto hbot_ = hbot.mutable_data();
    auto hmid_ = hmid.mutable_data();
    auto actnumsv_ = grd.get_actnumsv().data();
    auto ffl_ = ffl.data();

    for (size_t i = 0; i < grd.get_ncol(); i++) {
        for (size_t j = 0; j < grd.get_nrow(); j++) {
            for (size_t k = 0; k < grd.get_nlay(); k++) {
                size_t idx =
                  i * grd.get_nrow() * grd.get_nlay() + j * grd.get_nlay() + k;
                if (actnumsv_[idx] == 0) {
                    htop_[idx] = numerics::UNDEF_XTGEO;
                    hbot_[idx] = numerics::UNDEF_XTGEO;
                    hmid_[idx] = numerics::UNDEF_XTGEO;
                    continue;
                }
                auto corners =
                  grd.get_cell_corners_cache()[i * grd.get_nrow() * grd.get_nlay() +
                                               j * grd.get_nlay() + k];
                if (option == 1) {
                    htop_[idx] =
                      ffl_[idx] - 0.25 * (corners.upper_sw.z() + corners.upper_se.z() +
                                          corners.upper_nw.z() + corners.upper_ne.z());
                    hbot_[idx] =
                      ffl_[idx] - 0.25 * (corners.lower_sw.z() + corners.lower_se.z() +
                                          corners.lower_nw.z() + corners.lower_ne.z());
                } else if (option == 2) {
                    double upper = corners.upper_sw.z();
                    upper = std::min(upper, corners.upper_se.z());
                    upper = std::min(upper, corners.upper_nw.z());
                    upper = std::min(upper, corners.upper_ne.z());
                    htop_[idx] = ffl_[idx] - upper;

                    double lower = corners.lower_sw.z();
                    lower = std::max(lower, corners.lower_se.z());
                    lower = std::max(lower, corners.lower_nw.z());
                    lower = std::max(lower, corners.lower_ne.z());
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
            coordsv_[ibc++] = p.x();
            coordsv_[ibc++] = p.y();
            coordsv_[ibc++] = apply_zori;
            coordsv_[ibc++] = p.x();
            coordsv_[ibc++] = p.y();
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
refine_vertically(const Grid &grd, const py::array_t<uint8_t> refinement_per_layer)
{
    auto actnumsv_ = grd.get_actnumsv().unchecked<3>();
    auto zcornsv_ = grd.get_zcornsv().unchecked<4>();
    auto refinement_data = refinement_per_layer.unchecked<1>();

    // logging here, more for demonstration than anything else
    auto &logger = xtgeo::logging::LoggerManager::get("refine_vertically");

    size_t total_refinement = 0;

    for (size_t i = 0; i < refinement_data.shape(0); i++) {
        total_refinement += refinement_data(i);
    }

    // Shape
    std::vector<size_t> zcorn_shape = { grd.get_ncol() + 1, grd.get_nrow() + 1,
                                        total_refinement + 1, 4 };
    py::array_t<float> zcornref(zcorn_shape);
    py::array_t<int8_t> actnumref({ grd.get_ncol(), grd.get_nrow(), total_refinement });
    auto zcornref_ = zcornref.mutable_unchecked<4>();
    auto actnumref_ = actnumref.mutable_unchecked<3>();

    logger.debug("Refine vertically from {} layers to {} layers", grd.get_nlay(),
                 total_refinement);

    for (size_t i = 0; i <= grd.get_ncol(); i++)
        for (size_t j = 0; j <= grd.get_nrow(); j++) {
            size_t kk = 0; /* refined grid K counter */
            for (size_t k = 0; k < grd.get_nlay(); k++) {
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
                        if (i < grd.get_ncol() && j < grd.get_nrow())
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

std::tuple<py::array_t<double>, py::array_t<float>, py::array_t<int8_t>>
refine_columns(const Grid &grid_cpp, const py::array_t<uint8_t> refinement)
{
    auto &logger = xtgeo::logging::LoggerManager::get("refine_columns");
    auto actnumsv_ = grid_cpp.get_actnumsv().unchecked<3>();
    auto coordsv_ = grid_cpp.get_coordsv().unchecked<3>();
    auto zcornsv_ = grid_cpp.get_zcornsv().unchecked<4>();

    const size_t ncol = grid_cpp.get_ncol();
    const size_t nrow = grid_cpp.get_nrow();
    const size_t nlay = grid_cpp.get_nlay();

    const auto refinement_data = refinement.data();
    const size_t ncol_ref =
      std::accumulate(refinement_data, refinement_data + ncol, size_t(0));

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
                        coordref_(ii + ir, j, 3 * ic) = pt.x();
                        coordref_(ii + ir, j, 3 * ic + 1) = pt.y();
                        coordref_(ii + ir, j, 3 * ic + 2) = pt.z();
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
    auto actnumsv_ = grid_cpp.get_actnumsv().unchecked<3>();
    auto coordsv_ = grid_cpp.get_coordsv().unchecked<3>();
    auto zcornsv_ = grid_cpp.get_zcornsv().unchecked<4>();

    const size_t ncol = grid_cpp.get_ncol();
    const size_t nrow = grid_cpp.get_nrow();
    const size_t nlay = grid_cpp.get_nlay();

    const auto refinement_data = refinement.data();
    const size_t nrow_ref =
      std::accumulate(refinement_data, refinement_data + nrow, size_t(0));

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
                        coordref_(i, jj + jr, 3 * ic) = pt.x();
                        coordref_(i, jj + jr, 3 * ic + 1) = pt.y();
                        coordref_(i, jj + jr, 3 * ic + 2) = pt.z();
                    }
                }
            }

            // Calculate ZCORN
            for (size_t k = 0; k < nlay; k++) {
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
