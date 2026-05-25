// File: grid_surf_oper.cpp focus on grid or gridproperty that are associated with
// and/or modified by surface(s) input. This file is part of the xtgeo library.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <tuple>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/regsurf.hpp>
#include <xtgeo/xtgeo.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

/*
 * Create a parameter that is 1 if the cell center is between the top and bottom
 * surfaces
 *
 * @param grd Grid instance
 * @param top The top surface (RegularSurface object)
 * @param bot The bottom surface (RegularSurface object)
 * @return An int array containing 1 if the cell is between the surfaces, 0 otherwise
 */

py::array_t<int8_t>
get_gridprop_value_between_surfaces(const Grid &grd,
                                    const regsurf::RegularSurface &top,
                                    const regsurf::RegularSurface &bot)
{
    pybind11::array_t<int8_t> result(
      { grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });
    auto result_ = result.mutable_unchecked<3>();

    py::array_t<double> xmid, ymid, zmid;
    std::tie(xmid, ymid, zmid) = get_cell_centers(grd, true);

    // Access the np array without bounds checking for optimizing speed
    auto xmid_ = xmid.unchecked<3>();
    auto ymid_ = ymid.unchecked<3>();
    auto zmid_ = zmid.unchecked<3>();

    for (size_t i = 0; i < grd.get_ncol(); i++) {
        for (size_t j = 0; j < grd.get_nrow(); j++) {
            for (size_t k = 0; k < grd.get_nlay(); k++) {
                // for every cell, project the center to the top and bottom surfaces
                double xm = xmid_(i, j, k);
                double ym = ymid_(i, j, k);
                double zm = zmid_(i, j, k);

                // check if zm is NaN which can occur for inactive cells
                if (std ::isnan(zm)) {
                    result_(i, j, k) = 0;
                    continue;
                }

                double top_z = regsurf::get_z_from_xy(top, xm, ym);
                double bot_z = regsurf::get_z_from_xy(bot, xm, ym);

                // check if top_z or bot_z is NaN
                if (std::isnan(top_z) || std::isnan(bot_z)) {
                    result_(i, j, k) = 0;
                    continue;
                }

                if (zm >= top_z && zm <= bot_z) {  // inside
                    result_(i, j, k) = 1;
                } else {
                    result_(i, j, k) = 0;
                }
            }
        }
    }
    return result;
}  // get_gridprop_value_between_surfaces

/*
 * Given a 3D grid and surfaces, construct layers in 3D grid (zcorns) from surface. Note
 * that this grid is based on a "shoebox grid", which has vertical pillars in the input!
 *
 * @param grd Grid instance, which already has the correct number of layers
 * @param rsurfs The RegularSurface objects (array/list input), sorted from top
 * to bottom
 * @param tolerance The tolerance for the interpolation
 * @return updated zcorn and actnum values in the grid
 */

std::tuple<py::array_t<float>, py::array_t<int>>
adjust_boxgrid_layers_from_regsurfs(Grid &grd,
                                    const std::vector<regsurf::RegularSurface> &rsurfs,
                                    const double tolerance)
{
    std::vector<size_t> shape = { grd.get_ncol() + 1, grd.get_nrow() + 1,
                                  grd.get_nlay() + 1, 4 };
    // Create the array with the specified shape
    py::array_t<float> zcorn_result(shape);
    py::array_t<int> actnum_result({ grd.get_ncol(), grd.get_nrow(), grd.get_nlay() });

    auto zcorn_result_ = zcorn_result.mutable_unchecked<4>();
    auto actnum_result_ = actnum_result.mutable_unchecked<3>();

    auto coordsv_ = grd.get_coordsv().unchecked<3>();

    for (size_t i = 0; i < actnum_result_.shape(0); ++i) {
        for (size_t j = 0; j < actnum_result_.shape(1); ++j) {
            for (size_t k = 0; k < actnum_result_.shape(2); ++k) {
                actnum_result_(i, j, k) = 1;
            }
        }
    }

    if (rsurfs.size() != grd.get_nlay() + 1) {
        throw std::invalid_argument("Wrong number of input surfaces vs grid layers");
    }

    for (size_t icol = 0; icol < grd.get_ncol() + 1; icol++) {
        for (size_t jrow = 0; jrow < grd.get_nrow() + 1; jrow++) {
            // Get pillar coordinate, just using the top pillar point as the pillars
            // shall be vertical
            double x = coordsv_(icol, jrow, 0);
            double y = coordsv_(icol, jrow, 1);
            size_t ksurf = 0;

            int actnum = 1;

            for (const auto &rsurf : rsurfs) {
                double z = regsurf::get_z_from_xy(rsurf, x, y, tolerance);

                if (std::isnan(z)) {
                    z = 1000;  // Default value; TODO make this better
                    actnum = 0;
                }

                // Update the z values of the cell corners
                for (size_t l = 0; l < 4; l++) {
                    zcorn_result_(icol, jrow, ksurf, l) = z;
                }

                ksurf++;
            }

            // If actnum is 0, then all 4 cells that share this pillar should be
            // inactive for the entire column
            if (actnum == 0) {
                int ia = static_cast<int>(icol);
                int ja = static_cast<int>(jrow);

                for (size_t k = 0; k < grd.get_nlay(); k++) {
                    for (int ii = ia - 1; ii <= ia; ii++) {
                        for (int jj = ja - 1; jj <= ja; jj++) {
                            // Ensure indices are within bounds
                            size_t apply_i =
                              std::clamp(ii, 0, static_cast<int>(grd.get_ncol() - 1));
                            size_t apply_j =
                              std::clamp(jj, 0, static_cast<int>(grd.get_nrow() - 1));

                            // Set actnum_result_ to 0 for the entire column
                            actnum_result_(apply_i, apply_j, k) = 0;
                        }
                    }
                }
            }
        }
    }
    return std::make_tuple(zcorn_result, actnum_result);
}

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

static inline double
avg4(const std::array<double, 4> &a)
{
    return (a[0] + a[1] + a[2] + a[3]) * 0.25;
}

static bool
is_pillar_faulted(const std::vector<std::array<double, 4>> &orig,
                  size_t nkp1,
                  double tol)
{
    return std::any_of(orig.begin(), orig.begin() + static_cast<ptrdiff_t>(nkp1),
                       [tol](const auto &corners) {
                           auto [mn, mx] =
                             std::minmax_element(corners.begin(), corners.end());
                           return (*mx - *mn) > tol;
                       });
}

/*
 * Find pillar parameter t where the pillar intersects a surface.
 * P(t) = P_top + t*(P_bot - P_top), so t∈[0,1] is within the pillar extent.
 * Returns NaN if no intersection is found.
 */
static double
find_pillar_surface_intersection_t(double x_t,
                                   double y_t,
                                   double z_t,
                                   double x_b,
                                   double y_b,
                                   double z_b,
                                   const regsurf::RegularSurface &surf,
                                   double tolerance)
{
    constexpr double eps = 1e-10;
    const double dx = x_b - x_t, dy = y_b - y_t, dz = z_b - z_t;

    if (std::abs(dz) < eps)
        return NaN;

    // Vertical pillar: direct Z lookup
    if (std::abs(dx) < eps && std::abs(dy) < eps) {
        double zs = regsurf::get_z_from_xy(surf, x_t, y_t, tolerance);
        return std::isnan(zs) ? NaN : (zs - z_t) / dz;
    }

    // Compute distance between z_pillar (with extension) and z_surface
    auto g = [&](double t) -> double {
        double zs = regsurf::get_z_from_xy(surf, x_t + t * dx, y_t + t * dy, tolerance);
        return std::isnan(zs) ? NaN : (z_t + t * dz) - zs;
    };

    // Scan t ∈ [-0.5, 1.5] to find where pillar intersect surface (g changes sign)
    constexpr int N = 8;
    constexpr double t0 = -0.5, t1 = 1.5, dt = (t1 - t0) / N;

    double lo_t = 0, lo_g = 0, hi_t = 0, hi_g = 0;
    bool found = false;
    double prev_t = t0, prev_g = g(t0);

    for (int i = 1; i <= N; ++i) {
        double cur_t = t0 + i * dt;
        double cur_g = g(cur_t);
        if (!std::isnan(prev_g) && !std::isnan(cur_g) && prev_g * cur_g <= 0) {
            lo_t = prev_t;
            lo_g = prev_g;
            hi_t = cur_t;
            hi_g = cur_g;
            found = true;
            break;  // take first bracket found
        }
        prev_t = cur_t;
        prev_g = cur_g;
    }
    if (!found)
        return NaN;

    // Bisect to convergence
    for (int i = 0; i < 50; ++i) {
        double mid_t = 0.5 * (lo_t + hi_t);
        double mid_g = g(mid_t);
        if (std::isnan(mid_g))
            return NaN;
        if (std::abs(mid_g) < 1e-8 || (hi_t - lo_t) < 1e-12)
            return mid_t;
        if (lo_g * mid_g <= 0) {
            hi_t = mid_t;
            hi_g = mid_g;
        } else {
            lo_t = mid_t;
            lo_g = mid_g;
        }
    }
    return 0.5 * (lo_t + hi_t);
}

/*
 * Conform grid ZCORN to a set of zone-boundary surfaces (in-place).
 *
 * For each grid pillar the algorithm:
 *   1. Finds the pillar/surface intersection (along the pillar line) for each surface,
 *   2. Preserves per-corner offsets from the pillar average (maintaining fault
 * geometry),
 *   3. Distributes interior layers proportionally within each zone,
 *   4. Enforces monotonicity (top-to-bottom) on the result.
 *
 * COORD is read-only (pillar geometry unchanged). ZCORN is modified in-place.
 *
 * @param grd Grid object (coordsv read-only, zcornsv modified in-place)
 * @param surfaces Surfaces sorted top-to-bottom
 * @param layers_per_zone Number of layers per zone
 * @param skip_faults If true, leave faulted pillars (corner spread > 0.01m) unchanged
 * @param tolerance Surface sampling tolerance
 */
void
conform_grid_to_surfaces(Grid &grd,
                         const std::vector<regsurf::RegularSurface> &surfaces,
                         const std::vector<int> &layers_per_zone,
                         bool skip_faults,
                         double tolerance)
{
    constexpr double threshold = 1e-10;
    constexpr double faults_tolerance = 0.01;

    const size_t n_surfaces = surfaces.size();
    const size_t n_zones = layers_per_zone.size();

    if (n_surfaces < 2)
        throw std::invalid_argument(
          "At least 2 surfaces are required (top and bottom).");
    if (n_surfaces != n_zones + 1)
        throw std::invalid_argument(
          "Number of surfaces must be len(layers_per_zone) + 1.");
    if (std::any_of(layers_per_zone.begin(), layers_per_zone.end(),
                    [](int v) { return v < 1; }))
        throw std::invalid_argument("All values in layers_per_zone must be >= 1.");

    const size_t total_layers =
      std::accumulate(layers_per_zone.begin(), layers_per_zone.end(), size_t{ 0 },
                      [](size_t a, int b) { return a + static_cast<size_t>(b); });

    auto coord = grd.get_coordsv().unchecked<3>();
    py::array_t<float> zcornsv = grd.get_zcornsv();
    auto zcorn = zcornsv.mutable_unchecked<4>();

    const size_t npil_i = static_cast<size_t>(coord.shape(0));
    const size_t npil_j = static_cast<size_t>(coord.shape(1));
    const size_t nkp1 = static_cast<size_t>(zcorn.shape(2));

    if (total_layers + 1 != nkp1)
        throw std::invalid_argument("Sum of layers_per_zone must equal grid nlay.");

    // Zone boundary k-indices: [0, lpz[0], lpz[0]+lpz[1], ..., nlay]
    std::vector<size_t> zone_k(n_zones + 1, 0);
    std::partial_sum(layers_per_zone.begin(), layers_per_zone.end(), zone_k.begin() + 1,
                     [](size_t a, int b) { return a + static_cast<size_t>(b); });

    // Reusable per-pillar buffers (allocated once, reused per pillar)
    std::vector<std::array<double, 4>> orig(nkp1);
    std::vector<double> z_smooth(n_surfaces);
    std::vector<std::array<double, 4>> z_bnd(n_surfaces);

    for (size_t i = 0; i < npil_i; ++i) {
        for (size_t j = 0; j < npil_j; ++j) {
            const double x_t = coord(i, j, 0), y_t = coord(i, j, 1),
                         z_t = coord(i, j, 2);
            const double x_b = coord(i, j, 3), y_b = coord(i, j, 4),
                         z_b = coord(i, j, 5);
            const double dz = z_b - z_t;

            if (std::abs(dz) < threshold)
                continue;

            for (size_t k = 0; k < nkp1; ++k)
                for (size_t c = 0; c < 4; ++c)
                    orig[k][c] = static_cast<double>(zcorn(i, j, k, c));

            if (skip_faults && is_pillar_faulted(orig, nkp1, faults_tolerance))
                continue;

            // Find pillar/surface intersections
            bool skip = false;
            for (size_t s = 0; s < n_surfaces; ++s) {
                double t = find_pillar_surface_intersection_t(
                  x_t, y_t, z_t, x_b, y_b, z_b, surfaces[s], tolerance);
                if (std::isnan(t)) {
                    skip = true;
                    break;
                }
                z_smooth[s] = z_t + t * dz;
            }

            if (skip)
                continue;

            for (size_t s = 1; s < n_surfaces; ++s)
                z_smooth[s] = std::max(z_smooth[s], z_smooth[s - 1]);

            // Compute per-corner zone boundaries
            for (size_t s = 0; s < n_surfaces; ++s) {
                const double avg = avg4(orig[zone_k[s]]);
                for (size_t c = 0; c < 4; ++c)
                    z_bnd[s][c] = z_smooth[s] + (orig[zone_k[s]][c] - avg);
            }

            for (size_t c = 0; c < 4; ++c)
                for (size_t s = 1; s < n_surfaces; ++s)
                    z_bnd[s][c] = std::max(z_bnd[s][c], z_bnd[s - 1][c]);

            // Distribute thickness for each layer in the zone
            for (size_t c = 0; c < 4; ++c) {
                for (size_t s = 0; s < n_surfaces; ++s)
                    zcorn(i, j, zone_k[s], c) = static_cast<float>(z_bnd[s][c]);

                for (size_t zi = 0; zi < n_zones; ++zi) {
                    const size_t kt = zone_k[zi], kb = zone_k[zi + 1];
                    const double zt_orig = orig[kt][c];
                    const double thick = orig[kb][c] - zt_orig;
                    const double zt_new = z_bnd[zi][c];
                    const double zb_new = z_bnd[zi + 1][c];

                    for (size_t k = kt + 1; k < kb; ++k) {
                        const double frac =
                          (std::abs(thick) < 1e-6)
                            ? 0.0
                            : std::clamp((orig[k][c] - zt_orig) / thick, 0.0, 1.0);
                        zcorn(i, j, k, c) =
                          static_cast<float>(zt_new + frac * (zb_new - zt_new));
                    }
                }

                for (size_t k = 1; k < nkp1; ++k)
                    zcorn(i, j, k, c) =
                      std::max(zcorn(i, j, k, c), zcorn(i, j, k - 1, c));
            }
        }
    }
}

}  // namespace xtgeo::grid3d
