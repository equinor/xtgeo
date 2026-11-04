#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xtgeo.h>
#include <xtgeo/xyz.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

// use Tetrahedrons method for point in hexahedron checks since it is fast

constexpr size_t INVALID = std::numeric_limits<size_t>::max();

constexpr int MAX_RADIUS = 2;  // Maximum radius to search for neighbors

/**
 * @brief Estimate the i,j range for a point based on top/base surfaces that represent
 * the grid indices
 */
static std::tuple<size_t, size_t, size_t, size_t>
estimate_ij_range(const xyz::Point &point,
                  const regsurf::RegularSurface &top_i,
                  const regsurf::RegularSurface &top_j,
                  const regsurf::RegularSurface &base_i,
                  const regsurf::RegularSurface &base_j,
                  const size_t ncol,
                  const size_t nrow)
{

    constexpr size_t buffer = 1;

    double i_top = regsurf::get_z_from_xy(top_i, point.x(), point.y());
    double j_top = regsurf::get_z_from_xy(top_j, point.x(), point.y());
    double i_base = regsurf::get_z_from_xy(base_i, point.x(), point.y());
    double j_base = regsurf::get_z_from_xy(base_j, point.x(), point.y());

    // If all values are NaN, the point is outside the grid
    if (std::isnan(i_top) && std::isnan(j_top) && std::isnan(i_base) &&
        std::isnan(j_base)) {
        return std::make_tuple(INVALID, INVALID, INVALID, INVALID);
    }

    // If any value is NaN, search the entire grid
    if (std::isnan(i_top) || std::isnan(j_top) || std::isnan(i_base) ||
        std::isnan(j_base)) {
        return std::make_tuple(0, ncol - 1, 0, nrow - 1);
    }

    int imin =
      std::max(0, static_cast<int>(std::floor(i_top)) - static_cast<int>(buffer));
    int imax = std::min(static_cast<int>(ncol) - 1,
                        static_cast<int>(std::ceil(i_base)) + static_cast<int>(buffer));
    int jmin =
      std::max(0, static_cast<int>(std::floor(j_top)) - static_cast<int>(buffer));
    int jmax = std::min(static_cast<int>(nrow) - 1,
                        static_cast<int>(std::ceil(j_base)) + static_cast<int>(buffer));

    // Clamp to valid range and cast to size_t
    return std::make_tuple(
      static_cast<size_t>(std::max(0, imin)), static_cast<size_t>(std::max(0, imax)),
      static_cast<size_t>(std::max(0, jmin)), static_cast<size_t>(std::max(0, jmax)));
}

/**
 * @brief Check if point is above top_d or below base_d surfaces
 */
static bool
is_within_depth(const xyz::Point &point,
                const regsurf::RegularSurface &top_d,
                const regsurf::RegularSurface &base_d,
                const double threshold = 0.1)
{
    double z_top = regsurf::get_z_from_xy(top_d, point.x(), point.y());
    double z_base = regsurf::get_z_from_xy(base_d, point.x(), point.y());
    if (std::isnan(z_top) && std::isnan(z_base)) {
        return false;
    }
    double apply_threshold = threshold * 2;  // Apply threshold to both top and base

    return point.z() > z_top - apply_threshold && point.z() < z_base + apply_threshold;
}

/**
 * @brief Find a proposed i,j coordinate for a point by looking at the one_grid which is
 * the grid but with just one layer (since cornerpoint grid, this is ok)
 */
static std::tuple<size_t, size_t>
get_proposed_ij(const Grid &one_grid,
                const xyz::Point &point,
                size_t i_min,
                size_t i_max,
                size_t j_min,
                size_t j_max,
                const geometry::PointInHexahedronMethod point_in_hex_method)
{

    for (size_t i = i_min; i <= i_max; ++i) {
        for (size_t j = j_min; j <= j_max; ++j) {
            auto cell_corners =
              one_grid.get_cell_corners_cache()[i * one_grid.get_nrow() *
                                                  one_grid.get_nlay() +
                                                j * one_grid.get_nlay() + 0];
            if (is_point_in_cell(point, cell_corners, point_in_hex_method)) {
                return std::make_tuple(i, j);
            }
        }
    }

    return std::make_tuple(INVALID, INVALID);  // No match found
}

/**
 * @brief Check if a point is inside the grid's bounding box, allowing quick rejection
 */
static bool
is_point_in_grid_bounds(const xyz::Point &point,
                        const xyz::Point &min_point,
                        const xyz::Point &max_point,
                        const double epsilon = 1e-9)
{
    return !(
      point.x() < min_point.x() - epsilon || point.x() > max_point.x() + epsilon ||
      point.y() < min_point.y() - epsilon || point.y() > max_point.y() + epsilon ||
      point.z() < min_point.z() - epsilon || point.z() > max_point.z() + epsilon);
}

/**
 * @brief Search for a point within a cell column (all K layers)
 * @return true if found, false otherwise
 */
static bool
find_in_column(const Grid &grid,
               const xyz::Point &point,
               size_t i,
               size_t j,
               size_t &previous_k,
               int &found_i,
               int &found_j,
               int &found_k,
               const bool active_only,
               const py::detail::unchecked_reference<int, 3> &actnumsv,
               const geometry::PointInHexahedronMethod point_in_hex_method)
{
    // Make sure previous_k is within bounds
    previous_k = std::clamp(previous_k, size_t(0), grid.get_nlay() - 1);

    // Search outward from previous_k in both directions
    for (size_t offset = 0; offset < grid.get_nlay(); ++offset) {
        // Try upward
        int k_up = static_cast<int>(previous_k) - static_cast<int>(offset);

        if (k_up >= 0 && k_up < static_cast<int>(grid.get_nlay())) {
            // Only check if cell is active (when required)
            size_t kk = static_cast<size_t>(k_up);
            if (!active_only || (active_only && actnumsv(i, j, kk) > 0)) {
                auto cell_corners =
                  grid.get_cell_corners_cache()[i * grid.get_nrow() * grid.get_nlay() +
                                                j * grid.get_nlay() + kk];
                if (is_point_in_cell(point, cell_corners, point_in_hex_method)) {
                    found_i = static_cast<int>(i);
                    found_j = static_cast<int>(j);
                    found_k = static_cast<int>(kk);
                    previous_k = kk;  // Update for next search
                    return true;
                }
            }
        }

        // Try downward (skip if same as upward)
        int k_down = static_cast<int>(previous_k) + static_cast<int>(offset);

        if (k_down >= 0 && k_down < static_cast<int>(grid.get_nlay()) &&
            k_down != k_up) {
            size_t kk = static_cast<size_t>(k_down);
            if (!active_only || (active_only && actnumsv(i, j, kk) > 0)) {
                auto cell_corners =
                  grid.get_cell_corners_cache()[i * grid.get_nrow() * grid.get_nlay() +
                                                j * grid.get_nlay() + kk];
                if (is_point_in_cell(point, cell_corners, point_in_hex_method)) {
                    found_i = static_cast<int>(i);
                    found_j = static_cast<int>(j);
                    found_k = static_cast<int>(kk);
                    previous_k = kk;  // Update for next search
                    return true;
                }
            }
        }
    }

    return false;  // Point not found in this column
}
/**
 * @brief Look up neighboring close grid cells for a given point, assuming it to be
 * close to previous found point
 *
 * @param grid
 * @param point
 * @param previous_i
 * @param previous_j
 * @param previous_k
 * @param radius
 * @param active_only
 * @param actnumsv
 * @return std::tuple<size_t, size_t, size_t>
 */
static std::tuple<size_t, size_t, size_t>
fast_lookup_neighbours(const Grid &grid,
                       const xyz::Point &point,
                       const size_t previous_i,
                       const size_t previous_j,
                       const size_t previous_k,
                       const int radius,
                       const bool active_only,
                       const py::detail::unchecked_reference<int, 3> &actnumsv,
                       const geometry::PointInHexahedronMethod point_in_hex_method)
{
    int p_i = static_cast<int>(previous_i);
    int p_j = static_cast<int>(previous_j);
    int p_k = static_cast<int>(previous_k);

    size_t zero = 0;

    for (int di = p_i - radius; di <= p_i + radius; ++di) {
        for (int dj = p_j - radius; dj <= p_j + radius; ++dj) {
            for (int dk = p_k - radius; dk <= p_k + radius; ++dk) {
                if (di >= 0 && di < grid.get_ncol() && dj >= 0 &&
                    dj < grid.get_nrow() && dk >= 0 && dk < grid.get_nlay()) {

                    // Check if the cell is active if required
                    if (active_only && actnumsv(di, dj, dk) <= 0) {
                        continue;  // Skip inactive cells
                    }

                    size_t actual_di =
                      std::clamp(static_cast<size_t>(di), zero, grid.get_ncol() - 1);
                    size_t actual_dj =
                      std::clamp(static_cast<size_t>(dj), zero, grid.get_nrow() - 1);
                    size_t actual_dk =
                      std::clamp(static_cast<size_t>(dk), zero, grid.get_nlay() - 1);

                    auto cell_corners =
                      grid.get_cell_corners_cache()[actual_di * grid.get_nrow() *
                                                      grid.get_nlay() +
                                                    actual_dj * grid.get_nlay() +
                                                    actual_dk];
                    if (is_point_in_cell(point, cell_corners, point_in_hex_method)) {
                        return std::make_tuple(actual_di, actual_dj, actual_dk);
                    }
                }
            }
        }
    }
    // If no neighbor found, return invalid indices
    return std::make_tuple(INVALID, INVALID, INVALID);
}

namespace {
struct SpatialIndex;

static std::tuple<int, int, int>
spatial_lookup_cell(const SpatialIndex &index,
                    const Grid &grid,
                    const xyz::Point &point,
                    geometry::PointInHexahedronMethod point_in_hex_method,
                    const int *prefer_actnum);
}  // anonymous namespace

static std::shared_ptr<SpatialIndex>
get_or_build_spatial_index(const Grid &grid, bool active_only);

/**
 * @brief Spatial index that sorts cells into boxes by their position. Unlike the
 * guided search it does not assume that nearby xyz means nearby ijk, so it also
 * works for nested hybrid grids. Built once and cached on the Grid (see
 * get_or_build_spatial_index).
 */
namespace {

struct AABB
{
    // float rather than double to halve the memory per cell. The bounds are always
    // rounded outwards, so the box is never too small.
    float xn, yn, zn, xx, yx, zx;  // min/max per axis
    bool contains(double px, double py, double pz) const
    {
        return px >= xn && px <= xx && py >= yn && py <= yx && pz >= zn && pz <= zx;
    }

    // Round to float away from the middle of the box, so the box can only grow. A
    // plain cast rounds to nearest and could shrink it by a few centimetres.
    static float round_out_low(const double v)
    {
        const float f = static_cast<float>(v);
        return f > v ? std::nextafter(f, -std::numeric_limits<float>::infinity()) : f;
    }
    static float round_out_high(const double v)
    {
        const float f = static_cast<float>(v);
        return f < v ? std::nextafter(f, std::numeric_limits<float>::infinity()) : f;
    }

    // Box around the given bounds, grown by `off` on every side.
    static AABB padded(const xyz::Point &cmin, const xyz::Point &cmax, const double off)
    {
        return { round_out_low(cmin.x() - off),  round_out_low(cmin.y() - off),
                 round_out_low(cmin.z() - off),  round_out_high(cmax.x() + off),
                 round_out_high(cmax.y() + off), round_out_high(cmax.z() + off) };
    }
};

struct SpatialIndex
{
    size_t nx = 0, ny = 0, nz = 0;
    double ox = 0, oy = 0, oz = 0;
    double ibx = 0, iby = 0, ibz = 0;
    std::vector<AABB> aabbs;       // one box per cell
    std::vector<size_t> offsets;   // where each bin starts in bin_data
    std::vector<size_t> bin_data;  // cell numbers, grouped by bin
    std::vector<size_t> overflow;  // cells too large to bin
    AABB overflow_bbox{};          // box around all overflow cells, for a quick reject

    size_t to_bin(double c, double o, double inv, size_t mx) const
    {
        const double v = (c - o) * inv;
        // Written so that NaN takes this branch and lands in bin 0; converting a
        // NaN or infinite value to size_t would be undefined. Such a point is
        // rejected by the box test later anyway.
        if (!(v > 0))
            return 0;
        return v >= static_cast<double>(mx) ? mx : static_cast<size_t>(v);
    }
    size_t bin_id(size_t x, size_t y, size_t z) const
    {
        return x + y * nx + z * nx * ny;
    }
};

static SpatialIndex
build_spatial_index(const Grid &grid, const bool active_only)
{
    auto &logger = xtgeo::logging::LoggerManager::get("build_spatial_index");
    logger.debug("Building spatial index, active_only: {}", active_only);

    const size_t ncol = grid.get_ncol();
    const size_t nrow = grid.get_nrow();
    const size_t nlay = grid.get_nlay();
    const size_t total_cells = ncol * nrow * nlay;

    const auto &corners = grid.get_cell_corners_cache();
    const int *actnum = grid.get_actnumsv().data();
    const auto [min_pt, max_pt] = grid.get_bounding_box();

    SpatialIndex sx;
    constexpr size_t MAX_BIN_SPAN = 64;
    constexpr size_t MAX_BIN_PER_AXIS = 4096;
    // Upper limit on the number of bins. Limiting each axis on its own is not enough:
    // 512 bins per axis would still be 1.3e8 bins in total, and we allocate two
    // size_t arrays of that length. The limit also never exceeds the cell count, so
    // small grids stay cheap.
    constexpr size_t MAX_TOTAL_BINS = 16000000;
    const double span_x = std::max(max_pt.x() - min_pt.x(), 1e-9);
    const double span_y = std::max(max_pt.y() - min_pt.y(), 1e-9);
    const double span_z = std::max(max_pt.z() - min_pt.z(), 1e-9);

    sx.aabbs.resize(total_cells);

    // Pass 1: make a box per cell, and sum the cell sizes so we can average them
    double sum_cell_x = 0.0, sum_cell_y = 0.0, sum_cell_z = 0.0;
    size_t n_used = 0;
    for (size_t ci = 0; ci < total_cells; ++ci) {
        if (active_only && actnum[ci] <= 0)
            continue;
        const auto [cmin, cmax] = get_cell_bounding_box(corners[ci]);
        const double cell_x = cmax.x() - cmin.x();
        const double cell_y = cmax.y() - cmin.y();
        const double cell_z = cmax.z() - cmin.z();
        const double off = 1e-6 * std::max({ cell_x, cell_y, cell_z });
        sx.aabbs[ci] = AABB::padded(cmin, cmax, off);
        sum_cell_x += cell_x;
        sum_cell_y += cell_y;
        sum_cell_z += cell_z;
        ++n_used;
    }

    // Make each bin about one cell wide, per axis. Using one cube-shaped bin size
    // for all axes works badly here: reservoir cells are usually wide and thin, so
    // a cube would cover many cells sideways and hardly any vertically.
    const double scale_n = n_used > 0 ? static_cast<double>(n_used) : 1.0;
    auto axis_bins = [&](const double span, const double sum_cell) -> size_t {
        const double mean_cell = std::max(sum_cell / scale_n, span * 1e-6);
        return std::clamp(static_cast<size_t>(std::ceil(span / mean_cell)), size_t(1),
                          MAX_BIN_PER_AXIS);
    };
    std::array<size_t, 3> nbins = { axis_bins(span_x, sum_cell_x),
                                    axis_bins(span_y, sum_cell_y),
                                    axis_bins(span_z, sum_cell_z) };

    // Too many bins in total: scale all three axes down by the same factor
    const double budget =
      static_cast<double>(std::clamp(n_used, size_t(1), MAX_TOTAL_BINS));
    auto total_bins = [&] {
        return static_cast<double>(nbins[0]) * nbins[1] * nbins[2];
    };
    if (total_bins() > budget) {
        const double shrink = std::cbrt(budget / total_bins());
        for (size_t &n : nbins)
            n = std::max(size_t(1), static_cast<size_t>(n * shrink));
        // An axis that is already 1 cannot shrink further, so we may still be over
        // the limit. Halve the longest axis until we fit.
        while (total_bins() > budget) {
            size_t &longest = *std::max_element(nbins.begin(), nbins.end());
            if (longest == 1)
                break;
            longest /= 2;
        }
    }

    sx.nx = nbins[0];
    sx.ny = nbins[1];
    sx.nz = nbins[2];

    sx.ibx = sx.nx / span_x;
    sx.iby = sx.ny / span_y;
    sx.ibz = sx.nz / span_z;
    sx.ox = min_pt.x();
    sx.oy = min_pt.y();
    sx.oz = min_pt.z();

    const size_t nbin = sx.nx * sx.ny * sx.nz;
    std::vector<size_t> counts(nbin, 0);

    auto span = [&](const AABB &box) {
        return std::array{
            sx.to_bin(box.xn, sx.ox, sx.ibx, sx.nx - 1),
            sx.to_bin(box.xx, sx.ox, sx.ibx, sx.nx - 1),
            sx.to_bin(box.yn, sx.oy, sx.iby, sx.ny - 1),
            sx.to_bin(box.yx, sx.oy, sx.iby, sx.ny - 1),
            sx.to_bin(box.zn, sx.oz, sx.ibz, sx.nz - 1),
            sx.to_bin(box.zx, sx.oz, sx.ibz, sx.nz - 1),
        };
    };
    auto is_overflow = [&](const std::array<size_t, 6> &s) {
        return (s[1] - s[0] + 1) * (s[3] - s[2] + 1) * (s[5] - s[4] + 1) > MAX_BIN_SPAN;
    };

    // Pass 2: count how many cells land in each bin, and set the large ones aside
    for (size_t ci = 0; ci < total_cells; ++ci) {
        if (active_only && actnum[ci] <= 0)
            continue;
        const auto s = span(sx.aabbs[ci]);
        if (is_overflow(s)) {
            const AABB &b = sx.aabbs[ci];
            if (sx.overflow.empty()) {
                sx.overflow_bbox = b;
            } else {
                sx.overflow_bbox = { std::min(sx.overflow_bbox.xn, b.xn),
                                     std::min(sx.overflow_bbox.yn, b.yn),
                                     std::min(sx.overflow_bbox.zn, b.zn),
                                     std::max(sx.overflow_bbox.xx, b.xx),
                                     std::max(sx.overflow_bbox.yx, b.yx),
                                     std::max(sx.overflow_bbox.zx, b.zx) };
            }
            sx.overflow.push_back(ci);
        } else {
            for (size_t bz = s[4]; bz <= s[5]; ++bz)
                for (size_t by = s[2]; by <= s[3]; ++by)
                    for (size_t bx = s[0]; bx <= s[1]; ++bx)
                        ++counts[sx.bin_id(bx, by, bz)];
        }
    }

    // Give every bin its slice of bin_data, then fill it. Cells are added in
    // increasing cell number, so each slice comes out sorted on its own and lookups
    // always return the same cell.
    sx.offsets.assign(nbin + 1, 0);
    for (size_t b = 0; b < nbin; ++b)
        sx.offsets[b + 1] = sx.offsets[b] + counts[b];
    counts = std::vector<size_t>();  // free before allocating bin_data

    sx.bin_data.resize(sx.offsets.back());
    std::vector<size_t> pos(nbin, 0);
    for (size_t ci = 0; ci < total_cells; ++ci) {
        if (active_only && actnum[ci] <= 0)
            continue;
        const auto s = span(sx.aabbs[ci]);
        if (is_overflow(s))
            continue;
        for (size_t bz = s[4]; bz <= s[5]; ++bz)
            for (size_t by = s[2]; by <= s[3]; ++by)
                for (size_t bx = s[0]; bx <= s[1]; ++bx) {
                    const size_t bid = sx.bin_id(bx, by, bz);
                    sx.bin_data[sx.offsets[bid] + pos[bid]++] = ci;
                }
    }
    return sx;
}

/**
 * @brief Find the cell that contains a point, using the spatial index.
 *
 * @param prefer_actnum When set, an active cell wins over an inactive one. Only
 * matters when the index includes inactive cells, where a mother cell and a refined
 * cell can both contain the point. Pass nullptr to take the first cell found.
 */
static std::tuple<int, int, int>
spatial_lookup_cell(const SpatialIndex &sx,
                    const Grid &grid,
                    const xyz::Point &point,
                    const geometry::PointInHexahedronMethod point_in_hex_method,
                    const int *prefer_actnum)
{
    const size_t nrow = grid.get_nrow();
    const size_t nlay = grid.get_nlay();
    const size_t nrow_nlay = nrow * nlay;
    const auto &corners = grid.get_cell_corners_cache();
    const double px = point.x(), py = point.y(), pz = point.z();

    size_t hit = INVALID;
    size_t inactive_hit = INVALID;

    auto try_cell = [&](size_t ci) -> bool {
        return sx.aabbs[ci].contains(px, py, pz) &&
               is_point_in_cell(point, corners[ci], point_in_hex_method);
    };
    // Returns true when a wanted cell is found. An inactive cell is kept aside and
    // used only if nothing better shows up.
    auto consider = [&](size_t ci) -> bool {
        if (!try_cell(ci))
            return false;
        if (!prefer_actnum || prefer_actnum[ci] > 0) {
            hit = ci;
            return true;
        }
        if (inactive_hit == INVALID)
            inactive_hit = ci;
        return false;
    };
    auto search_bin = [&](size_t bid) -> bool {
        for (size_t i = sx.offsets[bid]; i < sx.offsets[bid + 1]; ++i)
            if (consider(sx.bin_data[i]))
                return true;
        return false;
    };

    const size_t bx0 = sx.to_bin(px, sx.ox, sx.ibx, sx.nx - 1);
    const size_t by0 = sx.to_bin(py, sx.oy, sx.iby, sx.ny - 1);
    const size_t bz0 = sx.to_bin(pz, sx.oz, sx.ibz, sx.nz - 1);

    bool found = search_bin(sx.bin_id(bx0, by0, bz0));

    // Last, check the large cells that were never binned. The box around them lets
    // us skip the whole list in one test.
    if (!found && !sx.overflow.empty() && sx.overflow_bbox.contains(px, py, pz)) {
        for (size_t oi = 0; oi < sx.overflow.size(); ++oi)
            if (consider(sx.overflow[oi]))
                break;
    }

    if (hit == INVALID)
        hit = inactive_hit;

    if (hit == INVALID)
        return std::make_tuple(-1, -1, -1);

    return std::make_tuple(static_cast<int>(hit / nrow_nlay),
                           static_cast<int>((hit % nrow_nlay) / nlay),
                           static_cast<int>(hit % nlay));
}

}  // anonymous namespace

/**
 * @brief Get the spatial index for this grid, building it the first time. It is
 * kept on the Grid and reused, until the geometry changes (which gives a new C++
 * Grid with an empty cache) or active_only differs from last time.
 */
static std::shared_ptr<SpatialIndex>
get_or_build_spatial_index(const Grid &grid, const bool active_only)
{
    auto cached = grid.get_spatial_index_cache();
    if (cached && grid.get_spatial_index_active_only() == active_only) {
        return std::static_pointer_cast<SpatialIndex>(cached);
    }
    auto sx = std::make_shared<SpatialIndex>(build_spatial_index(grid, active_only));
    grid.set_spatial_index_cache(sx, active_only);
    return sx;
}

/**
 * @brief MAIN entry point. Given an array of Points (organized as Polygon/PointSet),
 * return the grid indices that contains the points
 */
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
get_indices_from_pointset(const Grid &grid,
                          const xyz::PointSet &points,
                          const Grid &one_grid,
                          const regsurf::RegularSurface &top_i,
                          const regsurf::RegularSurface &top_j,
                          const regsurf::RegularSurface &base_i,
                          const regsurf::RegularSurface &base_j,
                          const regsurf::RegularSurface &top_d,
                          const regsurf::RegularSurface &base_d,
                          const double threshold_magic,
                          const bool active_only,
                          const geometry::PointInHexahedronMethod point_in_hex_method)
{

    // Get the number of points
    size_t num_points = points.size();

    // Create output arrays for indices (initialized to -1)
    py::array_t<int> i_indices(num_points);
    py::array_t<int> j_indices(num_points);
    py::array_t<int> k_indices(num_points);

    auto i_indices_ = i_indices.mutable_unchecked<1>();
    auto j_indices_ = j_indices.mutable_unchecked<1>();
    auto k_indices_ = k_indices.mutable_unchecked<1>();

    auto actnumsv_ = grid.get_actnumsv().unchecked<3>();

    // With active_only=false a point can lie in more than one cell, e.g. in both an
    // inactive mother cell and the active refined cell inside it. Pick the active
    // one, so the answer is the useful one and is always the same.
    const int *prefer_actnum = active_only ? nullptr : grid.get_actnumsv().data();
    // The fast path below therefore only accepts active cells. A point that lies in
    // inactive cells only is left to the fallback further down.
    constexpr bool fastpath_active_only = true;

    // Initialize all indices to -1 (default for points not found in any cell)
    for (size_t idx = 0; idx < num_points; ++idx) {
        i_indices_(idx) = -1;
        j_indices_(idx) = -1;
        k_indices_(idx) = -1;
    }

    // Get grid bounding box once
    auto [min_grid_point, max_grid_point] = one_grid.get_bounding_box();

    size_t previous_i = INVALID;
    size_t previous_j = INVALID;
    size_t previous_k = 0;

    // Built the first time a point needs it, then reused for the rest of the loop
    std::shared_ptr<SpatialIndex> fallback_index;

    // Process each point
    xyz::Point previous_point(0.0, 0.0, 0.0);
    for (size_t idx = 0; idx < num_points; ++idx) {
        const auto &point = points.get_point(idx);

        if (!is_point_in_grid_bounds(point, min_grid_point, max_grid_point)) {
            continue;
        }

        bool found = false;

        // Fast path: search guided by the top and base surfaces. Works well for
        // ordinary grids, but can miss in nested hybrid grids, where a cell's (i,j)
        // says little about where it actually lies. Misses go to the fallback below.
        if (is_within_depth(point, top_d, base_d, threshold_magic)) {
            // compute the Euclidean distance to the previous point
            double distance_to_previous = (point - previous_point).norm();

            if (distance_to_previous < threshold_magic && previous_i != INVALID) {
                // If we have a previous point, use its i,j as starting point.
                // This is an optimization for organized pointsets, meaning that the
                // next point is likely close to the previous one.
                for (int radius = 0; radius <= MAX_RADIUS && !found; ++radius) {
                    auto [new_i, new_j, new_k] = fast_lookup_neighbours(
                      grid, point, previous_i, previous_j, previous_k, radius,
                      fastpath_active_only, actnumsv_, point_in_hex_method);

                    if (new_i != INVALID) {
                        i_indices_(idx) = static_cast<int>(new_i);
                        j_indices_(idx) = static_cast<int>(new_j);
                        k_indices_(idx) = static_cast<int>(new_k);
                        previous_i = new_i;
                        previous_j = new_j;
                        previous_k = new_k;
                        found = true;
                    }
                }
            }

            if (!found) {
                // Get potential i,j range for the point
                auto [imin, imax, jmin, jmax] =
                  estimate_ij_range(point, top_i, top_j, base_i, base_j,
                                    grid.get_ncol(), grid.get_nrow());

                if (imin != INVALID) {
                    // Try to find a proposed column for efficiency
                    auto [i_est, j_est] = get_proposed_ij(
                      one_grid, point, imin, imax, jmin, jmax, point_in_hex_method);

                    // If we found a proposed column, restrict search to it
                    if (i_est != INVALID && j_est != INVALID) {
                        imin = imax = i_est;
                        jmin = jmax = j_est;
                    }

                    // Search for the point in the possible cell columns
                    for (size_t i = imin; i <= imax && !found; ++i) {
                        for (size_t j = jmin; j <= jmax && !found; ++j) {
                            int found_i = -1, found_j = -1, found_k = -1;
                            found = find_in_column(
                              grid, point, i, j, previous_k, found_i, found_j, found_k,
                              fastpath_active_only, actnumsv_, point_in_hex_method);
                            if (found) {
                                i_indices_(idx) = found_i;
                                j_indices_(idx) = found_j;
                                k_indices_(idx) = found_k;
                                previous_i = static_cast<size_t>(found_i);
                                previous_j = static_cast<size_t>(found_j);
                                previous_k = static_cast<size_t>(found_k);
                            }
                        }
                    }
                }
            }
        }

        // Fallback: used when the fast path found nothing. This is what makes nested
        // hybrid grids work, and it is also the only place that can return an
        // inactive cell when active_only=false.
        if (!found) {
            if (!fallback_index)
                fallback_index = get_or_build_spatial_index(grid, active_only);
            auto [fb_i, fb_j, fb_k] = spatial_lookup_cell(
              *fallback_index, grid, point, point_in_hex_method, prefer_actnum);
            if (fb_i >= 0) {
                i_indices_(idx) = fb_i;
                j_indices_(idx) = fb_j;
                k_indices_(idx) = fb_k;
                previous_i = static_cast<size_t>(fb_i);
                previous_j = static_cast<size_t>(fb_j);
                previous_k = static_cast<size_t>(fb_k);
                found = true;
            }
        }

        // Reset previous indices if no cell found
        if (!found) {
            previous_i = INVALID;
            previous_j = INVALID;
            previous_k = 0;
        }
        previous_point = point;  // Update previous point for next iteration
    }

    return std::make_tuple(i_indices, j_indices, k_indices);
}

/**
 * @brief Point lookup that uses the spatial index for every point. It does not
 * assume that nearby xyz means nearby ijk, so it handles both ordinary and nested
 * hybrid grids. The index is kept on the Grid and reused, the points are shared
 * between threads, and points close together are looked up near the previous hit.
 */
std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<int>>
get_indices_from_pointset_cached(
  const Grid &grid,
  const xyz::PointSet &points,
  const bool active_only,
  const geometry::PointInHexahedronMethod point_in_hex_method)
{
    const size_t num_points = points.size();

    py::array_t<int> i_indices(num_points);
    py::array_t<int> j_indices(num_points);
    py::array_t<int> k_indices(num_points);
    int *iout = i_indices.mutable_data();
    int *jout = j_indices.mutable_data();
    int *kout = k_indices.mutable_data();
    std::fill_n(iout, num_points, -1);
    std::fill_n(jout, num_points, -1);
    std::fill_n(kout, num_points, -1);

    // Empty grids cannot contain points and have no bounding box to query.
    if (num_points == 0 || grid.get_ncol() == 0 || grid.get_nrow() == 0 ||
        grid.get_nlay() == 0)
        return std::make_tuple(i_indices, j_indices, k_indices);

    // Build caches and the index before splitting the work between threads.
    const auto [min_pt, max_pt] = grid.get_bounding_box();
    auto index = get_or_build_spatial_index(grid, active_only);
    const double bounds_eps = 1e-9;

    const auto &corners = grid.get_cell_corners_cache();
    const int *actnum = grid.get_actnumsv().data();
    // With active_only=false, pick the active cell when a mother cell and a refined
    // cell both contain the point.
    const int *prefer_actnum = active_only ? nullptr : actnum;
    const size_t nrow = grid.get_nrow();
    const size_t nlay = grid.get_nlay();
    const size_t nrow_nlay = nrow * nlay;
    const int ic = static_cast<int>(grid.get_ncol());
    const int ir = static_cast<int>(nrow);
    const int il = static_cast<int>(nlay);

    constexpr size_t BLOCK = 16384;
    const int nblocks = static_cast<int>((num_points + BLOCK - 1) / BLOCK);

    // clang-format off
#ifdef XTGEO_USE_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    // clang-format on
    for (int blk = 0; blk < nblocks; ++blk) {
        const size_t begin = static_cast<size_t>(blk) * BLOCK;
        const size_t end = std::min(begin + BLOCK, num_points);
        int pi = -1, pj = -1, pk = -1;

        for (size_t idx = begin; idx < end; ++idx) {
            const auto &pt = points.get_point(idx);
            if (!is_point_in_grid_bounds(pt, min_pt, max_pt, bounds_eps)) {
                pi = -1;
                continue;
            }
            const double px = pt.x(), py = pt.y(), pz = pt.z();
            bool found = false;

            // Points often come in order, so try around the previous hit first
            if (pi >= 0) {
                for (int r = 0; r <= MAX_RADIUS && !found; ++r)
                    for (int di = std::max(0, pi - r);
                         di <= std::min(ic - 1, pi + r) && !found; ++di)
                        for (int dj = std::max(0, pj - r);
                             dj <= std::min(ir - 1, pj + r) && !found; ++dj)
                            for (int dk = std::max(0, pk - r);
                                 dk <= std::min(il - 1, pk + r) && !found; ++dk) {
                                const size_t ci = static_cast<size_t>(di) * nrow_nlay +
                                                  static_cast<size_t>(dj) * nlay +
                                                  static_cast<size_t>(dk);
                                if (active_only && actnum[ci] <= 0)
                                    continue;
                                // Skip inactive cells here; if the point really is in
                                // one, the index lookup below reports it.
                                if (prefer_actnum && actnum[ci] <= 0)
                                    continue;
                                if (index->aabbs[ci].contains(px, py, pz) &&
                                    is_point_in_cell(pt, corners[ci],
                                                     point_in_hex_method)) {
                                    iout[idx] = di;
                                    jout[idx] = dj;
                                    kout[idx] = dk;
                                    pi = di;
                                    pj = dj;
                                    pk = dk;
                                    found = true;
                                }
                            }
            }
            if (found)
                continue;

            // Not nearby, so look it up in the index
            auto [ri, rj, rk] =
              spatial_lookup_cell(*index, grid, pt, point_in_hex_method, prefer_actnum);
            if (ri >= 0) {
                iout[idx] = ri;
                jout[idx] = rj;
                kout[idx] = rk;
                pi = ri;
                pj = rj;
                pk = rk;
            } else {
                pi = -1;
            }
        }
    }

    return std::make_tuple(i_indices, j_indices, k_indices);
}

}  // namespace xtgeo::grid3d
