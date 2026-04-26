/*
 * =====================================================================================
 *  well_cell_intersections.cpp
 *
 *  Compute the entry and exit point (X, Y, Z, MD) of a well trajectory through
 *  every cell of a 3D corner-point grid it passes through.
 *
 *  HYBRID ALGORITHM (ray-tracing + sample-and-bisect fallback)
 *  -----------------------------------------------------------
 *  For every consecutive pair (P_a, P_b) of trajectory samples we walk the
 *  straight line joining them. The walk uses two cooperating phases:
 *
 *    (1) FAST PATH — analytic ray vs. cell intersection.
 *        For the *current* cell we compute the smallest parameter t_exit > t_now
 *        for which the ray crosses one of the 6 cell faces (each face is split
 *        into two triangles). The face that is hit determines the next cell.
 *        Cost is O(1) per cell traversed.
 *
 *    (2) FALLBACK — sample-and-bisect.
 *        Used when the current cell is non-convex, when the ray test fails
 *        for numerical reasons, or when we do not yet know which cell we are
 *        in (segment start, or after the well has left and re-entered the
 *        grid). The trajectory segment is sub-sampled with `sampling_step`
 *        and the cell of every sample is located via the existing
 *        point-in-hexahedron predicate; cell changes are localised by
 *        bisection.
 *
 *  An "intersection record" is opened when the trajectory enters a cell, and
 *  closed when it leaves. Re-entries produce additional records.
 *
 *  Several cheap fast-rejects (grid AABB, per-cell AABB, bounded spiral
 *  search around the previous cell, lazy per-column XY AABB) keep the cost
 *  bounded when the trajectory leaves the grid or visits numerically
 *  degenerate cell boundaries; without them point-location degrades to a
 *  full O(ncells) scan per sample.
 *
 *  RETURN VALUE
 *  ------------
 *  A pybind11 dict with these 1D numpy arrays (length = number of records):
 *      "i", "j", "k"                                 (int32, 0-based)
 *      "entry_x", "entry_y", "entry_z", "entry_md"   (float64)
 *      "exit_x",  "exit_y",  "exit_z",  "exit_md"    (float64)
 * =====================================================================================
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>
#include <xtgeo/xyz.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

namespace {

constexpr double T_EPS = 1e-12;

// Namespace alias for shared utilities
namespace generic = geometry::generic;

// -----------------------------------------------------------------------------
// Geometry helpers
// -----------------------------------------------------------------------------

/// Checks if a point lies within an axis-aligned bounding box (AABB) with tolerance.
inline bool
point_in_aabb(const xyz::Point &p,
              const xyz::Point &corner_min,
              const xyz::Point &corner_max,
              double eps = 1e-6)
{
    return p.x() >= corner_min.x() - eps && p.x() <= corner_max.x() + eps &&
           p.y() >= corner_min.y() - eps && p.y() <= corner_max.y() + eps &&
           p.z() >= corner_min.z() - eps && p.z() <= corner_max.z() + eps;
}

// AABB pre-reject for point-in-cell. Checks each axis with early exit to
// skip remaining work when the point is clearly outside.
inline bool
point_in_cell_aabb(const xyz::Point &p, const CellCorners &c, double eps = 1e-9)
{
    auto [xlo, xhi] =
      std::minmax({ c.upper_sw.x(), c.upper_se.x(), c.upper_nw.x(), c.upper_ne.x(),
                    c.lower_sw.x(), c.lower_se.x(), c.lower_nw.x(), c.lower_ne.x() });
    if (p.x() < xlo - eps || p.x() > xhi + eps)
        return false;
    auto [ylo, yhi] =
      std::minmax({ c.upper_sw.y(), c.upper_se.y(), c.upper_nw.y(), c.upper_ne.y(),
                    c.lower_sw.y(), c.lower_se.y(), c.lower_nw.y(), c.lower_ne.y() });
    if (p.y() < ylo - eps || p.y() > yhi + eps)
        return false;
    auto [zlo, zhi] =
      std::minmax({ c.upper_sw.z(), c.upper_se.z(), c.upper_nw.z(), c.upper_ne.z(),
                    c.lower_sw.z(), c.lower_se.z(), c.lower_nw.z(), c.lower_ne.z() });
    return p.z() >= zlo - eps && p.z() <= zhi + eps;
}

// Liang-Barsky line clipping against a 3D AABB. Returns (t_in, t_out) for
// the segment p0 + t*(p1-p0) clipped to the box; if the line misses the box,
// returns t_in > t_out.
inline std::pair<double, double>
clip_segment_to_aabb(const xyz::Point &p0,
                     const xyz::Point &p1,
                     const xyz::Point &mn,
                     const xyz::Point &mx,
                     double eps = 1e-6)
{
    double t_in = 0.0, t_out = 1.0;
    auto clip = [&](double p, double q) {
        if (std::fabs(p) < 1e-30)
            return q >= 0.0;  // line parallel; inside iff q >= 0
        double t = q / p;
        if (p < 0.0) {
            if (t > t_out)
                return false;
            if (t > t_in)
                t_in = t;
        } else {
            if (t < t_in)
                return false;
            if (t < t_out)
                t_out = t;
        }
        return true;
    };
    const double dx = p1.x() - p0.x();
    const double dy = p1.y() - p0.y();
    const double dz = p1.z() - p0.z();
    if (!clip(-dx, p0.x() - (mn.x() - eps)))
        return { 1.0, 0.0 };
    if (!clip(dx, (mx.x() + eps) - p0.x()))
        return { 1.0, 0.0 };
    if (!clip(-dy, p0.y() - (mn.y() - eps)))
        return { 1.0, 0.0 };
    if (!clip(dy, (mx.y() + eps) - p0.y()))
        return { 1.0, 0.0 };
    if (!clip(-dz, p0.z() - (mn.z() - eps)))
        return { 1.0, 0.0 };
    if (!clip(dz, (mx.z() + eps) - p0.z()))
        return { 1.0, 0.0 };
    return { t_in, t_out };
}

// -----------------------------------------------------------------------------
// Cell faces (for ray–cell exit test)
// -----------------------------------------------------------------------------
// CellCorners corner index map: 0=usw 1=use 2=unw 3=une 4=lsw 5=lse 6=lnw 7=lne
// Faces:  0 = -i (west),  1 = +i (east),  2 = -j (south),  3 = +j (north),
//         4 = -k (top),   5 = +k (base)

constexpr std::array<std::array<int, 3>, 6> FACE_NEIGHBOUR = { { { -1, 0, 0 },
                                                                 { +1, 0, 0 },
                                                                 { 0, -1, 0 },
                                                                 { 0, +1, 0 },
                                                                 { 0, 0, -1 },
                                                                 { 0, 0, +1 } } };

constexpr std::array<std::array<std::array<int, 3>, 2>, 6> FACE_TRIS = { {
  { { { { 0, 2, 6 } }, { { 0, 6, 4 } } } },  // -i
  { { { { 1, 3, 7 } }, { { 1, 7, 5 } } } },  // +i
  { { { { 0, 1, 5 } }, { { 0, 5, 4 } } } },  // -j
  { { { { 2, 3, 7 } }, { { 2, 7, 6 } } } },  // +j
  { { { { 0, 1, 3 } }, { { 0, 3, 2 } } } },  // -k
  { { { { 4, 5, 7 } }, { { 4, 7, 6 } } } },  // +k
} };

// Smallest t > t_min on ray (origin + t*dir) where the ray crosses one of the
// 6 faces of `corners`. Returns (t_exit, face_id) or (NaN, -1).
std::pair<double, int>
ray_cell_exit(const xyz::Point &origin,
              const xyz::Point &dir,
              const CellCorners &corners,
              double t_min)
{
    double best_t = std::numeric_limits<double>::infinity();
    int best_face = -1;
    const double t_floor = t_min + T_EPS;
    for (int f = 0; f < 6; ++f) {
        for (int tri = 0; tri < 2; ++tri) {
            const auto &idx = FACE_TRIS[f][tri];
            double t =
              geometry::ray::triangle_t(origin, dir, corners.corner(idx[0]),
                                        corners.corner(idx[1]), corners.corner(idx[2]));
            if (!std::isnan(t) && t > t_floor && t < best_t) {
                best_t = t;
                best_face = f;
            }
        }
    }
    if (best_face < 0)
        return { std::numeric_limits<double>::quiet_NaN(), -1 };
    return { best_t, best_face };
}

// Lightweight struct CellId to keep track cell i, j, k
struct CellId
{
    int i = -1, j = -1, k = -1;
    bool valid() const { return i >= 0 && j >= 0 && k >= 0; }
    bool operator==(const CellId &o) const { return i == o.i && j == o.j && k == o.k; }
    bool operator!=(const CellId &o) const { return !(*this == o); }
};

// Lazy per-(i,j) XY AABB built on first need from the corner cache. Each
// entry is the union of XY-extents of all cells (k=0..nlay-1) in that column.
// Used as a cheap O(ncols) prune before falling back to a full scan over
// (i,j,k) when locating a point with no usable hint.
struct ColumnXYBBoxCache
{
    std::vector<double> xmin, xmax, ymin, ymax;
    int nrow = 0;
    bool built = false;

    void build(const Grid &grid)
    {
        if (built)
            return;
        const int ncol = static_cast<int>(grid.get_ncol());
        nrow = static_cast<int>(grid.get_nrow());
        // Each column (i, j) is bounded by 4 corner pillars at (i, j),
        // (i+1, j), (i, j+1), (i+1, j+1). Cell corners along a pillar lie on
        // the line segment between its top and bottom endpoints, so the column
        // XY footprint is contained in the union of the four pillars' XY
        // AABBs (built from 8 endpoints total). This is an overestimate (safe)
        // and avoids walking all ncol*nrow*nlay cells.
        const auto coords = grid.get_coordsv().unchecked<3>();
        const size_t nrowp1 = static_cast<size_t>(nrow + 1);
        // Per-pillar XY AABB: (min(top_x, bot_x), max(...), min(top_y, bot_y), ...)
        const size_t npill = static_cast<size_t>(ncol + 1) * nrowp1;
        std::vector<double> pxmin(npill), pxmax(npill), pymin(npill), pymax(npill);
        for (int i = 0; i <= ncol; ++i) {
            for (int j = 0; j <= nrow; ++j) {
                const size_t p = static_cast<size_t>(i) * nrowp1 + j;
                const double tx = coords(i, j, 0), ty = coords(i, j, 1);
                const double bx = coords(i, j, 3), by = coords(i, j, 4);
                pxmin[p] = std::min(tx, bx);
                pxmax[p] = std::max(tx, bx);
                pymin[p] = std::min(ty, by);
                pymax[p] = std::max(ty, by);
            }
        }
        const size_t ncols = static_cast<size_t>(ncol) * static_cast<size_t>(nrow);
        xmin.resize(ncols);
        xmax.resize(ncols);
        ymin.resize(ncols);
        ymax.resize(ncols);
        for (int i = 0; i < ncol; ++i) {
            for (int j = 0; j < nrow; ++j) {
                const size_t c = static_cast<size_t>(i) * nrow + j;
                const size_t p00 = static_cast<size_t>(i) * nrowp1 + j;
                const size_t p10 = p00 + nrowp1;
                const size_t p01 = p00 + 1;
                const size_t p11 = p10 + 1;
                xmin[c] = std::min({ pxmin[p00], pxmin[p10], pxmin[p01], pxmin[p11] });
                xmax[c] = std::max({ pxmax[p00], pxmax[p10], pxmax[p01], pxmax[p11] });
                ymin[c] = std::min({ pymin[p00], pymin[p10], pymin[p01], pymin[p11] });
                ymax[c] = std::max({ pymax[p00], pymax[p10], pymax[p01], pymax[p11] });
            }
        }
        built = true;
    }

    bool xy_in_column(int i, int j, double x, double y, double eps = 1e-6) const
    {
        const size_t c = static_cast<size_t>(i) * nrow + j;
        return x >= xmin[c] - eps && x <= xmax[c] + eps && y >= ymin[c] - eps &&
               y <= ymax[c] + eps;
    }
};

// -----------------------------------------------------------------------------
// LocateCtx: bundles everything point-location and cell-marching needs.
// Replaces what would otherwise be 7-8 parameters threaded through every
// helper (grid, actnumsv, method, active_only, grid AABB, column cache).
// -----------------------------------------------------------------------------

struct LocateCtx
{
    const Grid &grid;
    const py::detail::unchecked_reference<int, 3> &actnumsv;
    geometry::PointInHexahedronMethod method;
    bool active_only;
    xyz::Point grid_min, grid_max;
    ColumnXYBBoxCache &col_cache;
    int ncol, nrow, nlay;
    const std::vector<CellCorners> &corners;

    // Single-slot cache for the most recently tested cell's non-convexity
    // flag. With long trajectories the same cell is tested once per segment
    // that stays inside it, so this avoids most repeated calls.
    CellId nc_cell;
    bool nc_value = false;

    bool in_grid(int i, int j, int k) const
    {
        return i >= 0 && i < ncol && j >= 0 && j < nrow && k >= 0 && k < nlay;
    }

    bool cell_active(int i, int j, int k) const
    {
        return !active_only || actnumsv(i, j, k) != 0;
    }

    size_t cell_idx(int i, int j, int k) const
    {
        return (static_cast<size_t>(i) * nrow + j) * nlay + k;
    }

    const CellCorners &cell(int i, int j, int k) const
    {
        return corners[cell_idx(i, j, k)];
    }

    // True iff (i,j,k) is in-grid, active, and `point` lies in its hexahedron.
    // Uses a cheap AABB pre-reject before the expensive tetrahedral test.
    bool point_in(int i, int j, int k, const xyz::Point &point) const
    {
        if (!in_grid(i, j, k) || !cell_active(i, j, k))
            return false;
        const auto &cc = cell(i, j, k);
        if (!point_in_cell_aabb(point, cc))
            return false;
        return is_point_in_cell(point, cc, method);
    }

    // Cached check for cells that need the sampling fallback (non-convex or
    // severely distorted faces). One-slot LRU avoids recomputing on every
    // segment that stays in the same cell.
    bool needs_fallback(const CellId &c, const CellCorners &cc)
    {
        if (c == nc_cell)
            return nc_value;
        nc_cell = c;
        nc_value = is_cell_non_convex(cc) || is_cell_distorted(cc);
        return nc_value;
    }
};

// -----------------------------------------------------------------------------
// Point-in-grid: spiral search around `previous`, then column-pruned scan.
// -----------------------------------------------------------------------------

CellId
find_cell_for_point(LocateCtx &ctx, const xyz::Point &point, const CellId &previous)
{
    // Quick reject: a point clearly outside the grid AABB cannot be in any
    // cell. Without this, every "well outside grid" sample triggers an
    // expensive scan.
    if (!point_in_aabb(point, ctx.grid_min, ctx.grid_max))
        return {};

    if (previous.valid()) {
        // Spiral around the previous hint. Capped at a small radius:
        // typically the next cell is one step away, and beyond ~8 cells the
        // shell visits Theta(r^2) cells with diminishing returns. If
        // exhausted (e.g. boundary-degenerate hint) we fall through to the
        // column-pruned scan below.
        const int spiral_cap = 8;
        const int max_radius = std::min({ ctx.ncol, ctx.nrow, ctx.nlay, spiral_cap });
        for (int radius = 0; radius <= max_radius; ++radius) {
            for (int di = -radius; di <= radius; ++di) {
                for (int dj = -radius; dj <= radius; ++dj) {
                    for (int dk = -radius; dk <= radius; ++dk) {
                        if (std::max({ std::abs(di), std::abs(dj), std::abs(dk) }) !=
                            radius) {
                            continue;
                        }
                        const int i = previous.i + di;
                        const int j = previous.j + dj;
                        const int k = previous.k + dk;
                        if (ctx.point_in(i, j, k, point))
                            return { i, j, k };
                    }
                }
            }
        }
    }

    // Column-pruned scan: only look at columns whose XY AABB contains the
    // point, then test every K within those columns. Reduces the worst case
    // from O(ncells) to O(ncols + matched * nlay).
    ctx.col_cache.build(ctx.grid);
    for (int i = 0; i < ctx.ncol; ++i) {
        for (int j = 0; j < ctx.nrow; ++j) {
            if (!ctx.col_cache.xy_in_column(i, j, point.x(), point.y()))
                continue;
            for (int k = 0; k < ctx.nlay; ++k) {
                if (ctx.point_in(i, j, k, point))
                    return { i, j, k };
            }
        }
    }
    return {};
}

// -----------------------------------------------------------------------------
// Sampling fallback path
// -----------------------------------------------------------------------------

struct OpenRecord
{
    CellId cell;
    xyz::Point entry_xyz;
    double entry_md;
};

// Bisection-localised cell change between t_lo (cell = lo_cell) and t_hi
// (cell != lo_cell) on the segment p0->p1.
double
bisect_cell_boundary(LocateCtx &ctx,
                     const xyz::Point &p0,
                     const xyz::Point &p1,
                     double t_lo,
                     double t_hi,
                     const CellId &lo_cell,
                     const CellId &hi_cell,
                     int max_iter)
{
    CellId hint = lo_cell.valid() ? lo_cell : hi_cell;
    for (int it = 0; it < max_iter; ++it) {
        double tmid = 0.5 * (t_lo + t_hi);
        CellId mid = find_cell_for_point(ctx, numerics::lerp3d(p0, p1, tmid), hint);
        if (mid == lo_cell) {
            t_lo = tmid;
        } else {
            t_hi = tmid;
            if (mid.valid())
                hint = mid;
        }
    }
    return 0.5 * (t_lo + t_hi);
}

// Process a segment using sample-and-bisect. Used when the analytic ray path
// can't proceed (degenerate cell / no current cell). Updates the open record
// state and emits any closed records via `close_record`.
template<typename CloseFn>
void
sample_segment_fallback(LocateCtx &ctx,
                        const xyz::Point &p0,
                        const xyz::Point &p1,
                        double md0,
                        double md1,
                        double t_start_seg,
                        std::optional<OpenRecord> &open,
                        CellId &previous_cell,
                        double sampling_step,
                        int refine_iters,
                        CloseFn close_record)
{
    // Clip the remaining segment to the grid AABB so we only sample inside
    // the grid. Without this, a long trajectory tail outside the grid still
    // costs O(seg_len/step) work per segment.
    auto [t_in, t_out] = clip_segment_to_aabb(p0, p1, ctx.grid_min, ctx.grid_max);
    if (t_in > t_out)
        return;
    double t_lo_eff = std::max(t_start_seg, t_in);
    double t_hi_eff = std::min(1.0, t_out);
    if (t_lo_eff >= t_hi_eff - T_EPS)
        return;

    const double seg_len = (p1 - p0).norm() * (t_hi_eff - t_lo_eff);
    const size_t nsub =
      std::max<size_t>(1, static_cast<size_t>(std::ceil(seg_len / sampling_step)));

    double t_prev = t_lo_eff;
    // If we entered via the bbox clip (well was outside), the cell at t_prev
    // is by construction invalid.
    CellId cell_prev = open.has_value() ? open->cell : CellId{};

    for (size_t s = 1; s <= nsub; ++s) {
        const double frac = static_cast<double>(s) / static_cast<double>(nsub);
        const double t_curr = t_lo_eff + (t_hi_eff - t_lo_eff) * frac;
        const xyz::Point pt_curr = numerics::lerp3d(p0, p1, t_curr);

        // Prefer the freshly-tracked cell_prev as the hint: it is the closest
        // cell we just touched, whereas previous_cell may be many segments stale.
        const CellId &hint = cell_prev.valid() ? cell_prev : previous_cell;
        const CellId cell_curr = find_cell_for_point(ctx, pt_curr, hint);

        if (cell_curr != cell_prev) {
            const double t_star = bisect_cell_boundary(
              ctx, p0, p1, t_prev, t_curr, cell_prev, cell_curr, refine_iters);
            const xyz::Point p_star = numerics::lerp3d(p0, p1, t_star);
            const double md_star = generic::lerp(md0, md1, t_star);

            if (open.has_value() && cell_prev.valid()) {
                close_record(*open, p_star, md_star);
                open.reset();
            }
            if (cell_curr.valid()) {
                open = OpenRecord{ cell_curr, p_star, md_star };
            }
        }

        t_prev = t_curr;
        cell_prev = cell_curr;
        if (cell_curr.valid())
            previous_cell = cell_curr;
    }
}

// -----------------------------------------------------------------------------
// Inactive-cell ride-through: when the analytic march exits an active cell
// into an inactive one, we ride through inactive territory (without opening
// records) until we either re-enter an active cell, leave the grid, or hit
// a cell that needs the sampling fallback.
// -----------------------------------------------------------------------------

enum class RideOutcome
{
    ReenteredActive,  // open contains a new record at t_now
    LeftGrid,         // segment terminates without re-entry
    EndOfSegment,     // stayed inactive to t_now == 1
    Fallback          // hit a non-convex cell; caller should run fallback
};

RideOutcome
ride_inactive_cells(LocateCtx &ctx,
                    const xyz::Point &p0,
                    const xyz::Point &p1,
                    const xyz::Point &dir,
                    double md0,
                    double md1,
                    CellId start,
                    double t_start,
                    std::optional<OpenRecord> &open,
                    CellId &previous_cell,
                    double &t_now)
{
    CellId riding = start;
    double t_ride = t_start;
    while (true) {
        if (!ctx.in_grid(riding.i, riding.j, riding.k)) {
            t_now = t_ride;
            previous_cell = riding;
            return RideOutcome::LeftGrid;
        }
        const CellCorners &cc = ctx.cell(riding.i, riding.j, riding.k);
        if (ctx.needs_fallback(riding, cc)) {
            t_now = t_ride;
            previous_cell = riding;
            return RideOutcome::Fallback;
        }
        auto [t2, f2] = ray_cell_exit(p0, dir, cc, t_ride);
        if (f2 < 0 || std::isnan(t2)) {
            t_now = t_ride;
            previous_cell = riding;
            return RideOutcome::Fallback;
        }
        if (t2 >= 1.0 - T_EPS) {
            t_now = 1.0;
            previous_cell = riding;
            return RideOutcome::EndOfSegment;
        }
        const auto &dn = FACE_NEIGHBOUR[f2];
        CellId next{ riding.i + dn[0], riding.j + dn[1], riding.k + dn[2] };
        if (!ctx.in_grid(next.i, next.j, next.k)) {
            t_now = t2;
            previous_cell = riding;
            return RideOutcome::LeftGrid;
        }
        if (ctx.cell_active(next.i, next.j, next.k)) {
            open = OpenRecord{ next, numerics::lerp3d(p0, p1, t2),
                               generic::lerp(md0, md1, t2) };
            previous_cell = next;
            t_now = t2;
            return RideOutcome::ReenteredActive;
        }
        riding = next;
        t_ride = t2;
    }
}

}  // namespace

// =====================================================================================
// Public API — see file header.
// =====================================================================================

py::dict
compute_well_cell_intersections(const Grid &grid,
                                const py::array_t<double> &xv,
                                const py::array_t<double> &yv,
                                const py::array_t<double> &zv,
                                const py::array_t<double> &mdv,
                                const double sampling_step = 1.0,
                                const int refine_iters = 20,
                                const bool active_only = false,
                                const geometry::PointInHexahedronMethod method =
                                  geometry::PointInHexahedronMethod::Optimized)
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("compute_well_cell_intersections");

    if (xv.ndim() != 1 || yv.ndim() != 1 || zv.ndim() != 1 || mdv.ndim() != 1) {
        throw std::invalid_argument(
          "xv, yv, zv, mdv must all be 1D numpy arrays of equal length");
    }
    const auto n = xv.shape(0);
    if (yv.shape(0) != n || zv.shape(0) != n || mdv.shape(0) != n) {
        throw std::invalid_argument("xv, yv, zv, mdv must all have the same length");
    }
    if (sampling_step <= 0.0) {
        throw std::invalid_argument("sampling_step must be > 0");
    }
    if (refine_iters < 0) {
        throw std::invalid_argument("refine_iters must be >= 0");
    }

    auto xv_ = xv.unchecked<1>();
    auto yv_ = yv.unchecked<1>();
    auto zv_ = zv.unchecked<1>();
    auto mdv_ = mdv.unchecked<1>();
    auto actnumsv_ = grid.get_actnumsv().unchecked<3>();
    const auto [grid_min, grid_max] = grid.get_bounding_box();

    ColumnXYBBoxCache col_cache;
    LocateCtx ctx{
        grid,
        actnumsv_,
        method,
        active_only,
        grid_min,
        grid_max,
        col_cache,
        static_cast<int>(grid.get_ncol()),
        static_cast<int>(grid.get_nrow()),
        static_cast<int>(grid.get_nlay()),
        grid.get_cell_corners_cache(),
    };

    // Output buffers
    std::vector<int32_t> out_i, out_j, out_k;
    std::vector<double> out_ex, out_ey, out_ez, out_emd;
    std::vector<double> out_xx, out_xy, out_xz, out_xmd;

    auto close_record = [&](const OpenRecord &rec, const xyz::Point &exit_xyz,
                            double exit_md) {
        out_i.push_back(rec.cell.i);
        out_j.push_back(rec.cell.j);
        out_k.push_back(rec.cell.k);
        out_ex.push_back(rec.entry_xyz.x());
        out_ey.push_back(rec.entry_xyz.y());
        out_ez.push_back(rec.entry_xyz.z());
        out_emd.push_back(rec.entry_md);
        out_xx.push_back(exit_xyz.x());
        out_xy.push_back(exit_xyz.y());
        out_xz.push_back(exit_xyz.z());
        out_xmd.push_back(exit_md);
    };

    std::optional<OpenRecord> open;
    CellId previous_cell;

    if (n < 2) {
        logger.debug("Trajectory has fewer than 2 samples; returning empty result.");
    }

    for (pybind11::ssize_t seg = 0; seg + 1 < n; ++seg) {
        const xyz::Point p0(xv_(seg), yv_(seg), zv_(seg));
        const xyz::Point p1(xv_(seg + 1), yv_(seg + 1), zv_(seg + 1));
        const xyz::Point dir = p1 - p0;
        const double md0 = mdv_(seg);
        const double md1 = mdv_(seg + 1);

        // ----- Phase 0: ensure we know what cell t=0 is in (only at seg 0) -----
        if (seg == 0) {
            CellId c0 = find_cell_for_point(ctx, p0, previous_cell);
            if (c0.valid()) {
                open = OpenRecord{ c0, p0, md0 };
                previous_cell = c0;
            }
        }

        double t_now = 0.0;
        bool fall_back = false;

        // ----- Phase 1: ray-tracing fast path -----
        // March cell-to-cell while we have an open record. Terminate when
        // t_now reaches 1.0, when we hit a non-convex cell (-> fallback), or
        // when the next cell is outside the grid.
        while (open.has_value() && t_now < 1.0 - T_EPS) {
            const CellCorners &cc = ctx.cell(open->cell.i, open->cell.j, open->cell.k);

            if (ctx.needs_fallback(open->cell, cc)) {
                fall_back = true;
                break;
            }

            auto [t_exit, exit_face] = ray_cell_exit(p0, dir, cc, t_now);
            if (exit_face < 0 || std::isnan(t_exit)) {
                fall_back = true;
                break;
            }

            if (t_exit >= 1.0 - T_EPS) {
                // Exit beyond end of segment — record stays open across segments.
                t_now = 1.0;
                break;
            }

            // Close the current record at the exit point.
            const xyz::Point p_exit = numerics::lerp3d(p0, p1, t_exit);
            const double md_exit = generic::lerp(md0, md1, t_exit);
            close_record(*open, p_exit, md_exit);
            const CellId exited = open->cell;
            open.reset();

            // Determine the next cell across the exit face.
            const auto &dn = FACE_NEIGHBOUR[exit_face];
            CellId next{ exited.i + dn[0], exited.j + dn[1], exited.k + dn[2] };

            if (!ctx.in_grid(next.i, next.j, next.k)) {
                t_now = t_exit;
                previous_cell = exited;
                break;
            }
            if (active_only && !ctx.cell_active(next.i, next.j, next.k)) {
                // Ride through the inactive block without opening records.
                auto outcome = ride_inactive_cells(ctx, p0, p1, dir, md0, md1, next,
                                                   t_exit, open, previous_cell, t_now);
                if (outcome == RideOutcome::Fallback) {
                    fall_back = true;
                    break;
                }
                if (outcome == RideOutcome::ReenteredActive)
                    continue;
                break;  // LeftGrid or EndOfSegment
            }

            // Open the new active cell record at the same crossing point.
            open = OpenRecord{ next, p_exit, md_exit };
            previous_cell = next;
            t_now = t_exit;
        }

        // ----- Phase 2: fallback (sampling) for the remainder of the segment
        // Run when phase 1 bailed out, or when there's no open record and the
        // segment is not fully consumed (between cells / outside the grid).
        if (fall_back || (!open.has_value() && t_now < 1.0 - T_EPS)) {
            sample_segment_fallback(ctx, p0, p1, md0, md1, t_now, open, previous_cell,
                                    sampling_step, refine_iters, close_record);
        }
    }

    // Close any record that is still open at the trajectory end.
    if (open.has_value()) {
        const xyz::Point p_last(xv_(n - 1), yv_(n - 1), zv_(n - 1));
        close_record(*open, p_last, mdv_(n - 1));
    }

    // Zero-copy: move each vector onto the heap and let pybind11 reference
    // its buffer through a capsule that frees it when the numpy array dies.
    auto to_numpy = [](auto &&v) {
        using Vec = std::remove_reference_t<decltype(v)>;
        using T = typename Vec::value_type;
        auto *vec = new Vec(std::move(v));
        auto cap = py::capsule(vec, [](void *p) { delete static_cast<Vec *>(p); });
        return py::array_t<T>(static_cast<py::ssize_t>(vec->size()), vec->data(), cap);
    };

    py::dict result;
    result["i"] = to_numpy(std::move(out_i));
    result["j"] = to_numpy(std::move(out_j));
    result["k"] = to_numpy(std::move(out_k));
    result["entry_x"] = to_numpy(std::move(out_ex));
    result["entry_y"] = to_numpy(std::move(out_ey));
    result["entry_z"] = to_numpy(std::move(out_ez));
    result["entry_md"] = to_numpy(std::move(out_emd));
    result["exit_x"] = to_numpy(std::move(out_xx));
    result["exit_y"] = to_numpy(std::move(out_xy));
    result["exit_z"] = to_numpy(std::move(out_xz));
    result["exit_md"] = to_numpy(std::move(out_xmd));
    return result;
}

}  // namespace xtgeo::grid3d
