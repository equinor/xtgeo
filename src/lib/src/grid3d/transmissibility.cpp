/*
 * compute_transmissibilities — TPFA transmissibility computation for corner-point
 * grids.
 *
 * Algorithm
 * =========
 * TRAN arrays (tranx, trany, tranz) hold the same-K direct IJK connections.
 * NNCs are emitted for:
 *   - Fault NNCs:   East/West/North/South face pairs where k' != k (fault throw spans
 *                   multiple K layers).  Found via a Z-range scan of the adjacent
 * column.
 *   - Pinchout NNCs: Connections skipping one or more inactive (or collapsed) K layers
 *                   inside a column.
 *
 * Each active cell pair is visited exactly once:
 *   - I-direction: cell (i,j,k) → (i+1,j,k') for all k' with Z-range overlap
 *   - J-direction: cell (i,j,k) → (i,j+1,k') for all k' with Z-range overlap
 *   - K-direction: cell (i,j,k) → (i,j,k+1)  direct connection only
 *   - Pinchout:    cell (i,j,k) → (i,j,k')    first active after inactive run
 *
 * Effective permeability
 * ======================
 *   HT_i = k_eff_i * A / d_i
 *   T    = HT_1 * HT_2 / (HT_1 + HT_2)
 *
 * k_eff = perm * ntg  for I and J;  k_eff = permz  for K (NTG not applied vertically).
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

inline double
tpfa(double k1, double k2, double area, double d1, double d2)
{
    if (area <= 0.0 || d1 <= 0.0 || d2 <= 0.0)
        return 0.0;
    double ht1 = k1 * area / d1;
    double ht2 = k2 * area / d2;
    double denom = ht1 + ht2;
    return (denom > 0.0) ? ht1 * ht2 / denom : 0.0;
}

// Z range (min, max) of a CellCorners object.
inline std::pair<double, double>
cell_z_range(const CellCorners &c)
{
    double zmin =
      std::min({ c.upper_sw.z(), c.upper_se.z(), c.upper_nw.z(), c.upper_ne.z(),
                 c.lower_sw.z(), c.lower_se.z(), c.lower_nw.z(), c.lower_ne.z() });
    double zmax =
      std::max({ c.upper_sw.z(), c.upper_se.z(), c.upper_nw.z(), c.upper_ne.z(),
                 c.lower_sw.z(), c.lower_se.z(), c.lower_nw.z(), c.lower_ne.z() });
    return { zmin, zmax };
}

// Average Z of the 4 upper corners (approximately the Top face centroid Z).
inline double
cell_top_z(const CellCorners &c)
{
    return (c.upper_sw.z() + c.upper_se.z() + c.upper_nw.z() + c.upper_ne.z()) * 0.25;
}

// Average Z of the 4 lower corners.
inline double
cell_bot_z(const CellCorners &c)
{
    return (c.lower_sw.z() + c.lower_se.z() + c.lower_nw.z() + c.lower_ne.z()) * 0.25;
}

// Per-column cache: precomputed corners and Z ranges.
struct ColumnCache
{
    size_t nl;
    std::vector<CellCorners> corners;
    std::vector<bool> active;
    std::vector<double> zlo, zhi;  // cell Z range; NaN for inactive cells

    ColumnCache(const Grid &grid,
                size_t i,
                size_t j,
                const py::detail::unchecked_reference<int, 3> &act) :
      nl(grid.get_nlay()), corners(nl), active(nl, false),
      zlo(nl, std::numeric_limits<double>::quiet_NaN()),
      zhi(nl, std::numeric_limits<double>::quiet_NaN())
    {
        for (size_t k = 0; k < nl; k++) {
            if (act(i, j, k) == 0)
                continue;
            active[k] = true;
            corners[k] = get_cell_corners_from_ijk(grid, i, j, k);
            auto [z0, z1] = cell_z_range(corners[k]);
            zlo[k] = z0;
            zhi[k] = z1;
        }
    }
};

struct NNCRec
{
    int32_t i1, j1, k1, i2, j2, k2;
    double T;
    int32_t type;  // NNCType value
};

// ---------------------------------------------------------------------------
// Column-pair scan for I or J direction (fault NNC detection)
// ---------------------------------------------------------------------------
//
// For each active cell k in col1, scan col2 for all cells k2 with Z-range
// overlap.  k2 == k → TRAN array; k2 != k → fault NNC.
//
// face1_label:  the face of col1 cells that touches col2 (East or North)
// face2_label:  the face of col2 cells that touches col1 (West or South)
// tran:         the output TRAN array slice for this column pair
// nncs:         NNC accumulator
// perm1, perm2: effective per-cell permeability arrays for this column pair
//               (permX * ntg for horizontal directions)
template<typename PermProxy1, typename PermProxy2>
void
scan_column_pair(const ColumnCache &col1,
                 const ColumnCache &col2,
                 CellFaceLabel face1_label,
                 CellFaceLabel face2_label,
                 double *tran,        // length nl; NULL means direction not stored here
                 size_t tran_stride,  // 1 (the k index maps directly)
                 const PermProxy1 &px1,  // callable(k) -> effective perm for col1
                 const PermProxy2 &px2,  // callable(k) -> effective perm for col2
                 std::vector<NNCRec> &nncs,
                 int32_t i1,
                 int32_t j1,
                 int32_t i2,
                 int32_t j2)
{
    const size_t nl = col1.nl;

    for (size_t k = 0; k < nl; k++) {
        if (!col1.active[k])
            continue;

        const double z1lo = col1.zlo[k];
        const double z1hi = col1.zhi[k];

        // Check same-K direct neighbour first
        if (col2.active[k]) {
            auto fr = face_overlap_result(col1.corners[k], face1_label, col2.corners[k],
                                          face2_label);
            if (fr.area > 0.0 && tran != nullptr) {
                tran[k * tran_stride] = tpfa(px1(k), px2(k), fr.area, fr.d1, fr.d2);
            }
            // Note: same-K pair with area > 0 IS put in TRAN, not NNCs.
            // (Even for faulted grids the same-K connection may be valid.)
        }

        // Scan UPWARD in col2 (k2 < k): shallower cells.
        // As k2 decreases, col2 cells get shallower (smaller Z).
        // Break only when col2 cell is entirely above col1 cell (col2.zhi < z1lo),
        // because further decrease of k2 (shallower cells) cannot overlap again.
        // If col2 cell is entirely below col1 cell (col2.zlo > z1hi), skip it but
        // keep scanning — a large-throw fault can place col2's shallow cells at
        // the same depth as col1's deep cell, so deeper col2 entries come first.
        for (int k2 = static_cast<int>(k) - 1; k2 >= 0; k2--) {
            // For inactive cells zlo/zhi are NaN; skip them but don't break.
            if (!std::isnan(col2.zhi[k2]) && col2.zhi[k2] < z1lo)
                break;  // col2 cell entirely shallower; all smaller k2 also shallower
            if (!std::isnan(col2.zlo[k2]) && col2.zlo[k2] > z1hi)
                continue;  // col2 cell entirely deeper; going up may reach overlap
            if (!col2.active[k2])
                continue;

            auto fr = face_overlap_result(col1.corners[k], face1_label,
                                          col2.corners[k2], face2_label);
            if (fr.area <= 0.0)
                continue;

            double T = tpfa(px1(k), px2(k2), fr.area, fr.d1, fr.d2);
            nncs.push_back({ i1, j1, static_cast<int32_t>(k), i2, j2,
                             static_cast<int32_t>(k2), T,
                             static_cast<int32_t>(NNCType::Fault) });
        }

        // Scan DOWNWARD in col2 (k2 > k): deeper cells.
        // As k2 increases, col2 cells get deeper (larger Z).
        // Break only when col2 cell is entirely below col1 cell (col2.zlo > z1hi).
        // If col2 cell is entirely above col1 cell (col2.zhi < z1lo), skip it but
        // keep scanning — a large-throw fault can place col2's deep cells at the
        // same depth as col1's shallow cell.
        for (size_t k2 = k + 1; k2 < nl; k2++) {
            if (!std::isnan(col2.zlo[k2]) && col2.zlo[k2] > z1hi)
                break;  // col2 cell entirely deeper; all larger k2 also deeper
            if (!std::isnan(col2.zhi[k2]) && col2.zhi[k2] < z1lo)
                continue;  // col2 cell entirely shallower; going down may reach overlap
            if (!col2.active[k2])
                continue;

            auto fr = face_overlap_result(col1.corners[k], face1_label,
                                          col2.corners[k2], face2_label);
            if (fr.area <= 0.0)
                continue;

            double T = tpfa(px1(k), px2(k2), fr.area, fr.d1, fr.d2);
            nncs.push_back({ i1, j1, static_cast<int32_t>(k), i2, j2,
                             static_cast<int32_t>(k2), T,
                             static_cast<int32_t>(NNCType::Fault) });
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

TransmissibilityResult
compute_transmissibilities(const Grid &grid,
                           const py::array_t<double> &permx,
                           const py::array_t<double> &permy,
                           const py::array_t<double> &permz,
                           const py::array_t<double> &ntg,
                           double /* min_dz_pinchout */)
{
    const size_t nc = grid.get_ncol();
    const size_t nr = grid.get_nrow();
    const size_t nl = grid.get_nlay();

    auto px = permx.unchecked<3>();
    auto py_ = permy.unchecked<3>();
    auto pz = permz.unchecked<3>();
    auto nt = ntg.unchecked<3>();
    auto act = grid.get_actnumsv().unchecked<3>();

    const double NaN = std::numeric_limits<double>::quiet_NaN();

    // ---------------------------------------------------------------
    // Allocate output TRAN arrays (NaN → inactive pair)
    // ---------------------------------------------------------------
    auto make_nan = [&](py::ssize_t d0, py::ssize_t d1, py::ssize_t d2) {
        py::array_t<double> arr(std::vector<py::ssize_t>{ d0, d1, d2 });
        std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(), NaN);
        return arr;
    };
    auto tranx_arr =
      make_nan(static_cast<py::ssize_t>(nc > 0 ? nc - 1 : 0),
               static_cast<py::ssize_t>(nr), static_cast<py::ssize_t>(nl));
    auto trany_arr = make_nan(static_cast<py::ssize_t>(nc),
                              static_cast<py::ssize_t>(nr > 0 ? nr - 1 : 0),
                              static_cast<py::ssize_t>(nl));
    auto tranz_arr =
      make_nan(static_cast<py::ssize_t>(nc), static_cast<py::ssize_t>(nr),
               static_cast<py::ssize_t>(nl > 0 ? nl - 1 : 0));

    auto tx = tranx_arr.mutable_unchecked<3>();
    auto ty = trany_arr.mutable_unchecked<3>();
    auto tz = tranz_arr.mutable_unchecked<3>();

    std::vector<NNCRec> nncs;

    // ---------------------------------------------------------------
    // I-direction: cell (i,j,k) → (i+1,j,k')
    // ---------------------------------------------------------------
    for (size_t i = 0; i + 1 < nc; i++) {
        for (size_t j = 0; j < nr; j++) {
            ColumnCache col1(grid, i, j, act);
            ColumnCache col2(grid, i + 1, j, act);

            // tran slice: tranx[i,j,*] — pointer to the k-contiguous strip
            double *tran_slice = &tx(i, j, 0);

            scan_column_pair(
              col1, col2, CellFaceLabel::East, CellFaceLabel::West, tran_slice, 1,
              [&](size_t k) { return px(i, j, k) * nt(i, j, k); },
              [&](size_t k) { return px(i + 1, j, k) * nt(i + 1, j, k); }, nncs,
              static_cast<int32_t>(i), static_cast<int32_t>(j),
              static_cast<int32_t>(i + 1), static_cast<int32_t>(j));
        }
    }

    // ---------------------------------------------------------------
    // J-direction: cell (i,j,k) → (i,j+1,k')
    // ---------------------------------------------------------------
    for (size_t i = 0; i < nc; i++) {
        for (size_t j = 0; j + 1 < nr; j++) {
            ColumnCache col1(grid, i, j, act);
            ColumnCache col2(grid, i, j + 1, act);

            double *tran_slice = &ty(i, j, 0);

            scan_column_pair(
              col1, col2, CellFaceLabel::North, CellFaceLabel::South, tran_slice, 1,
              [&](size_t k) { return py_(i, j, k) * nt(i, j, k); },
              [&](size_t k) { return py_(i, j + 1, k) * nt(i, j + 1, k); }, nncs,
              static_cast<int32_t>(i), static_cast<int32_t>(j), static_cast<int32_t>(i),
              static_cast<int32_t>(j + 1));
        }
    }

    // ---------------------------------------------------------------
    // K-direction: cell (i,j,k) → (i,j,k+1) — standard connection
    // ---------------------------------------------------------------
    for (size_t i = 0; i < nc; i++) {
        for (size_t j = 0; j < nr; j++) {
            for (size_t k = 0; k + 1 < nl; k++) {
                if (act(i, j, k) == 0 || act(i, j, k + 1) == 0)
                    continue;
                auto c1 = get_cell_corners_from_ijk(grid, i, j, k);
                auto c2 = get_cell_corners_from_ijk(grid, i, j, k + 1);
                auto fr =
                  face_overlap_result(c1, CellFaceLabel::Bottom, c2, CellFaceLabel::Top,
                                      std::numeric_limits<double>::max());
                if (fr.area > 0.0) {
                    tz(i, j, k) =
                      tpfa(pz(i, j, k), pz(i, j, k + 1), fr.area, fr.d1, fr.d2);
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // K pinch-out NNCs: within each column, connect the first active
    // cell above and below each inactive run.
    // ---------------------------------------------------------------
    for (size_t i = 0; i < nc; i++) {
        for (size_t j = 0; j < nr; j++) {
            // Walk the column; when we hit an inactive cell after an active
            // one, scan forward to find the next active cell.
            for (size_t k = 0; k + 1 < nl; k++) {
                if (act(i, j, k) == 0)
                    continue;
                // Check if immediately below is inactive (potential pinch-out)
                if (act(i, j, k + 1) != 0)
                    continue;

                // Find next active cell below k
                size_t k2 = k + 2;
                while (k2 < nl && act(i, j, k2) == 0)
                    k2++;
                if (k2 >= nl)
                    continue;  // no active cell found below

                auto c1 = get_cell_corners_from_ijk(grid, i, j, k);
                auto c2 = get_cell_corners_from_ijk(grid, i, j, k2);

                // Use infinite max_normal_gap: faces ARE separated (inactive gap).
                auto fr =
                  face_overlap_result(c1, CellFaceLabel::Bottom, c2, CellFaceLabel::Top,
                                      std::numeric_limits<double>::max());
                if (fr.area <= 0.0)
                    continue;

                double T = tpfa(pz(i, j, k), pz(i, j, k2), fr.area, fr.d1, fr.d2);
                nncs.push_back({ static_cast<int32_t>(i), static_cast<int32_t>(j),
                                 static_cast<int32_t>(k), static_cast<int32_t>(i),
                                 static_cast<int32_t>(j), static_cast<int32_t>(k2), T,
                                 static_cast<int32_t>(NNCType::Pinchout) });
            }
        }
    }

    // ---------------------------------------------------------------
    // Pack NNC vectors into numpy arrays
    // ---------------------------------------------------------------
    const size_t n = nncs.size();
    auto nn = static_cast<py::ssize_t>(n);
    py::array_t<int32_t> ni1({ nn }), nj1({ nn }), nk1({ nn });
    py::array_t<int32_t> ni2({ nn }), nj2({ nn }), nk2({ nn });
    py::array_t<double> nT({ nn });
    py::array_t<int32_t> ntype({ nn });

    for (size_t m = 0; m < n; m++) {
        ni1.mutable_at(m) = nncs[m].i1;
        nj1.mutable_at(m) = nncs[m].j1;
        nk1.mutable_at(m) = nncs[m].k1;
        ni2.mutable_at(m) = nncs[m].i2;
        nj2.mutable_at(m) = nncs[m].j2;
        nk2.mutable_at(m) = nncs[m].k2;
        nT.mutable_at(m) = nncs[m].T;
        ntype.mutable_at(m) = nncs[m].type;
    }

    return { tranx_arr, trany_arr, tranz_arr, ni1, nj1, nk1, ni2, nj2, nk2, nT, ntype };
}

}  // namespace xtgeo::grid3d
