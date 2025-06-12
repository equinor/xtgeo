#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cstddef>
#include <tuple>
#include <unordered_map>
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

    // Process each point
    xyz::Point previous_point(0.0, 0.0, 0.0);
    for (size_t idx = 0; idx < num_points; ++idx) {
        const auto &point = points.get_point(idx);

        if (!is_point_in_grid_bounds(point, min_grid_point, max_grid_point)) {
            continue;
        }
        if (!is_within_depth(point, top_d, base_d, threshold_magic)) {
            continue;
        }

        // compute the Euclidean distance to the previous point
        double distance_to_previous = (point - previous_point).norm();

        if (distance_to_previous < threshold_magic && previous_i != INVALID) {
            // If we have a previous point, use its i,j as starting point
            // This is an optimization for organized pointsets, meaning that next
            // point is likely close to the previous one

            bool inner_found = false;
            for (int radius = 0; radius <= MAX_RADIUS; ++radius) {
                auto [new_i, new_j, new_k] = fast_lookup_neighbours(
                  grid, point, previous_i, previous_j, previous_k, radius, active_only,
                  actnumsv_, point_in_hex_method);

                if (new_i != INVALID) {
                    i_indices_(idx) = static_cast<int>(new_i);
                    j_indices_(idx) = static_cast<int>(new_j);
                    k_indices_(idx) = static_cast<int>(new_k);
                    previous_i = new_i;
                    previous_j = new_j;
                    previous_k = new_k;
                    inner_found = true;
                    previous_point = point;  // Update previous point for next iteration
                    break;                   // Found a match, no need to search further
                }
            }
            if (inner_found) {
                continue;  // Skip to next point since we found a match
            }
        }

        // Get potential i,j range for the point
        auto [imin, imax, jmin, jmax] = estimate_ij_range(
          point, top_i, top_j, base_i, base_j, grid.get_ncol(), grid.get_nrow());

        if (imin == INVALID) {  // Point outside grid (all NaN values from surfaces)
            previous_k = 0;
            continue;
        }

        // Try to find a proposed column for efficiency
        auto [i_est, j_est] =
          get_proposed_ij(one_grid, point, imin, imax, jmin, jmax, point_in_hex_method);

        // If we found a proposed column, restrict search to just that
        // column
        if (i_est != INVALID && j_est != INVALID) {
            imin = imax = i_est;
            jmin = jmax = j_est;
        }

        // Search for the point in the possible cell columns
        bool found = false;
        for (size_t i = imin; i <= imax && !found; ++i) {
            for (size_t j = jmin; j <= jmax && !found; ++j) {

                int found_i = -1, found_j = -1, found_k = -1;
                found =
                  find_in_column(grid, point, i, j, previous_k, found_i, found_j,
                                 found_k, active_only, actnumsv_, point_in_hex_method);
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

        // Reset previous_k if no cell found
        if (!found) {
            previous_i = INVALID;
            previous_j = INVALID;
            previous_k = 0;
        }
        previous_point = point;  // Update previous point for next iteration
    }

    return std::make_tuple(i_indices, j_indices, k_indices);
}

}  // namespace xtgeo::grid3d
