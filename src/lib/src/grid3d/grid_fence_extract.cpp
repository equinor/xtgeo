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

#ifdef XTGEO_USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace xtgeo::grid3d {

constexpr size_t INVALID = std::numeric_limits<size_t>::max();

const geometry::PointInHexahedronMethod point_in_hex_method =
  geometry::PointInHexahedronMethod::Tetrahedrons;  // Use Tetrahedrons method for this
                                                    // purpose, since it is faster

/**
 * Given a fencespec fspec with shape (N, 5), a z_vector with shape (M,),
 * and a property with shape (NCOL, NROW, NLAY), this function computes the grid fence
 * by iterating over each point in fspec and checking the corresponding property
 * values at the specified z_vector levels. The function returns a numpy array
 * with shape (N, M) containing the property values at each point in fspec (need
 * transpose in python to M, N)
 */
py::array_t<double>
get_grid_fence(const Grid &grd,
               const Grid &one_grid,
               const py::array_t<double> &fspec,
               const py::array_t<double> &property,
               const py::array_t<double> &z_vector,
               const regsurf::RegularSurface &top_i,
               const regsurf::RegularSurface &top_j,
               const regsurf::RegularSurface &base_i,
               const regsurf::RegularSurface &base_j,
               const regsurf::RegularSurface &top_d,
               const regsurf::RegularSurface &base_d,
               const double threshold_magic)
{

    auto &logger = xtgeo::logging::LoggerManager::get("get_grid_fence");
    logger.debug("Extracting grid fence from grid, fspec, property and z_vector");
    auto fspec_ = fspec.unchecked<2>();
    auto property_ = property.unchecked<3>();
    auto z_vector_ = z_vector.unchecked<1>();

    py::array_t<double> result({ fspec_.shape(0), z_vector_.shape(0) });
    auto result_ = result.mutable_unchecked<2>();

    logger.debug("Iterating over fspec points to extract grid fence");
    std::vector<xyz::Point> points(z_vector_.shape(0));

    for (size_t i = 0; i < fspec_.shape(0); i++) {
        double x = fspec_(i, 0);
        double y = fspec_(i, 1);

        // Fill points vector in-place
        for (size_t k = 0; k < z_vector_.shape(0); k++) {
            points[k] = xyz::Point(x, y, z_vector_(k));
        }
        xyz::PointSet pset(points);

        py::array_t<int> i_idx, j_idx, k_idx;
        std::tie(i_idx, j_idx, k_idx) = grid3d::get_indices_from_pointset(
          grd, pset, one_grid, top_i, top_j, base_i, base_j, top_d, base_d,
          threshold_magic, false, point_in_hex_method);

        auto i_idx_ = i_idx.unchecked<1>();
        auto j_idx_ = j_idx.unchecked<1>();
        auto k_idx_ = k_idx.unchecked<1>();

        for (size_t k = 0; k < z_vector_.shape(0); k++) {
            auto ii = i_idx_[k];
            auto jj = j_idx_[k];
            auto kk = k_idx_[k];
            if (ii != INVALID) {
                result_(i, k) = property_(ii, jj, kk);
            } else {
                result_(i, k) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }

    logger.debug("Finished extracting grid fence");
    return result;
}

}  // namespace xtgeo::grid3d
