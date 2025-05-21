#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/types.hpp>

namespace py = pybind11;
namespace xtgeo::grid3d {

// Constants for array dimensions
constexpr size_t COORDS_DIM = 3;
constexpr size_t CORNERS_DIM = 4;

std::tuple<py::array_t<double>,
           py::array_t<double>,
           py::array_t<double>,
           py::array_t<bool>>
convert_xtgeo_to_rmsapi(const Grid &grd)
{
    auto &logger =
      xtgeo::logging::LoggerManager::get("xtgeo.grid3d.convert_xtgeo_to_rmsapi");
    logger.debug("Converting XTGeo grid to RMSAPI grid layout");

    auto coordsv = grd.coordsv.unchecked<3>();
    auto zcornsv = grd.zcornsv.unchecked<4>();
    auto actnumsv = grd.actnumsv.unchecked<3>();

    const size_t nncol = grd.ncol + 1;
    const size_t nnrow = grd.nrow + 1;
    const size_t nnlay = grd.nlay + 1;
    const double undef = 1e32;

    // Pre-allocate arrays with correct sizes
    py::array_t<double> tpillars({ nncol, nnrow, COORDS_DIM });
    py::array_t<double> bpillars({ nncol, nnrow, COORDS_DIM });
    py::array_t<double> zcorners({ nncol, nnrow, CORNERS_DIM, nnlay });
    py::array_t<bool> zmask({ nncol, nnrow, CORNERS_DIM, nnlay });

    auto tpillars_ = tpillars.mutable_unchecked<3>();
    auto bpillars_ = bpillars.mutable_unchecked<3>();
    auto zcorners_ = zcorners.mutable_unchecked<4>();
    auto zmask_ = zmask.mutable_unchecked<4>();

    // Initialize zmask to false
    std::fill(zmask.mutable_data(), zmask.mutable_data() + zmask.size(), false);

    bool pillar_warn = false;
    bool zcross_warn = false;

    // loop rows and columns
    for (size_t icol = 0; icol < nncol; icol++) {
        for (size_t jrow = 0; jrow < nnrow; jrow++) {
            for (size_t nn = 0; nn < 6; nn++) {
                if (nn < 3) {
                    tpillars_(icol, jrow, nn) = coordsv(icol, jrow, nn);
                } else {
                    bpillars_(icol, jrow, nn - 3) = coordsv(icol, jrow, nn);
                }
            }
            // Handle equal Z coordinates which may cause issues in some cases
            if (std::abs(bpillars_(icol, jrow, 2) - tpillars_(icol, jrow, 2)) <
                numerics::TOLERANCE) {
                bpillars_(icol, jrow, 2) += numerics::TOLERANCE;
                pillar_warn = true;
            }

            for (size_t ic = 0; ic < 4; ic++) {

                double previous_zpillar = -1 * undef;

                for (size_t klay = 0; klay < nnlay; klay++) {

                    double zpillar = static_cast<double>(zcornsv(icol, jrow, klay, ic));

                    /* avoid depths that crosses in depth */
                    if (zpillar < previous_zpillar) {
                        zpillar = previous_zpillar;
                        zcross_warn = true;
                    }

                    previous_zpillar = zpillar;
                    zcorners_(icol, jrow, ic, klay) = zpillar;
                }
            }
        }
    }

    if (pillar_warn == true) {
        // Get Python builtins and warnings modules
        auto builtins = py::module_::import("builtins");
        auto warnings = py::module_::import("warnings");

        warnings.attr("warn")("Equal Z coordinates detected and adjusted to avoid "
                              "issues when converting to RMAPI.",
                              builtins.attr("UserWarning"));
    }

    if (zcross_warn == true) {
        // Get Python builtins and warnings modules
        auto builtins = py::module_::import("builtins");
        auto warnings = py::module_::import("warnings");

        warnings.attr("warn")(
          "One or more ZCORN values are crossing in depth. This is corrected prior "
          "to storing in RMS since the RMSAPI does not allow inconsistent Z values",
          builtins.attr("UserWarning"));
    }

    // From the RMSAPI docs, it is stated that inactive pillar nodes at edges shall be
    // masked. Whether this is strictly necessary is unknown, but done here
    // anyway.
    //
    // e.g at cell 0, 0 corner pillar, SW, SE, NW shall be masked, NE not:
    //
    //           |
    //           | SW
    //    MASKED | UNMASKED
    //         - |---------------------
    //    MASKED | MASKED

    // The API docs also indicates(?) that inactive columns should be masked, but
    // experience tells us that this may trigger problems for e.g. `activate_all()`
    // method. Since this is a bit experimental, a DEVELOPER Boolean _COLUMNS is set
    // false at the current stage.

    bool _COLUMNS = false;  // TODO evaluate later

    for (size_t icol = 0; icol < grd.ncol; icol++) {
        for (size_t jrow = 0; jrow < grd.nrow; jrow++) {
            bool inactive_column = true;
            for (size_t klay = 0; klay < nnlay; klay++) {

                if (icol == 0) {
                    zmask_(icol, jrow, 0, klay) = true;
                    zmask_(icol, jrow, 2, klay) = true;
                    zmask_(icol, jrow + 1, 0, klay) = true;
                    zmask_(icol, jrow + 1, 2, klay) = true;
                }
                if (icol == grd.ncol - 1) {
                    zmask_(icol + 1, jrow, 1, klay) = true;
                    zmask_(icol + 1, jrow, 3, klay) = true;
                    zmask_(icol + 1, jrow + 1, 1, klay) = true;
                    zmask_(icol + 1, jrow + 1, 3, klay) = true;
                }
                if (jrow == 0) {
                    zmask_(icol, jrow, 0, klay) = true;
                    zmask_(icol, jrow, 1, klay) = true;
                    zmask_(icol + 1, jrow, 0, klay) = true;
                    zmask_(icol + 1, jrow, 1, klay) = true;
                }
                if (jrow == nnrow - 1) {
                    zmask_(icol, jrow + 1, 2, klay) = true;
                    zmask_(icol, jrow + 1, 3, klay) = true;
                    zmask_(icol + 1, jrow + 1, 2, klay) = true;
                    zmask_(icol + 1, jrow + 1, 3, klay) = true;
                }

                // now if ALL cells in a column has actnum = 0
                if (klay < nnlay - 1 && actnumsv(icol, jrow, klay) == 1) {
                    inactive_column = false;
                }
            }
            if (_COLUMNS == true && inactive_column == true) {
                for (size_t k = 0; k < nnlay; k++) {
                    zmask_(icol, jrow, 3, k) = true;
                    zmask_(icol + 1, jrow, 2, k) = true;
                    zmask_(icol, jrow + 1, 1, k) = true;
                    zmask_(icol + 1, jrow + 1, 0, k) = true;
                }
            }
        }
    }
    logger.debug("Conversion to RMSAPI grid layout completed");
    return std::make_tuple(tpillars, bpillars, zcorners, zmask);
}

/* A postprocessing of zcornsv array when importing 3D grids from Roxar API to xtgeo*/
void
process_edges_rmsapi(py::array_t<float> zcornsv)
{
    auto zcornsv_ = zcornsv.mutable_unchecked<4>();

    const size_t mcol = zcornsv_.shape(0) - 1;
    const size_t mrow = zcornsv_.shape(1) - 1;
    const size_t nnlay = zcornsv_.shape(2);

    for (size_t k = 0; k < nnlay; k++) {
        // corner i = 0 j = 0
        zcornsv_(0, 0, k, 0) = zcornsv_(0, 0, k, 3);
        zcornsv_(0, 0, k, 1) = zcornsv_(0, 0, k, 3);
        zcornsv_(0, 0, k, 2) = zcornsv_(0, 0, k, 3);

        // corner i = 0 j = mrow
        zcornsv_(0, mrow, k, 0) = zcornsv_(0, mrow, k, 1);
        zcornsv_(0, mrow, k, 2) = zcornsv_(0, mrow, k, 1);
        zcornsv_(0, mrow, k, 3) = zcornsv_(0, mrow, k, 1);

        // corner i = mcol j = 0
        zcornsv_(mcol, 0, k, 0) = zcornsv_(mcol, 0, k, 2);
        zcornsv_(mcol, 0, k, 1) = zcornsv_(mcol, 0, k, 2);
        zcornsv_(mcol, 0, k, 3) = zcornsv_(mcol, 0, k, 2);

        // corner i = mcol j = mrow
        zcornsv_(mcol, mrow, k, 1) = zcornsv_(mcol, mrow, k, 0);
        zcornsv_(mcol, mrow, k, 2) = zcornsv_(mcol, mrow, k, 0);
        zcornsv_(mcol, mrow, k, 3) = zcornsv_(mcol, mrow, k, 0);

        // i == 0 boundary
        for (size_t j = 1; j < mrow; j++) {
            zcornsv_(0, j, k, 2) = zcornsv_(0, j, k, 3);
            zcornsv_(0, j, k, 0) = zcornsv_(0, j, k, 1);
        }

        // i == mcol boundary
        for (size_t j = 1; j < mrow; j++) {
            zcornsv_(mcol, j, k, 3) = zcornsv_(mcol, j, k, 2);
            zcornsv_(mcol, j, k, 1) = zcornsv_(mcol, j, k, 0);
        }

        // j == 0 boundary
        for (size_t i = 1; i < mcol; i++) {
            zcornsv_(i, 0, k, 0) = zcornsv_(i, 0, k, 2);
            zcornsv_(i, 0, k, 1) = zcornsv_(i, 0, k, 3);
        }

        // j == mrow boundary
        for (size_t i = 1; i < mcol; i++) {
            zcornsv_(i, mrow, k, 2) = zcornsv_(i, mrow, k, 0);
            zcornsv_(i, mrow, k, 3) = zcornsv_(i, mrow, k, 1);
        }
    }
}

/**
 * @brief: Given an input cornerpoint grid, return a grid with only top and base
 *         The usage for this is mainly to speed up some operations, like finding
 *         if a point is inside a grid.
 */

Grid
extract_onelayer_grid(const Grid &original_grid)
{
    // Access the original zcornsv array
    auto zcornsv_ = original_grid.zcornsv.unchecked<4>();

    const size_t mcol = zcornsv_.shape(0);
    const size_t mrow = zcornsv_.shape(1);
    const size_t nnlay = zcornsv_.shape(2);
    const size_t ncorners = zcornsv_.shape(3);

    // If the grid has fewer than 2 layers, return a true copy of the original grid
    if (nnlay < 2) {
        Grid copy_grid = original_grid;  // Create a copy of the original grid
        return copy_grid;
    }

    // Create a new zcornsv array with only 2 layers
    std::vector<size_t> zcornsv_shape = { mcol, mrow, 2, ncorners };
    py::array_t<float> new_zcornsv(zcornsv_shape);
    auto new_zcornsv_ = new_zcornsv.mutable_unchecked<4>();

    // Copy the top layer (first layer)
    for (size_t i = 0; i < mcol; i++) {
        for (size_t j = 0; j < mrow; j++) {
            for (size_t c = 0; c < 4; c++) {
                new_zcornsv_(i, j, 0, c) = zcornsv_(i, j, 0, c);
                new_zcornsv_(i, j, 1, c) = zcornsv_(i, j, nnlay - 1, c);
            }
        }
    }

    // Create a new actnumsv array with only 1 layer and set all values to 1
    auto actnum_ = original_grid.actnumsv.unchecked<3>();
    const size_t ncol = actnum_.shape(0);
    const size_t nrow = actnum_.shape(1);
    const size_t nlay = actnum_.shape(2);
    std::vector<size_t> actnumsv_shape = { ncol, nrow, 1 };
    py::array_t<int8_t> new_actnumsv(actnumsv_shape);
    auto new_actnumsv_ = new_actnumsv.mutable_unchecked<3>();

    // Use std::fill to set all values in the actnumsv array to 1
    std::fill(new_actnumsv.mutable_data(),
              new_actnumsv.mutable_data() + new_actnumsv.size(), 1);

    Grid new_grid = original_grid;   // Copy the original grid
    new_grid.zcornsv = new_zcornsv;  // Replace zcornsv with the reduced version
    new_grid.actnumsv = new_actnumsv;
    new_grid.nlay = 1;  // Update the number of layers to reflect the new grid

    return new_grid;
}

}  // namespace xtgeo::grid3d
