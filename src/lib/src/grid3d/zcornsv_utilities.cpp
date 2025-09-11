/*
 * xtgeo: utilities for converting zcornsv between pillar and cell formats
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <xtgeo/grid3d.hpp>

namespace py = pybind11;

namespace xtgeo::grid3d {

/**
 * @brief Convert from pillar-based zcornsv to cell-based zcornsv_cell format
 *
 * @param zcornsv_pillar Input array with shape (ncol+1, nrow+1, nlay+1, 4)
 *                       where each pillar has 4 corner values indexed as:
 *                       3: northeast, 2: northwest, 1: southeast, 0: southwest
 *                       This is the native xtgeo _zcornsv format in Python
 *
 * @return py::array_t<float> Output array with shape (ncol, nrow, nlay+1, 4)
 *                            where each cell has 4 corner values indexed as:
 *                            0: southwest, 1: southeast, 2: northwest, 3: northeast
 *
 * The mapping is:
 * - zcornsv_pillar(i, j, k, 3)     (southwest corner)  -> zcornsv_cell(i, j, k, 0)
 * - zcornsv_pillar(i+1, j, k, 2)   (southeast corner)  -> zcornsv_cell(i, j, k, 1)
 * - zcornsv_pillar(i, j+1, k, 1)   (northwest corner)  -> zcornsv_cell(i, j, k, 2)
 * - zcornsv_pillar(i+1, j+1, k, 0) (northeast corner)  -> zcornsv_cell(i, j, k, 3)
 *
 *         |                                  |------------|
 *         |    CELL I,J,K                    |2          3|
 *        2|3               |                 |    I,J,K   |
 *    ----------------------|                 |0          1|
 *        0|1                                 |------------|
 *         |
 *     Pillar corners                            Cell corners
 *
 *
 */
py::array_t<float>
zcornsv_pillar_to_cell(const py::array_t<float> &zcornsv_pillar)
{
    // Validate input dimensions
    if (zcornsv_pillar.ndim() != 4) {
        throw std::invalid_argument("zcornsv_pillar must have 4 dimensions");
    }

    const size_t ncol_plus1 = zcornsv_pillar.shape(0);
    const size_t nrow_plus1 = zcornsv_pillar.shape(1);
    const size_t nlay_plus1 = zcornsv_pillar.shape(2);
    const size_t corners = zcornsv_pillar.shape(3);

    if (corners != 4) {
        throw std::invalid_argument("zcornsv_pillar last dimension must be 4");
    }

    if (ncol_plus1 < 2 || nrow_plus1 < 2) {
        throw std::invalid_argument(
          "zcornsv_pillar must have at least 2 pillars in each direction");
    }

    const size_t ncol = ncol_plus1 - 1;
    const size_t nrow = nrow_plus1 - 1;

    // Create output array with shape (ncol, nrow, nlay+1, 4)
    py::array_t<float> zcornsv_cell =
      py::array_t<float>(std::vector<size_t>{ ncol, nrow, nlay_plus1, 4 });

    auto pillar_view = zcornsv_pillar.unchecked<4>();
    auto cell_view = zcornsv_cell.mutable_unchecked<4>();

    // Map from pillar format to cell format
    for (size_t i = 0; i < ncol; ++i) {
        for (size_t j = 0; j < nrow; ++j) {
            for (size_t k = 0; k < nlay_plus1; ++k) {
                // SW corner: zcornsv_pillar(i, j, k, 3) -> zcornsv_cell(i, j, k, 0)
                cell_view(i, j, k, 0) = pillar_view(i, j, k, 3);

                // SE corner: zcornsv_pillar(i+1, j, k, 2) -> zcornsv_cell(i, j, k, 1)
                cell_view(i, j, k, 1) = pillar_view(i + 1, j, k, 2);

                // NW corner: zcornsv_pillar(i, j+1, k, 1) -> zcornsv_cell(i, j, k, 2)
                cell_view(i, j, k, 2) = pillar_view(i, j + 1, k, 1);

                // NE corner: zcornsv_pillar(i+1, j+1, k, 0) -> zcornsv_cell(i, j, k, 3)
                cell_view(i, j, k, 3) = pillar_view(i + 1, j + 1, k, 0);
            }
        }
    }

    return zcornsv_cell;
}

py::array_t<float>
zcornsv_cell_to_pillar(const py::array_t<float> &zcornsv_cell, bool fill_boundary)
{
    // Validate input dimensions
    if (zcornsv_cell.ndim() != 4) {
        throw std::invalid_argument("zcornsv_cell must have 4 dimensions");
    }

    const size_t ncol = zcornsv_cell.shape(0);
    const size_t nrow = zcornsv_cell.shape(1);
    const size_t nlay_plus1 = zcornsv_cell.shape(2);
    const size_t corners = zcornsv_cell.shape(3);

    if (corners != 4) {
        throw std::invalid_argument("zcornsv_cell last dimension must be 4");
    }

    const size_t ncol_plus1 = ncol + 1;
    const size_t nrow_plus1 = nrow + 1;

    // Create output array with shape (ncol+1, nrow+1, nlay+1, 4)
    py::array_t<float> zcornsv_pillar =
      py::array_t<float>(std::vector<size_t>{ ncol_plus1, nrow_plus1, nlay_plus1, 4 });

    auto cell_view = zcornsv_cell.unchecked<4>();
    auto pillar_view = zcornsv_pillar.mutable_unchecked<4>();

    // Initialize all values to NaN to detect unassigned values
    for (size_t i = 0; i < ncol_plus1; ++i) {
        for (size_t j = 0; j < nrow_plus1; ++j) {
            for (size_t k = 0; k < nlay_plus1; ++k) {
                for (size_t c = 0; c < 4; ++c) {
                    pillar_view(i, j, k, c) = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
    }

    // Map from cell format to pillar format
    for (size_t i = 0; i < ncol; ++i) {
        for (size_t j = 0; j < nrow; ++j) {
            for (size_t k = 0; k < nlay_plus1; ++k) {
                // SW corner: zcornsv_cell(i, j, k, 0) -> zcornsv_pillar(i, j, k, 3)
                pillar_view(i, j, k, 3) = cell_view(i, j, k, 0);

                // SE: zcornsv_cell(i, j, k, 1) -> zcornsv_pillar(i+1, j, k, 2)
                pillar_view(i + 1, j, k, 2) = cell_view(i, j, k, 1);

                // NW corner: zcornsv_cell(i, j, k, 2) -> zcornsv_pillar(i, j + 1, k, 1)
                pillar_view(i, j + 1, k, 1) = cell_view(i, j, k, 2);

                // NE corner: zcornsv_cell(i, j, k, 3) -> zcornsv_pillar(i+1, j+1, k, 0)
                pillar_view(i + 1, j + 1, k, 0) = cell_view(i, j, k, 3);
            }
        }
    }

    // Fill boundary values if requested
    if (fill_boundary) {
        for (size_t i = 0; i < ncol_plus1; ++i) {
            for (size_t j = 0; j < nrow_plus1; ++j) {
                for (size_t k = 0; k < nlay_plus1; ++k) {
                    for (size_t c = 0; c < 4; ++c) {
                        if (std::isnan(pillar_view(i, j, k, c))) {
                            // Calculate average from defined values at this pillar
                            float sum = 0.0f;
                            size_t count = 0;
                            for (size_t cc = 0; cc < 4; ++cc) {
                                if (!std::isnan(pillar_view(i, j, k, cc))) {
                                    sum += pillar_view(i, j, k, cc);
                                    count++;
                                }
                            }
                            if (count > 0) {
                                pillar_view(i, j, k, c) = sum / count;
                            }
                            // If no values are defined at this pillar, leave as NaN
                        }
                    }
                }
            }
        }
    }

    return zcornsv_pillar;
}

}  // namespace xtgeo::grid3d
