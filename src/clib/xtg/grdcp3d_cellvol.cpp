extern "C" {
#include "libxtg.h"
}

#include <cstddef>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void
grdcp3d_cellvol(size_t ncol,
                 size_t nrow,
                 size_t nlay,
                 py::array_t<double> coordsv,
                 py::array_t<float> zcornsv,
                 py::array_t<int> actnumsv,
                 py::array_t<double> cellvolsv,
                 int precision,
                 bool asmasked = false)
{
    py::buffer_info coordsv_buf = coordsv.request();
    py::buffer_info zcornsv_buf = zcornsv.request();
    py::buffer_info actnumsv_buf = actnumsv.request();
    py::buffer_info cellvolsv_buf = cellvolsv.request();

    double *coordsv_ = static_cast<double *>(coordsv_buf.ptr);
    float *zcornsv_ = static_cast<float *>(zcornsv_buf.ptr);
    int *actnumsv_ = static_cast<int *>(actnumsv_buf.ptr);
    double *cellvolsv_ = static_cast<double *>(cellvolsv_buf.ptr);

    double corners[24]{};

    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
            for (size_t k = 0; k < nlay; k++) {
                size_t ic = i * nrow * nlay + j * nlay + k;
                if (asmasked && actnumsv_[ic] == 0) {
                    cellvolsv_[ic] = UNDEF;
                    continue;
                }
                grdcp3d_corners(i, j, k,
                                ncol, nrow, nlay,
                                coordsv_, coordsv_buf.size,
                                zcornsv_, zcornsv_buf.size,
                                corners);
                cellvolsv_[ic] = x_hexahedron_volume(corners, 24, precision);
            }
        }
    }
}

PYBIND11_MODULE(_internal, m) {
    m.def("grdcp3d_cellvol", &grdcp3d_cellvol, "Compute bulk volume of cells.");
}
