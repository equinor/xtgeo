#include <cstddef>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <xtgeo/xtgeo.h>

namespace py = pybind11;

py::array_t<double>
grid3d_cellvol(const size_t ncol,
               const size_t nrow,
               const size_t nlay,
               const py::array_t<double> coordsv,
               const py::array_t<float> zcornsv,
               const py::array_t<int> actnumsv,
               const int precision,
               const bool asmasked = false)
{
    auto coordsv_buf = coordsv.request();
    auto zcornsv_buf = zcornsv.request();
    auto actnumsv_buf = actnumsv.request();

    double *coordsv_ = static_cast<double *>(coordsv_buf.ptr);
    float *zcornsv_ = static_cast<float *>(zcornsv_buf.ptr);
    int *actnumsv_ = static_cast<int *>(actnumsv_buf.ptr);

    pybind11::array_t<double> cellvols({ ncol, nrow, nlay });
    double *cellvolsv_ = static_cast<double *>(cellvols.request().ptr);

    double corners[24]{};
    for (size_t i = 0; i < ncol; i++) {
        for (size_t j = 0; j < nrow; j++) {
            for (size_t k = 0; k < nlay; k++) {
                size_t idx = i * nrow * nlay + j * nlay + k;
                if (asmasked && actnumsv_[idx] == 0) {
                    cellvolsv_[idx] = UNDEF;
                    continue;
                }
                grdcp3d_corners(i, j, k, ncol, nrow, nlay, coordsv_, coordsv_buf.size,
                                zcornsv_, zcornsv_buf.size, corners);
                cellvolsv_[idx] = x_hexahedron_volume(corners, 24, precision);
            }
        }
    }
    return cellvols;
}

PYBIND11_MODULE(_internal, m)
{
    m.def("grid3d_cellvol", &grid3d_cellvol, "Compute the bulk volume of cells.");
}
