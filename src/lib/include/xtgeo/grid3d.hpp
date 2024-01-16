#ifndef XTGEO_GRID3D_HPP_
#define XTGEO_GRID3D_HPP_

#include <cstddef>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace xtgeo::grid3d {

py::array_t<double>
grid_cell_volumes(const size_t ncol,
                  const size_t nrow,
                  const size_t nlay,
                  const py::array_t<double> &coordsv,
                  const py::array_t<float> &zcornsv,
                  const py::array_t<int> &actnumsv,
                  const int precision,
                  const bool asmasked);

std::vector<double>
cell_corners(const size_t i,
             const size_t j,
             const size_t k,
             const size_t ncol,
             const size_t nrow,
             const size_t nlay,
             const py::array_t<double> &coordsv,
             const py::array_t<float> &zcornsv);

inline void
init(py::module &m)
{
    auto m_grid3d =
      m.def_submodule("grid3d", "Internal functions for operations on 3d grids.");

    m_grid3d.def("grid_cell_volumes", &grid_cell_volumes,
                 "Compute the bulk volume of cell.");
    m_grid3d.def("cell_corners", &cell_corners,
                 "Get a vector containing the corners of a cell.");
}

}  // namespace xtgeo::grid3d

#endif  // XTGEO_GRID3D_HPP_
