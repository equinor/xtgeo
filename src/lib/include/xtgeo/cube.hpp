#ifndef XTGEO_CUBE_HPP_
#define XTGEO_CUBE_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstddef>

namespace py = pybind11;

namespace xtgeo::cube {

std::unordered_map<std::string, py::array_t<double>>
cube_stats_along_z(const size_t ncol,
                   const size_t nrow,
                   const size_t nlay,
                   const py::array_t<float> &cubev);

inline void
init(py::module &m)
{
    auto m_cube =
      m.def_submodule("cube", "Internal functions for operations on 3d cubes.");

    m_cube.def("cube_stats_along_z", &cube_stats_along_z,
               "Compute various statistics for cube along the Z axis, returning maps.");
}

}  // namespace xtgeo::cube

#endif  // XTGEO_CUBE_HPP_
