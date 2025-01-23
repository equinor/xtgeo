#ifndef XTGEO_NUMERICS_HPP_
#define XTGEO_NUMERICS_HPP_
#include <pybind11/pybind11.h>
#include <limits>
#include <xtgeo/types.hpp>

namespace py = pybind11;

namespace xtgeo::numerics {

inline xyz::Point
lerp3d(double x1, double y1, double z1, double x2, double y2, double z2, double t)
{
    return xyz::Point{ x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1) };
}

inline void
init(py::module &m)
{
    auto m_numerics = m.def_submodule("numerics", "Internal functions for numerics.");

    m_numerics.attr("UNDEF_DOUBLE") = numerics::UNDEF_DOUBLE;
    m_numerics.attr("EPSILON") = numerics::EPSILON;
    m_numerics.attr("TOLERANCE") = numerics::TOLERANCE;
}

}  // namespace xtgeo::numerics

#endif  // XTGEO_NUMERICS_HPP_
