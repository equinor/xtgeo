#ifndef XTGEO_NUMERICS_HPP_
#define XTGEO_NUMERICS_HPP_
#include <pybind11/pybind11.h>
#include <limits>

namespace py = pybind11;

namespace xtgeo::numerics {

constexpr double UNDEF_DOUBLE = std::numeric_limits<double>::max();
constexpr double EPSILON = std::numeric_limits<double>::epsilon();
constexpr double TOLERANCE = 1e-6;
constexpr double QUIET_NAN = std::numeric_limits<double>::quiet_NaN();

template<typename T>
struct Vec3
{
    T x, y, z;
};

inline Vec3<double>
lerp3d(double x1, double y1, double z1, double x2, double y2, double z2, double t)
{
    return Vec3<double>{ x1 + t * (x2 - x1), y1 + t * (y2 - y1), z1 + t * (z2 - z1) };
}

inline void
init(py::module &m)
{
    auto m_numerics = m.def_submodule("numerics", "Internal functions for numerics.");

    m_numerics.attr("UNDEF_DOUBLE") = UNDEF_DOUBLE;
    m_numerics.attr("EPSILON") = EPSILON;
    m_numerics.attr("TOLERANCE") = TOLERANCE;
}

}  // namespace xtgeo::numerics

#endif  // XTGEO_NUMERICS_HPP_
