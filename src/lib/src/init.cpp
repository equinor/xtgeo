#include <pybind11/pybind11.h>

#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>

PYBIND11_MODULE(_internal, m)
{
    m.doc() =
      "XTGeo's internal C++ library. Not intended to be directly used by users.";

    xtgeo::geometry::init(m);
    xtgeo::grid3d::init(m);
}
