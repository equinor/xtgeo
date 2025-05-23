#include <pybind11/pybind11.h>
#include <xtgeo/cube.hpp>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/regsurf.hpp>
#include <xtgeo/xyz.hpp>

PYBIND11_MODULE(_internal, m)
{
    m.doc() =
      "XTGeo's internal C++ library. Not intended to be directly used by users.";

    xtgeo::cube::init(m);
    xtgeo::geometry::init(m);
    xtgeo::grid3d::init(m);
    xtgeo::logging::init(m);
    xtgeo::regsurf::init(m);
    xtgeo::numerics::init(m);
    xtgeo::xyz::init(m);
}
