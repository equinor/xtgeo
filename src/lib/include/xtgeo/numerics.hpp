#ifndef XTGEO_NUMERICS_HPP_
#define XTGEO_NUMERICS_HPP_

namespace xtgeo::numerics {

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

}  // namespace xtgeo::numerics

#endif  // XTGEO_NUMERICS_HPP_
