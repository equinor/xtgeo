#include <xtgeo/geometry.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {
/*
 * A generic function to estimate if a point is inside a polygon seen from bird view.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param polygon A vector of doubles, length 2 (N points in the polygon)
 * @return Boolean
 */

bool
is_xy_point_in_polygon(const double x, const double y, const xyz::Polygon &polygon)
{
    bool inside = false;
    int n = polygon.size();  // Define the variable n
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon.points[i].x, yi = polygon.points[i].y;
        double xj = polygon.points[j].x, yj = polygon.points[j].y;

        bool intersect =
          ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

}  // namespace xtgeo::geometry
