#include <xtgeo/geometry.hpp>

namespace xtgeo::geometry {
/*
 * A generic function to estimate if a point is inside a polygon seen from bird view.
 * @param x X coordinate of the point
 * @param y Y coordinate of the point
 * @param polygon A vector of doubles, length 2 (N points in the polygon)
 * @return Boolean
 */
bool
is_xy_point_in_polygon(const double x,
                       const double y,
                       const std::vector<std::array<double, 2>> &polygon)
{
    bool inside = false;
    int n = polygon.size();  // Define the variable n
    for (int i = 0, j = n - 1; i < n; j = i++) {
        double xi = polygon[i][0], yi = polygon[i][1];
        double xj = polygon[j][0], yj = polygon[j][1];

        bool intersect =
          ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}  // is_xy_point_in_polygon

}  // namespace xtgeo::geometry
