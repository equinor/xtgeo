#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
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
        double xi = polygon.points[i].x(), yi = polygon.points[i].y();
        double xj = polygon.points[j].x(), yj = polygon.points[j].y();

        bool intersect =
          ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) {
            inside = !inside;
        }
    }
    return inside;
}

std::array<double, 4>
get_polygon_xy_bbox(const xyz::Polygon &polygon)
{
    if (polygon.size() == 0) {
        return { 0.0, 0.0, 0.0, 0.0 };
    }

    double min_x = polygon.points[0].x();
    double max_x = min_x;
    double min_y = polygon.points[0].y();
    double max_y = min_y;

    for (size_t i = 1; i < polygon.size(); ++i) {
        const double px = polygon.points[i].x();
        const double py = polygon.points[i].y();
        min_x = std::min(min_x, px);
        max_x = std::max(max_x, px);
        min_y = std::min(min_y, py);
        max_y = std::max(max_y, py);
    }

    return { min_x, max_x, min_y, max_y };
}

/*
 * Cell-polygon relationship check with precomputed polygon bbox.
 *
 * @param cell The cell corners to check
 * @param polygon The polygon boundary
 * @param poly_bbox Precomputed polygon bounding box [min_x, max_x, min_y, max_y]
 * @return CellPolygonRelation enum value
 */
CellPolygonRelation
cell_polygon_relation(const grid3d::CellCorners &cell,
                      const xyz::Polygon &polygon,
                      const std::array<double, 4> &poly_bbox)
{
    // Fast bounding box check - find cell bbox on the fly
    double cell_min_x = std::min(
      { cell.upper_sw.x(), cell.upper_se.x(), cell.upper_nw.x(), cell.upper_ne.x(),
        cell.lower_sw.x(), cell.lower_se.x(), cell.lower_nw.x(), cell.lower_ne.x() });
    double cell_max_x = std::max(
      { cell.upper_sw.x(), cell.upper_se.x(), cell.upper_nw.x(), cell.upper_ne.x(),
        cell.lower_sw.x(), cell.lower_se.x(), cell.lower_nw.x(), cell.lower_ne.x() });
    double cell_min_y = std::min(
      { cell.upper_sw.y(), cell.upper_se.y(), cell.upper_nw.y(), cell.upper_ne.y(),
        cell.lower_sw.y(), cell.lower_se.y(), cell.lower_nw.y(), cell.lower_ne.y() });
    double cell_max_y = std::max(
      { cell.upper_sw.y(), cell.upper_se.y(), cell.upper_nw.y(), cell.upper_ne.y(),
        cell.lower_sw.y(), cell.lower_se.y(), cell.lower_nw.y(), cell.lower_ne.y() });

    // Check if bounding boxes don't overlap
    if (cell_max_x < poly_bbox[0] || cell_min_x > poly_bbox[1] ||
        cell_max_y < poly_bbox[2] || cell_min_y > poly_bbox[3]) {
        return CellPolygonRelation::Outside;
    }

    // Check first corner to establish baseline
    bool first_inside =
      is_xy_point_in_polygon(cell.upper_sw.x(), cell.upper_sw.y(), polygon);

    // Early exit strategy: check remaining corners for any mismatch
    if (is_xy_point_in_polygon(cell.upper_se.x(), cell.upper_se.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.upper_nw.x(), cell.upper_nw.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.upper_ne.x(), cell.upper_ne.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.lower_sw.x(), cell.lower_sw.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.lower_se.x(), cell.lower_se.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.lower_nw.x(), cell.lower_nw.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;
    if (is_xy_point_in_polygon(cell.lower_ne.x(), cell.lower_ne.y(), polygon) !=
        first_inside)
        return CellPolygonRelation::Intersecting;

    // All corners have same status
    return first_inside ? CellPolygonRelation::Inside : CellPolygonRelation::Outside;
}

/*
 * Cell-polygon relationship check without precomputed polygon bbox.
 * It will use the overload function with polygon bbox
 *
 * @param cell The cell corners to check
 * @param polygon The polygon boundary
 * @return CellPolygonRelation enum value
 */
CellPolygonRelation
cell_polygon_relation(const grid3d::CellCorners &cell, const xyz::Polygon &polygon)
{
    auto poly_bbox = get_polygon_xy_bbox(polygon);
    return cell_polygon_relation(cell, polygon, poly_bbox);
}

}  // namespace xtgeo::geometry
