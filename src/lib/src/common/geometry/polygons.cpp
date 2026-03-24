#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <xtgeo/geometry.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

// =====================================================================================
// Internal helpers for Sutherland-Hodgman polygon clipping in 2D
// =====================================================================================

namespace {

// Returns true when point (x,y) is on the left side (or on) the directed edge a→b.
bool
sh_inside(double x, double y, double ax, double ay, double bx, double by)
{
    return (bx - ax) * (y - ay) - (by - ay) * (x - ax) >= 0.0;
}

// Intersection of segment (p1→p2) with the infinite line through (a, b).
std::pair<double, double>
sh_intersect(double p1x,
             double p1y,
             double p2x,
             double p2y,
             double ax,
             double ay,
             double bx,
             double by)
{
    double dx1 = p2x - p1x, dy1 = p2y - p1y;
    double dx2 = bx - ax, dy2 = by - ay;
    double denom = dx1 * dy2 - dy1 * dx2;
    if (std::abs(denom) < 1e-15)
        return { p1x, p1y };  // parallel – return start point as fallback
    double t = ((ax - p1x) * dy2 - (ay - p1y) * dx2) / denom;
    return { p1x + t * dx1, p1y + t * dy1 };
}

// Clip *subject* polygon against each half-plane defined by the edges of *clip*.
// Both polygons are given as vectors of (x, y) pairs.
std::vector<std::pair<double, double>>
sutherland_hodgman_2d(std::vector<std::pair<double, double>> subject,
                      const std::vector<std::pair<double, double>> &clip)
{
    for (size_t c = 0; c < clip.size() && !subject.empty(); ++c) {
        auto [ax, ay] = clip[c];
        auto [bx, by] = clip[(c + 1) % clip.size()];

        std::vector<std::pair<double, double>> output;
        output.reserve(subject.size() + 1);

        for (size_t i = 0; i < subject.size(); ++i) {
            auto [sx, sy] = subject[i];
            auto [ex, ey] = subject[(i + 1) % subject.size()];

            bool s_in = sh_inside(sx, sy, ax, ay, bx, by);
            bool e_in = sh_inside(ex, ey, ax, ay, bx, by);

            if (e_in) {
                if (!s_in)
                    output.push_back(sh_intersect(sx, sy, ex, ey, ax, ay, bx, by));
                output.push_back({ ex, ey });
            } else if (s_in) {
                output.push_back(sh_intersect(sx, sy, ex, ey, ax, ay, bx, by));
            }
        }
        subject = std::move(output);
    }
    return subject;
}

// Signed shoelace area of a 2-D polygon (positive = CCW, negative = CW).
double
signed_area_2d(const std::vector<std::pair<double, double>> &poly)
{
    double area = 0.0;
    int n = static_cast<int>(poly.size());
    for (int i = 0, j = n - 1; i < n; j = i++) {
        area += poly[j].first * poly[i].second;
        area -= poly[i].first * poly[j].second;
    }
    return 0.5 * area;
}

}  // anonymous namespace
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

// =====================================================================================
// Quadrilateral face overlap area
// =====================================================================================

/*
 * Project a 3-D point onto a 2-D plane defined by orthonormal basis (u, v) relative
 * to origin point *o*.
 */
static std::pair<double, double>
project_to_plane(const xyz::Point &p,
                 const xyz::Point &o,
                 const Eigen::Vector3d &u,
                 const Eigen::Vector3d &v)
{
    Eigen::Vector3d d = p.data() - o.data();
    return { u.dot(d), v.dot(d) };
}

QuadOverlapResult
quadrilateral_face_overlap_result(const std::array<xyz::Point, 4> &face1,
                                  const std::array<xyz::Point, 4> &face2,
                                  double max_normal_gap)
{
    // ------------------------------------------------------------------
    // 1. Compute the average face normal from both quads.  Each quad's
    //    normal is the cross product of its diagonals.
    // ------------------------------------------------------------------
    auto quad_normal = [](const std::array<xyz::Point, 4> &f) -> Eigen::Vector3d {
        Eigen::Vector3d d1 = f[2].data() - f[0].data();  // diagonal A→C
        Eigen::Vector3d d2 = f[3].data() - f[1].data();  // diagonal B→D
        return d1.cross(d2);
    };

    Eigen::Vector3d n = quad_normal(face1) + quad_normal(face2);
    double n_norm = n.norm();
    if (n_norm < 1e-15)
        return { 0.0, xyz::Point(0, 0, 0) };  // degenerate faces
    n /= n_norm;

    // ------------------------------------------------------------------
    // 1b. Non-adjacency guard.
    //     A negative max_normal_gap is a sentinel meaning "auto-compute":
    //     use the longer face diagonal as the limit.  Adjacent cells have
    //     centroid gap ≈ 0 (conforming) or ≪ face size (faulted), so the
    //     guard passes in 99.9 % of real-grid cases.  Non-adjacent cells in
    //     the same column have gap ≈ cell thickness ≈ face diagonal and
    //     therefore get rejected.
    // ------------------------------------------------------------------
    auto centroid = [](const std::array<xyz::Point, 4> &f) -> Eigen::Vector3d {
        return (f[0].data() + f[1].data() + f[2].data() + f[3].data()) * 0.25;
    };

    if (max_normal_gap < 0.0) {
        auto quad_diag = [](const std::array<xyz::Point, 4> &f) -> double {
            double da = (f[2].data() - f[0].data()).norm();  // diagonal A→C
            double db = (f[3].data() - f[1].data()).norm();  // diagonal B→D
            return std::max(da, db);
        };
        max_normal_gap = std::max(quad_diag(face1), quad_diag(face2));
    }

    double normal_gap = std::abs(n.dot(centroid(face2) - centroid(face1)));
    if (normal_gap > max_normal_gap)
        return { 0.0, xyz::Point(n) };

    // ------------------------------------------------------------------
    // 2. Build an orthonormal basis (u, v) in the face plane.
    // ------------------------------------------------------------------
    Eigen::Vector3d seed =
      (std::abs(n.x()) <= std::abs(n.y()) && std::abs(n.x()) <= std::abs(n.z()))
        ? Eigen::Vector3d(1, 0, 0)
        : Eigen::Vector3d(0, 1, 0);
    Eigen::Vector3d u = (seed - seed.dot(n) * n).normalized();
    Eigen::Vector3d v = n.cross(u);

    // ------------------------------------------------------------------
    // 3. Project all corners to 2-D (u, v) coordinates.
    // ------------------------------------------------------------------
    const xyz::Point &origin = face1[0];

    auto project_quad =
      [&](
        const std::array<xyz::Point, 4> &f) -> std::vector<std::pair<double, double>> {
        std::vector<std::pair<double, double>> pts;
        pts.reserve(4);
        for (const auto &p : f)
            pts.push_back(project_to_plane(p, origin, u, v));
        return pts;
    };

    auto p1 = project_quad(face1);
    auto p2 = project_quad(face2);

    // ------------------------------------------------------------------
    // 4. Ensure the clip polygon (p2) is CCW so SH "inside" is correct.
    // ------------------------------------------------------------------
    if (signed_area_2d(p2) < 0.0)
        std::reverse(p2.begin(), p2.end());

    // ------------------------------------------------------------------
    // 5. Clip face1 against face2, compute area and centroid.
    // ------------------------------------------------------------------
    auto clipped = sutherland_hodgman_2d(p1, p2);
    if (clipped.size() < 3)
        return { 0.0, xyz::Point(n), xyz::Point(0, 0, 0) };

    double area = std::abs(signed_area_2d(clipped));

    // Compute the area-weighted 2D centroid of the intersection polygon.
    double cx = 0.0, cy = 0.0;
    const size_t nc = clipped.size();
    for (size_t ci = 0; ci < nc; ci++) {
        const auto &p = clipped[ci];
        const auto &pn = clipped[(ci + 1) % nc];
        double cross = p.first * pn.second - pn.first * p.second;
        cx += (p.first + pn.first) * cross;
        cy += (p.second + pn.second) * cross;
    }
    double denom = 6.0 * signed_area_2d(clipped);
    if (std::abs(denom) > 1e-15) {
        cx /= denom;
        cy /= denom;
    } else {
        // Degenerate: fall back to vertex average
        cx = cy = 0.0;
        for (const auto &p : clipped) {
            cx += p.first;
            cy += p.second;
        }
        cx /= nc;
        cy /= nc;
    }
    Eigen::Vector3d centroid_3d = origin.data() + cx * u + cy * v;

    return { area, xyz::Point(n), xyz::Point(centroid_3d) };
}

double
quadrilateral_face_overlap_area(const std::array<xyz::Point, 4> &face1,
                                const std::array<xyz::Point, 4> &face2,
                                double max_normal_gap)
{
    return quadrilateral_face_overlap_result(face1, face2, max_normal_gap).area;
}

}  // namespace xtgeo::geometry
