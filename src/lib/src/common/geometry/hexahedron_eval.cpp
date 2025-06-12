#include <limits>   // Required for std::numeric_limits
#include <numeric>  // Required for std::accumulate
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

namespace xtgeo::geometry {

using xyz::Point;

using geometry::point::add;
using geometry::point::cross_product;
using geometry::point::dot_product;
using geometry::point::magnitude;
using geometry::point::magnitude_squared;
using geometry::point::scale;
using geometry::point::subtract;

/**
 * @brief Faster test for convexity of a hexahedron involving point-plane
 *        distance checks.
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
static bool
is_hexahedron_non_convex_test1(const HexahedronCorners &corners)
{
    // Create an array of vertices for easier access
    std::array<Point, 8> vertices = { corners.upper_sw, corners.upper_se,
                                      corners.upper_ne, corners.upper_nw,
                                      corners.lower_sw, corners.lower_se,
                                      corners.lower_ne, corners.lower_nw };

    // Define faces with consistent outward-pointing normal winding order
    const std::array<std::array<int, 4>, 6> faces = { {
      { 4, 7, 6, 5 },  // bottom face
      { 0, 1, 2, 3 },  // top face
      { 4, 5, 1, 0 },  // front face
      { 5, 6, 2, 1 },  // right face
      { 6, 7, 3, 2 },  // back face
      { 7, 4, 0, 3 }   // left face
    } };

    constexpr double POINT_PLANE_TOLERANCE = 1e-9;

    // Iterate over each face
    for (const auto &face : faces) {
        const Point &p0 = vertices[face[0]];
        const Point &p1 = vertices[face[1]];
        const Point &p2 = vertices[face[2]];

        // Calculate the normal of the face using the cross product
        Point edge1 = subtract(p1, p0);
        Point edge2 = subtract(p2, p0);
        Point normal = cross_product(edge1, edge2);

        // Check the sign of the dot product for all other vertices
        double reference_sign = 0.0;
        for (size_t i = 0; i < vertices.size(); ++i) {
            // Skip vertices that are part of the current face
            if (i == face[0] || i == face[1] || i == face[2] || i == face[3]) {
                continue;
            }

            // Calculate the vector from the face to the vertex
            Point vec = subtract(vertices[i], p0);

            // Calculate the dot product with the face normal
            double dot = dot_product(vec, normal);

            // Determine the sign of the dot product
            if (std::abs(dot) > POINT_PLANE_TOLERANCE) {
                double sign = (dot > 0) ? 1.0 : -1.0;

                // If the reference sign is not set, initialize it
                if (reference_sign == 0.0) {
                    reference_sign = sign;
                }
                // If the sign differs from the reference, the cell is non-convex
                else if (sign != reference_sign) {
                    return true;  // Non-convex
                }
            }
        }
    }

    // If all vertices are on the same side of all face planes, the cell is convex
    return false;
}

/**
 * Determines if a hexahedral cell is non-convex.
 * Checks for non-planar faces and whether the centroid lies inside all face planes.
 *
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
static bool
is_hexahedron_non_convex_test2(const HexahedronCorners &corners)
{
    // Create more accessible array of vertices
    std::array<Point, 8> vertices = { corners.upper_sw, corners.upper_se,
                                      corners.upper_ne, corners.upper_nw,
                                      corners.lower_sw, corners.lower_se,
                                      corners.lower_ne, corners.lower_nw };

    // Define faces with consistent outward-pointing normal winding order (assuming
    // standard node numbering) (e.g., using the right-hand rule)
    const std::array<std::array<int, 4>, 6> faces = { {
      { 4, 7, 6, 5 },  // bottom face (points down)
      { 0, 1, 2, 3 },  // top face (points up)
      { 4, 5, 1, 0 },  // front face (points -y)
      { 5, 6, 2, 1 },  // right face (points +x)
      { 6, 7, 3, 2 },  // back face (points +y)
      { 7, 4, 0, 3 }   // left face (points -x)
    } };

    constexpr double PLANAR_TOLERANCE_VOL = 1e-9;   // For scalar triple product (vol)
    constexpr double POINT_PLANE_TOLERANCE = 1e-9;  // For point-plane distance

    // 1. Check if any face is non-planar using scalar triple product
    for (const auto &face : faces) {
        const Point &a = vertices[face[0]];
        const Point &b = vertices[face[1]];
        const Point &c = vertices[face[2]];
        const Point &d = vertices[face[3]];

        Point ab = subtract(b, a);
        Point ac = subtract(c, a);
        Point ad = subtract(d, a);

        // Volume of tetrahedron formed by A, B, C, D
        double volume = dot_product(ab, cross_product(ac, ad));

        // If volume is significantly non-zero, the face is non-planar
        // Note: Need to consider the scale of the cell coordinates for the tolerance.
        // A relative tolerance might be better, but requires calculating face area/edge
        // lengths. Using a small absolute tolerance for now.
        if (std::abs(volume) > PLANAR_TOLERANCE_VOL) {
            // Optional: Add logging here if needed
            // xtgeo::log_trace("Non-planar face detected (volume: {})", volume);
            return true;  // Face is non-planar, cell is non-convex
        }
    }

    // 2. Check if the centroid lies on the inner side of all face planes
    // Calculate centroid
    Point centroid = { 0.0, 0.0, 0.0 };
    for (const auto &v : vertices) {
        centroid = add(centroid, v);
    }
    centroid = scale(centroid, 1.0 / 8.0);

    for (const auto &face : faces) {
        const Point &p0 = vertices[face[0]];
        const Point &p1 = vertices[face[1]];
        // Use p3 for normal calculation based on winding order {0, 1, 2, 3} -> (p1-p0)
        // x (p3-p0)
        const Point &p3 = vertices[face[3]];

        // Calculate face normal (consistent winding order assumed in `faces` array)
        Point v1 = subtract(p1, p0);
        Point v2 = subtract(p3, p0);
        Point normal = cross_product(v1, v2);

        // Check if the centroid is on the correct side of the plane defined by p0 and
        // normal (p - p0) . N <= 0 means p is on the side opposite to N direction (or
        // on the plane)
        double dist = dot_product(subtract(centroid, p0), normal);

        // If dist is positive (beyond tolerance), the centroid is "outside" this face
        // plane.
        if (dist > POINT_PLANE_TOLERANCE) {
            // Optional: Add logging here if needed
            // xtgeo::log_trace("Centroid outside face plane (dist: {})", dist);
            return true;  // Centroid is outside, cell is non-convex
        }
    }

    // Cell passed all tests, it's convex
    return false;
}

/**
 * @brief Check if a hexahedron is non-convex.
 * @param corners The 8 corners of the hexahedron
 * @return true if the cell is non-convex, false if it is convex
 */
bool
is_hexahedron_non_convex(const HexahedronCorners &corners)
{
    // Check if the hexahedron is non-convex using the first test
    if (is_hexahedron_non_convex_test1(corners)) {
        return true;
    }

    // If the first test fails, check using the second test
    return is_hexahedron_non_convex_test2(corners);
}  // is_hexahedron_non_convex

/*
 * Get the minimum and maximum values of the corners of a hexahedron.
 * @param HexahedronCorners struct
 * @return std::vector<double>
 */
std::vector<double>
get_hexahedron_minmax(const HexahedronCorners &cell_corners)
{
    double xmin = std::numeric_limits<double>::max();
    double xmax = std::numeric_limits<double>::min();
    double ymin = std::numeric_limits<double>::max();
    double ymax = std::numeric_limits<double>::min();
    double zmin = std::numeric_limits<double>::max();
    double zmax = std::numeric_limits<double>::min();

    // List of all corners
    std::array<Point, 8> corners = { cell_corners.upper_sw, cell_corners.upper_se,
                                     cell_corners.upper_ne, cell_corners.upper_nw,
                                     cell_corners.lower_sw, cell_corners.lower_se,
                                     cell_corners.lower_ne, cell_corners.lower_nw };

    // Iterate over all corners to find min/max values
    for (const auto &corner : corners) {
        if (corner.x() < xmin)
            xmin = corner.x();
        if (corner.x() > xmax)
            xmax = corner.x();
        if (corner.y() < ymin)
            ymin = corner.y();
        if (corner.y() > ymax)
            ymax = corner.y();
        if (corner.z() < zmin)
            zmin = corner.z();
        if (corner.z() > zmax)
            zmax = corner.z();
    }

    return { xmin, xmax, ymin, ymax, zmin, zmax };
}

bool
is_hexahedron_severely_distorted(const xtgeo::geometry::HexahedronCorners &corners)
{
    // Thresholds for distortion checks
    constexpr double ASPECT_RATIO_THRESHOLD = 200.0;  // since res. cells often are thin
    constexpr double MIN_VOLUME_THRESHOLD = 1e-8;
    constexpr double PLANARITY_TOLERANCE = 12.0;
    constexpr double DIHEDRAL_ANGLE_TOLERANCE = 30.0;  // In degrees

    // Helper function to calculate the length of an edge
    auto edge_length = [](const Point &p1, const Point &p2) {
        return std::sqrt(std::pow(p2.x() - p1.x(), 2) + std::pow(p2.y() - p1.y(), 2) +
                         std::pow(p2.z() - p1.z(), 2));
    };

    // ---------------------------------------------------------------------------------
    // Check aspect ratios
    std::array<double, 12> edge_lengths = {
        edge_length(corners.upper_sw, corners.upper_se),
        edge_length(corners.upper_se, corners.upper_ne),
        edge_length(corners.upper_ne, corners.upper_nw),
        edge_length(corners.upper_nw, corners.upper_sw),
        edge_length(corners.lower_sw, corners.lower_se),
        edge_length(corners.lower_se, corners.lower_ne),
        edge_length(corners.lower_ne, corners.lower_nw),
        edge_length(corners.lower_nw, corners.lower_sw),
        edge_length(corners.upper_sw, corners.lower_sw),
        edge_length(corners.upper_se, corners.lower_se),
        edge_length(corners.upper_ne, corners.lower_ne),
        edge_length(corners.upper_nw, corners.lower_nw),
    };

    double max_edge = *std::max_element(edge_lengths.begin(), edge_lengths.end());
    double min_edge = *std::min_element(edge_lengths.begin(), edge_lengths.end());

    if (min_edge <= 0.0) {
        return true;  // Severely distorted due to zero or negative edge length
    }
    if (max_edge / min_edge > ASPECT_RATIO_THRESHOLD) {
        return true;  // Severely distorted due to aspect ratio
    }

    // ---------------------------------------------------------------------------------
    // Check face planarity
    double planarity_tol = PLANARITY_TOLERANCE * max_edge;

    auto check_face_planarity = [planarity_tol](const std::array<Point, 4> &face) {
        auto edge1 = subtract(face[1], face[0]);
        auto edge2 = subtract(face[2], face[0]);
        auto normal = cross_product(edge1, edge2);
        double mag = magnitude(normal);
        if (mag <= 1e-9) {
            return false;
        }
        normal.x() /= mag;
        normal.y() /= mag;
        normal.z() /= mag;

        for (const auto &point : face) {
            auto vec = subtract(point, face[0]);
            if (std::abs(dot_product(vec, normal)) > planarity_tol) {
                return false;  // Face is not sufficiently planar
            }
        }
        return true;
    };

    std::array<std::array<Point, 4>, 6> faces = { {
      { corners.upper_sw, corners.upper_se, corners.upper_ne, corners.upper_nw },
      { corners.lower_sw, corners.lower_se, corners.lower_ne, corners.lower_nw },
      { corners.upper_sw, corners.upper_se, corners.lower_se, corners.lower_sw },
      { corners.upper_se, corners.upper_ne, corners.lower_ne, corners.lower_se },
      { corners.upper_ne, corners.upper_nw, corners.lower_nw, corners.lower_ne },
      { corners.upper_nw, corners.upper_sw, corners.lower_sw, corners.lower_nw },
    } };

    for (const auto &face : faces) {
        if (!check_face_planarity(face)) {
            return true;  // Severely distorted due to non-planar face
        }
    }

    // ---------------------------------------------------------------------------------
    // Check dihedral angles - correctly checking only adjacent faces
    auto calculate_angle = [](const Point &normal1, const Point &normal2) {
        double dot = dot_product(normal1, normal2);
        double magnitude1 = std::sqrt(dot_product(normal1, normal1));
        double magnitude2 = std::sqrt(dot_product(normal2, normal2));

        // Avoid division by zero and clamp dot product to [-1, 1]
        if (magnitude1 < 1e-10 || magnitude2 < 1e-10) {
            return 0.0;  // Degenerate face
        }

        double cosine = dot / (magnitude1 * magnitude2);
        // Clamp cosine to avoid domain errors with acos
        cosine = std::max(-1.0, std::min(1.0, cosine));

        return std::acos(cosine) * 180.0 / M_PI;  // Convert to degrees
    };

    const std::array<std::array<int, 4>, 6> adjacent_faces = { {
      { 2, 3, 4, 5 },  // Face 0 (top) is adjacent to 2,3,4,5
      { 2, 3, 4, 5 },  // Face 1 (bottom) is adjacent to 2,3,4,5
      { 0, 1, 3, 5 },  // Face 2 (front) is adjacent to 0,1,3,5
      { 0, 1, 2, 4 },  // Face 3 (right) is adjacent to 0,1,2,4
      { 0, 1, 3, 5 },  // Face 4 (back) is adjacent to 0,1,3,5
      { 0, 1, 2, 4 }   // Face 5 (left) is adjacent to 0,1,2,4
    } };

    // Check dihedral angles between adjacent faces only
    for (size_t i = 0; i < faces.size(); ++i) {
        // Calculate normal for face i
        auto normal1 = cross_product(subtract(faces[i][1], faces[i][0]),
                                     subtract(faces[i][2], faces[i][0]));

        for (const auto &j : adjacent_faces[i]) {
            if (j > i) {  // To avoid checking pairs twice
                // Calculate normal for face j
                auto normal2 = cross_product(subtract(faces[j][1], faces[j][0]),
                                             subtract(faces[j][2], faces[j][0]));

                double angle = calculate_angle(normal1, normal2);

                // For a perfect cube, adjacent faces would have 90-degree angles
                if (std::abs(angle - 90.0) > DIHEDRAL_ANGLE_TOLERANCE) {
                    return true;  // Severely distorted due to dihedral angle deviation
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Check volume
    double volume = hexahedron_volume(corners, HexVolumePrecision::P2);
    if (volume < MIN_VOLUME_THRESHOLD * max_edge * max_edge * max_edge) {
        return true;  // Severely distorted due to near-zero volume
    }

    return false;  // Hexahedron is not severely distorted
}

/**
 * @brief Check if a hexahedron cell is thin based on the ratio of thickness to area.
 * @param corners The 8 corners of the hexahedron
 * @param threshold The threshold for the thickness-to-area ratio
 * @return bool Returns true if the cell is thin, false otherwise
 */
bool
is_hexahedron_thin(const HexahedronCorners &corners, const double threshold)
{
    // Helper function to calculate the area of a quadrilateral face
    auto calculate_area = [](const Point &p1, const Point &p2, const Point &p3,
                             const Point &p4) -> double {
        auto cross = [](const Point &a, const Point &b) {
            return Point{ a.y() * b.z() - a.z() * b.y(), a.z() * b.x() - a.x() * b.z(),
                          a.x() * b.y() - a.y() * b.x() };
        };

        auto subtract = [](const Point &a, const Point &b) {
            return Point{ a.x() - b.x(), a.y() - b.y(), a.z() - b.z() };
        };

        // Divide the quadrilateral into two triangles and calculate their areas
        Point v1 = subtract(p2, p1);
        Point v2 = subtract(p3, p1);
        Point v3 = subtract(p4, p1);

        double area1 = 0.5 * std::sqrt(std::pow(cross(v1, v2).x(), 2) +
                                       std::pow(cross(v1, v2).y(), 2) +
                                       std::pow(cross(v1, v2).z(), 2));
        double area2 = 0.5 * std::sqrt(std::pow(cross(v2, v3).x(), 2) +
                                       std::pow(cross(v2, v3).y(), 2) +
                                       std::pow(cross(v2, v3).z(), 2));

        return area1 + area2;
    };

    // Calculate the area of the top and bottom faces
    double top_area = calculate_area(corners.upper_sw, corners.upper_se,
                                     corners.upper_ne, corners.upper_nw);
    double bottom_area = calculate_area(corners.lower_sw, corners.lower_se,
                                        corners.lower_ne, corners.lower_nw);

    // Use the average of the top and bottom areas
    double average_area = (top_area + bottom_area) / 2.0;

    // Calculate the thickness (difference in Z-coordinates between upper and lower
    // faces)
    double thickness = 0.25 * (std::abs(corners.upper_sw.z() - corners.lower_sw.z()) +
                               std::abs(corners.upper_se.z() - corners.lower_se.z()) +
                               std::abs(corners.upper_ne.z() - corners.lower_ne.z()) +
                               std::abs(corners.upper_nw.z() - corners.lower_nw.z()));

    if (thickness <= numerics::TOLERANCE) {
        return true;  // Cell is considered thin if thickness or area is too small
    }
    if (average_area <= numerics::TOLERANCE) {
        return false;  // Cell is probably not thin if area is too small
    }

    // Check if the thickness-to-area ratio is below the threshold
    return (thickness / average_area) < threshold;
}

/**
 * @brief Detect if a hexahedron is concave when viewed from above (projected onto the
 * XY plane). A cell is concave if one corner is within the triangle formed by the other
 * corners at the top and/or base.
 *
 * @param corners The 8 corners of the hexahedron
 * @return bool Returns true if the cell is concave, false if it is convex
 */
bool
is_hexahedron_concave_projected(const HexahedronCorners &corners)
{
    // Extract the X and Y coordinates of the corners for top and base
    std::array<std::array<double, 2>, 4> xp, yp;
    xp[0] = { corners.upper_sw.x(), corners.lower_sw.x() };
    xp[1] = { corners.upper_se.x(), corners.lower_se.x() };
    xp[2] = { corners.upper_ne.x(), corners.lower_ne.x() };
    xp[3] = { corners.upper_nw.x(), corners.lower_nw.x() };

    yp[0] = { corners.upper_sw.y(), corners.lower_sw.y() };
    yp[1] = { corners.upper_se.y(), corners.lower_se.y() };
    yp[2] = { corners.upper_ne.y(), corners.lower_ne.y() };
    yp[3] = { corners.upper_nw.y(), corners.lower_nw.y() };

    // Check for concavity at both the top and base
    for (int ntop = 0; ntop < 2; ++ntop) {
        for (int nchk = 0; nchk < 4; ++nchk) {
            // Form a triangle with the other three corners
            std::vector<Point> triangle_points;
            for (int n = 0; n < 4; ++n) {
                if (n != nchk) {
                    triangle_points.push_back({ xp[n][ntop], yp[n][ntop], 0.0 });
                }
            }

            // Construct the Polygon directly with its points
            xyz::Polygon triangle(triangle_points);

            // Check if the current corner is inside the triangle
            if (is_xy_point_in_polygon(xp[nchk][ntop], yp[nchk][ntop], triangle)) {
                return true;  // The cell is concave
            }
        }
    }

    return false;  // The cell is convex
}

/**
 * @brief Get the bounding box for the hexahedron defined by its corners.
 * @param corners The corners of the hexahedron
 * @return std::tuple<Point, Point> {min_point, max_point}
 */
std::tuple<Point, Point>
get_hexahedron_bounding_box(const HexahedronCorners &corners)
{
    auto minmax = get_hexahedron_minmax(corners);
    auto min_point = Point(minmax[0], minmax[2], minmax[4]);
    auto max_point = Point(minmax[1], minmax[3], minmax[5]);
    return std::make_tuple(min_point, max_point);
}  // get_hexahedron_bounding_box

/**
 * @brief Check if a point is inside a hexahedron bounding box
 * @param point The point to check
 * @param hexahedron_corners The corners of the hexahedron
 * @return true if the point is inside the bounding box, false otherwise
 */
bool
is_point_in_hexahedron_bounding_box(const Point &point,
                                    const HexahedronCorners &hexahedron_corners)
{

    // Quick rejection test using bounding box; this is independent of the method
    auto [min_pt, max_pt] = get_hexahedron_bounding_box(hexahedron_corners);

    double epsilon = 1e-8 * std::max({ max_pt.x() - min_pt.x(), max_pt.y() - min_pt.y(),
                                       max_pt.z() - min_pt.z() });

    // Use an epsilon for the bounding box check to handle numerical precision
    if (point.x() < min_pt.x() - epsilon || point.x() > max_pt.x() + epsilon ||
        point.y() < min_pt.y() - epsilon || point.y() > max_pt.y() + epsilon ||
        point.z() < min_pt.z() - epsilon || point.z() > max_pt.z() + epsilon) {
        return false;
    }
    return true;  // Point is within the bounding box
}  // is_point_in_hexahedron_bounding_box

/**
 * @brief Check if a point is inside a hexahedron bounding box defined by its min and
 * max points. This is a variant of the previous function that takes min and max points
 * directly instead of the hexahedron corners.
 * @param point The point to check
 * @param min_pt The minimum point of the bounding box
 * @param max_pt The maximum point of the bounding box
 * @return true if the point is inside the bounding box, false otherwise
 */
bool
is_point_in_hexahedron_bounding_box_minmax_pt(const Point &point,
                                              const Point &min_pt,
                                              const Point &max_pt)
{

    double epsilon = 1e-8 * std::max({ max_pt.x() - min_pt.x(), max_pt.y() - min_pt.y(),
                                       max_pt.z() - min_pt.z() });

    // Use an epsilon for the bounding box check to handle numerical precision
    if (point.x() < min_pt.x() - epsilon || point.x() > max_pt.x() + epsilon ||
        point.y() < min_pt.y() - epsilon || point.y() > max_pt.y() + epsilon ||
        point.z() < min_pt.z() - epsilon || point.z() > max_pt.z() + epsilon) {
        return false;
    }
    return true;  // Point is within the bounding box
}  // is_point_in_hexahedron_bounding_box_minmax_pt
}  // namespace xtgeo::geometry
