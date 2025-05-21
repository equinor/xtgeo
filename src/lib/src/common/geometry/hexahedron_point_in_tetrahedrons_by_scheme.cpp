#include <cmath>   // Required for fabs
#include <limits>  // Required for numeric_limits
#include <xtgeo/geometry.hpp>
#include <xtgeo/geometry_basics.hpp>
#include <xtgeo/grid3d.hpp>
#include <xtgeo/logging.hpp>
#include <xtgeo/numerics.hpp>
#include <xtgeo/types.hpp>

/**
 * Computing if a point is inside a hexahedron is a non-trivial task, and
 * several methods exist. The most variation in corner point cells is at top and base
 * faces, hence we select a scheme here based on top/base faces.
 */

namespace xtgeo::geometry {

// schemes used for tetrahedron decomposition when the cell is re-arranged to
// counter clock order.
//     3----2
//    /|   /|
//   0----1 |   Upper face (z+)
//   | 7--|-6
//   |/   |/
//   4----5 Lower face (z-)

constexpr int TETRAHEDRON_SCHEMES[4][6][4] = {
    // Scheme 0: Diagonal 0-2 at top and 4-6 at base
    { { 0, 1, 2, 6 },
      { 0, 2, 3, 6 },
      { 0, 4, 5, 6 },
      { 0, 6, 7, 4 },
      { 0, 3, 6, 7 },
      { 0, 1, 6, 5 } },
    // Scheme 1: Diagonal 0-2 at top and 5-7 at base
    { { 0, 1, 2, 5 },
      { 0, 2, 3, 5 },
      { 5, 6, 7, 2 },
      { 5, 7, 4, 0 },
      { 0, 3, 5, 7 },
      { 2, 3, 7, 5 } },
    // Scheme 2: Diagonal 1-3 at top and 4-6 at base
    { { 0, 1, 3, 4 },
      { 1, 2, 3, 6 },
      { 1, 3, 4, 6 },
      { 1, 4, 5, 6 },
      { 3, 6, 7, 4 },
      { 1, 3, 6, 4 } },
    // Scheme 3: Diagonal 1-3 at top and 5-7 at base
    { { 0, 1, 3, 5 },
      { 1, 2, 3, 5 },
      { 5, 6, 7, 3 },
      { 5, 7, 4, 0 },
      { 0, 3, 5, 7 },
      { 2, 3, 7, 5 } }
};

using xyz::Point;

using geometry::point::calculate_normal;
using geometry::point::dot;
using geometry::point::magnitude;

static double
calculate_normal_difference(const Point &p1,
                            const Point &p2,
                            const Point &p3,
                            const Point &p4)
{
    // Calculate normals for the two triangles in the first triangulation option
    Point normal1_option1 = calculate_normal(p1, p2, p3);
    Point normal2_option1 = calculate_normal(p3, p4, p1);

    // Calculate the angle between these normals
    double dot_product1 = dot(normal1_option1, normal2_option1);
    double magnitude1_option1 = magnitude(normal1_option1);
    double magnitude2_option1 = magnitude(normal2_option1);
    double cosine_angle1 = dot_product1 / (magnitude1_option1 * magnitude2_option1);

    // The closer cosine is to 1, the more parallel the normals (more planar)
    // The closer to -1, the more opposite they are
    // For a perfectly planar quad, cosine would be 1
    double planarity_score1 = std::abs(cosine_angle1);

    // Now calculate for the second triangulation option (p1,p3,p2) and (p2,p3,p4)
    Point normal1_option2 = calculate_normal(p1, p3, p2);
    Point normal2_option2 = calculate_normal(p2, p3, p4);

    double dot_product2 = dot(normal1_option2, normal2_option2);
    double magnitude1_option2 = magnitude(normal1_option2);
    double magnitude2_option2 = magnitude(normal2_option2);
    double cosine_angle2 = dot_product2 / (magnitude1_option2 * magnitude2_option2);
    double planarity_score2 = std::abs(cosine_angle2);

    // Return the difference between the two options
    // Higher value means greater difference between the two triangulation options
    return std::abs(planarity_score1 - planarity_score2);
}

static int
select_best_scheme(const std::array<Point, 8> &vertices)
{
    // Calculate normal differences to find most non-planar triangulation
    double normal_diff_top_0_2 =
      calculate_normal_difference(vertices[0], vertices[1], vertices[2], vertices[3]);
    double normal_diff_top_1_3 =
      calculate_normal_difference(vertices[0], vertices[3], vertices[1], vertices[2]);

    // Do the same for bottom face
    double normal_diff_base_4_6 =
      calculate_normal_difference(vertices[4], vertices[5], vertices[6], vertices[7]);
    double normal_diff_base_5_7 =
      calculate_normal_difference(vertices[4], vertices[7], vertices[5], vertices[6]);

    // Choose diagonal with larger normal difference for top
    bool use_top_0_2 = normal_diff_top_0_2 > normal_diff_top_1_3;
    // Choose diagonal with larger normal difference for bottom
    bool use_base_4_6 = normal_diff_base_4_6 > normal_diff_base_5_7;

    // Determine scheme based on these choices
    if (use_top_0_2 && use_base_4_6)
        return 0;
    if (use_top_0_2 && !use_base_4_6)
        return 1;
    if (!use_top_0_2 && use_base_4_6)
        return 2;
    return 3;  // (!use_top_0_2 && !use_base_4_6)
}

/**
 * Determines if a point is inside a hexahedron (8-vertex cell) using the tetrahedrons.
 * The selection of tetrahedrons is based on the best scheme determined by
 * planarity and elevation differences.
 *
 * @param point The point to test
 * @param corners The 8 corners of the hexahedron
 * @return true if the point is inside the hexahedron, false otherwise
 */
static int
is_point_in_hexahedron_using_tetrahedrons_schemes(const Point &point_rh,
                                                  const std::array<Point, 8> &vertices)
{

    // Select the best scheme
    int best_scheme = select_best_scheme(vertices);

    // Use the selected scheme to check if the point is inside
    int result = 0;
    for (size_t i = 0; i < 6; ++i) {
        const auto &tetra = TETRAHEDRON_SCHEMES[best_scheme][i];
        int result =
          is_point_in_tetrahedron(point_rh, vertices[tetra[0]], vertices[tetra[1]],
                                  vertices[tetra[2]], vertices[tetra[3]]);

        // no need to check other tetrahedrons if we are inside
        if (result >= 1) {
            return result;
            break;
        }
    }

    if (result == -1) {
        // if last tetrahedron is degenerate, we return 0 anyway
        return 0;
    }

    return result;
}

// =====================================================================================
// Main function schemes based tetrahedrons
// The schemes should select the best one based on planarity difference (normal vector)
// =====================================================================================

/**
 * @brief A central function where one can select appropriate method
 * for point-in-cell test.
 * @param point The point to test, negated Z compared to python
 * @param corners The 8 corners of the hexahedron
 * @param method The method to use for the test
 * @return true if the point is inside the hexahedron, false otherwise
 * @throws std::invalid_argument if the method is not recognized
 */
bool
is_point_in_hexahedron_tetrahedrons_by_scheme(
  const Point &point,
  const HexahedronCorners &hexahedron_corners)
{

    std::array<Point, 8> vertices = {
        hexahedron_corners.upper_sw, hexahedron_corners.upper_se,
        hexahedron_corners.upper_ne, hexahedron_corners.upper_nw,
        hexahedron_corners.lower_sw, hexahedron_corners.lower_se,
        hexahedron_corners.lower_ne, hexahedron_corners.lower_nw
    };

    int result = is_point_in_hexahedron_using_tetrahedrons_schemes(point, vertices);
    return result >= 1;
}  // is_point_in_hexahedron

}  // namespace geometry
