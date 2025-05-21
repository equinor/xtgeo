// separate header file for structs etc used in xtgeo to avoid circular dependencies
#ifndef XTGEO_TYPES_HPP_
#define XTGEO_TYPES_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <array>
#include <cmath>
#include <cstddef>

#ifndef M_PI
#define M_PI 3.14159265358979323846  // seems like Windows does not define M_PI i cmath
#endif

// =====================================================================================
// NOTE CAREFULLY!
//
// XTGeo and C **may** (or not) use two different Z-coordinate conventions:
// - In Python: Z increases downward (geological depth convention), left-handed XYZ
// - In C++: Z follows normally python, but some structs are defined with Z
// increasing
//   upward (geometrical convention) -> right-handed XYZ
// =====================================================================================

namespace py = pybind11;

namespace xtgeo {

// =====================================================================================

namespace numerics {

constexpr double UNDEF_XTGEO = 10e32;
constexpr double UNDEF_DOUBLE = std::numeric_limits<double>::max();
constexpr double EPSILON = std::numeric_limits<double>::epsilon();  // 2.22e-16
constexpr double TOLERANCE = 1e-6;
constexpr double QUIET_NAN = std::numeric_limits<double>::quiet_NaN();

}  // namespace numerics

// =====================================================================================

namespace xyz {

struct Point
{
    // a single point in 3D space, Z is increasing down normally (but agnostic)
    double x;
    double y;
    double z;

    // Default constructor
    Point() : x(0), y(0), z(0) {}

    // Constructor that takes two arguments and sets z to 0
    Point(double x, double y) : x(x), y(y), z(0) {}

    // Constructor that takes three arguments
    Point(double x, double y, double z) : x(x), y(y), z(z) {}
};  // struct Point

struct Polygon
{
    std::vector<Point> points;

    // Default constructor (deleted)
    Polygon() = delete;

    // Constructor that takes a vector of points
    Polygon(const std::vector<Point> &points) : points(points) {}

    // Constructor that takes a 2D NumPy array with shape (N, 3)
    Polygon(const py::array_t<double> &array)
    {
        if (array.ndim() != 2 || array.shape(1) != 3) {
            throw std::runtime_error(
              "Input array must be a 2D array with shape (N, 3)");
        }

        auto r = array.unchecked<2>();  // Access the array without bounds checking
        for (size_t i = 0; i < r.shape(0); ++i) {
            points.emplace_back(r(i, 0), r(i, 1), r(i, 2));
        }
    }
    // Method to add a point to the polygon
    void add_point(const Point &point) { points.push_back(point); }

    // Method to get a point by index
    const Point &get_point(size_t index) const
    {
        if (index >= points.size()) {
            throw std::out_of_range("Index out of range");
        }
        return points[index];
    }

    // Method to get the number of points in the polygon
    size_t size() const { return points.size(); }

};  // struct Polygon

using PointSet = Polygon;  // alias for Polygon where the points are not ordered

}  // namespace xyz
// =====================================================================================

namespace geometry {
/*
 * These are geometrical representation that follow common rule; right handed with Z
 * increasing upward (in constrast to geological convention where Z is increasing
 * downward)
 *
 *
 * Need Grid3D HEXAHEDRON corners for a single Grid cell for some operations
 * The cell is a general hexahedron with 8 corners in space, organized as this
 * for "upper" (top) and "lower" (base). Elevation of "upper" is >= elevation to "lower"
 *
 *  nw ---- ne    nw refers to North-West corner, ne to North-East, etc.
 *  |       |
 *  |       |     Notice order is anticlock wise: sw - se - ne - nw
 *  sw ---- se    and DIFFERENT from CellCorners
 */

struct HexahedronCorners
{
    // The order of the corners is as follows:
    // upper corners: sw, se, ne, nw
    // lower corners: sw, se, ne, nw
    // The Z convention means that upper corners have higher Z values than lower corners

    xyz::Point upper_sw;
    xyz::Point upper_se;
    xyz::Point upper_ne;
    xyz::Point upper_nw;
    xyz::Point lower_sw;
    xyz::Point lower_se;
    xyz::Point lower_ne;
    xyz::Point lower_nw;

    // Default constructor
    HexahedronCorners() = delete;

    // Constructor that takes 8 xyz::Point objects
    HexahedronCorners(xyz::Point usw,
                      xyz::Point use,
                      xyz::Point une,
                      xyz::Point unw,
                      xyz::Point lsw,
                      xyz::Point lse,
                      xyz::Point lne,
                      xyz::Point lnw) :
      upper_sw(usw), upper_se(use), upper_ne(une), upper_nw(unw), lower_sw(lsw),
      lower_se(lse), lower_ne(lne), lower_nw(lnw)
    {
        validate_z_coordinates();
    }

    // Constructor that takes a one-dimensional numpy array of 24 elements
    HexahedronCorners(const py::array_t<double> &arr) :
      upper_sw(arr.at(0), arr.at(1), arr.at(2)),
      upper_se(arr.at(3), arr.at(4), arr.at(5)),
      upper_ne(arr.at(9), arr.at(10), arr.at(11)),
      upper_nw(arr.at(6), arr.at(7), arr.at(8)),
      lower_sw(arr.at(12), arr.at(13), arr.at(14)),
      lower_se(arr.at(15), arr.at(16), arr.at(17)),
      lower_ne(arr.at(18), arr.at(19), arr.at(20)),
      lower_nw(arr.at(21), arr.at(22), arr.at(23))
    {
        validate_z_coordinates();
    }

    // arrange the corners in a single array for easier access
    py::array_t<double> arrange_corners() const
    {
        // Create a NumPy array with shape (8, 3)
        py::array_t<double> arr({ 8, 3 });
        auto r =
          arr.mutable_unchecked<2>();  // Access the array without bounds checking

        // Fill the array with the corner coordinates
        r(0, 0) = upper_sw.x;
        r(0, 1) = upper_sw.y;
        r(0, 2) = upper_sw.z;

        r(1, 0) = upper_se.x;
        r(1, 1) = upper_se.y;
        r(1, 2) = upper_se.z;

        r(2, 0) = upper_ne.x;
        r(2, 1) = upper_ne.y;
        r(2, 2) = upper_ne.z;

        r(3, 0) = upper_nw.x;
        r(3, 1) = upper_nw.y;
        r(3, 2) = upper_nw.z;

        r(4, 0) = lower_sw.x;
        r(4, 1) = lower_sw.y;
        r(4, 2) = lower_sw.z;

        r(5, 0) = lower_se.x;
        r(5, 1) = lower_se.y;
        r(5, 2) = lower_se.z;

        r(6, 0) = lower_ne.x;
        r(6, 1) = lower_ne.y;
        r(6, 2) = lower_ne.z;

        r(7, 0) = lower_nw.x;
        r(7, 1) = lower_nw.y;
        r(7, 2) = lower_nw.z;

        return arr;
    }

private:
    // Helper function to validate Z-coordinates
    void validate_z_coordinates() const
    {
        assert(upper_sw.z >= lower_sw.z && "upper_sw.z must be >= lower_sw.z");
        assert(upper_se.z >= lower_se.z && "upper_se.z must be >= lower_se.z");
        assert(upper_ne.z >= lower_ne.z && "upper_ne.z must be >= lower_ne.z");
        assert(upper_nw.z >= lower_nw.z && "upper_nw.z must be >= lower_nw.z");
    }
};  // struct HexahedronCorners

}  // namespace geometry

// =====================================================================================

namespace grid3d {

/*
 * Need Grid3D cell corners for a single Grid cell for some operations
 * The cell is a general hexahedron with 8 corners in space, organized as this
 * for "upper" (top) and "lower" (base). Depth to "upper" is <= depth to "lower"
 *
 *  nw ---- ne    nw refers to North-West corner, ne to North-East, etc.
 *  |       |
 *  |       |     Notice order: sw - se - nw - ne
 *  sw ---- se
 */

struct CellCorners
{
    // For cellcorners, Z is increasing upward in C++! For most North Sea cases it means
    // that Z values are "negative" depths
    // The order of the corners is as follows:
    // upper corners: sw, se, nw, ne
    // lower corners: sw, se, nw, ne
    // The Z convention mens that upper corners have higher Z values than lower corners

    xyz::Point upper_sw;
    xyz::Point upper_se;
    xyz::Point upper_nw;
    xyz::Point upper_ne;
    xyz::Point lower_sw;
    xyz::Point lower_se;
    xyz::Point lower_nw;
    xyz::Point lower_ne;

    // Default constructor
    CellCorners() = default;

    // Constructor that takes 8 xyz::Point objects
    CellCorners(xyz::Point usw,
                xyz::Point use,
                xyz::Point unw,
                xyz::Point une,
                xyz::Point lsw,
                xyz::Point lse,
                xyz::Point lnw,
                xyz::Point lne) :
      upper_sw(usw), upper_se(use), upper_nw(unw), upper_ne(une), lower_sw(lsw),
      lower_se(lse), lower_nw(lnw), lower_ne(lne)
    {
    }

    // Constructor that takes a one-dimensional numpy array of 24 elements
    CellCorners(const py::array_t<double> &arr) :
      upper_sw(arr.at(0), arr.at(1), arr.at(2)),
      upper_se(arr.at(3), arr.at(4), arr.at(5)),
      upper_nw(arr.at(6), arr.at(7), arr.at(8)),
      upper_ne(arr.at(9), arr.at(10), arr.at(11)),
      lower_sw(arr.at(12), arr.at(13), arr.at(14)),
      lower_se(arr.at(15), arr.at(16), arr.at(17)),
      lower_nw(arr.at(18), arr.at(19), arr.at(20)),
      lower_ne(arr.at(21), arr.at(22), arr.at(23))
    {
    }

    // Constructor that takes a one-dimensional array of 24 elements
    CellCorners(const std::array<double, 24> &arr) :
      upper_sw(arr[0], arr[1], arr[2]), upper_se(arr[3], arr[4], arr[5]),
      upper_nw(arr[6], arr[7], arr[8]), upper_ne(arr[9], arr[10], arr[11]),
      lower_sw(arr[12], arr[13], arr[14]), lower_se(arr[15], arr[16], arr[17]),
      lower_nw(arr[18], arr[19], arr[20]), lower_ne(arr[21], arr[22], arr[23])
    {
    }

    // arrange the corners in a single array for easier access in some cases
    std::array<double, 24> arrange_corners() const
    {
        return {
            upper_sw.x, upper_sw.y, upper_sw.z, upper_se.x, upper_se.y, upper_se.z,
            upper_nw.x, upper_nw.y, upper_nw.z, upper_ne.x, upper_ne.y, upper_ne.z,
            lower_sw.x, lower_sw.y, lower_sw.z, lower_se.x, lower_se.y, lower_se.z,
            lower_nw.x, lower_nw.y, lower_nw.z, lower_ne.x, lower_ne.y, lower_ne.z
        };
    }

    // convert CellCorners struct to a std::array<double, 24> array

    py::array_t<double> to_numpy() const
    {
        auto result = py::array_t<double>({ 8, 3 });
        auto result_m = result.mutable_unchecked<2>();
        auto corners = arrange_corners();
        for (size_t i = 0; i < 8; i++) {
            result_m(i, 0) = corners[3 * i];
            result_m(i, 1) = corners[3 * i + 1];
            result_m(i, 2) = corners[3 * i + 2];
        }
        return result;
    }

    // Method to convert CellCorners to HexahedronCorners with Z multiplied by -1
    // and change order to match the HexahedronCorners constructor sw - se - ne - nw
    xtgeo::geometry::HexahedronCorners to_hexahedron_corners() const
    {
        auto negate_z = [](const xyz::Point &point) {
            return xyz::Point(point.x, point.y, -point.z);
        };

        return xtgeo::geometry::HexahedronCorners(
          negate_z(upper_sw), negate_z(upper_se), negate_z(upper_ne),
          negate_z(upper_nw), negate_z(lower_sw), negate_z(lower_se),
          negate_z(lower_ne), negate_z(lower_nw));
    }

};  // struct CellCorners

struct Grid
{

    // Also for a 3D grid, Z is increasing upward in C++! For most North Sea cases it
    // means that Z values are "negative" depths

    size_t ncol;
    size_t nrow;
    size_t nlay;
    py::array_t<double> coordsv;
    py::array_t<float> zcornsv;
    py::array_t<int> actnumsv;

    mutable std::vector<CellCorners> cell_corners_cache;

    // Default constructor (deleted)
    Grid() = delete;

    // Constructor that takes a Python xtgeo.Grid object (using a subset of the attrs)
    Grid(const py::object &grid)
    {
        ncol = grid.attr("ncol").cast<size_t>();
        nrow = grid.attr("nrow").cast<size_t>();
        nlay = grid.attr("nlay").cast<size_t>();
        coordsv = grid.attr("_coordsv").cast<py::array_t<double>>();
        zcornsv = grid.attr("_zcornsv").cast<py::array_t<float>>();
        actnumsv = grid.attr("_actnumsv").cast<py::array_t<int>>();

        // Add the cache as a member

        // check dimensions for coords that should be (ncol+1, nrow+1, 6)
        if (coordsv.ndim() != 3) {
            throw std::runtime_error("coordsv should have 3 dimensions");
        }
        if (coordsv.shape(0) != ncol + 1 || coordsv.shape(1) != nrow + 1 ||
            coordsv.shape(2) != 6) {
            throw std::runtime_error("coordsv should have shape (ncol+1, nrow+1, 6)");
        }

        // check dimensions for zcornsv that should be (ncol+1, nrow+1, nlay+1, 4)
        if (zcornsv.ndim() != 4) {
            throw std::runtime_error("zcornsv should have 4 dimensions");
        }
        if (zcornsv.shape(0) != ncol + 1 || zcornsv.shape(1) != nrow + 1 ||
            zcornsv.shape(2) != nlay + 1 || zcornsv.shape(3) != 4) {
            throw std::runtime_error(
              "zcornsv should have shape (ncol+1, nrow+1, nlay+1, 4)");
        }

        // check dimensions for actnumsv that should be (ncol, nrow, nlay)
        if (actnumsv.ndim() != 3) {
            throw std::runtime_error("actnumsv should have 3 dimensions");
        }
        if (actnumsv.shape(0) != ncol || actnumsv.shape(1) != nrow ||
            actnumsv.shape(2) != nlay) {
            throw std::runtime_error("actnumsv should have shape (ncol, nrow, nlay)");
        }
    };

    void compute_cell_corners();

    void ensure_cell_corners_cache() const;
};

}  // namespace grid3d

// =====================================================================================

namespace regsurf {

struct RegularSurface
{
    size_t ncol;
    size_t nrow;
    double xori;
    double yori;
    double xinc;
    double yinc;
    double rotation;
    py::array_t<double> values;
    py::array_t<bool> mask;

    // Default constructor (deleted)
    RegularSurface() = delete;

    // Constructor that takes a subset of the attributes
    RegularSurface(size_t ncol,
                   size_t nrow,
                   double xori,
                   double yori,
                   double xinc,
                   double yinc,
                   double rotation) :
      ncol(ncol), nrow(nrow), xori(xori), yori(yori), xinc(xinc), yinc(yinc),
      rotation(rotation)
    {
    }

    // Constructor that takes a Python object and a skip_values flag
    RegularSurface(const py::object &rs)
    {
        ncol = rs.attr("ncol").cast<size_t>();
        nrow = rs.attr("nrow").cast<size_t>();
        xori = rs.attr("xori").cast<double>();
        yori = rs.attr("yori").cast<double>();
        xinc = rs.attr("xinc").cast<double>();
        yinc = rs.attr("yinc").cast<double>();
        rotation = rs.attr("rotation").cast<double>();

        // Extract the masked numpy array
        py::object np_values_masked = rs.attr("values");
        values = np_values_masked.attr("data").cast<py::array_t<double>>();
        mask = np_values_masked.attr("mask").cast<py::array_t<bool>>();

        py::buffer_info buf_info_values = values.request();
        py::buffer_info buf_info_mask = mask.request();

        if (buf_info_values.ndim != 2 || buf_info_mask.ndim != 2) {
            throw std::runtime_error(
              "RegularSurface data values and mask must be 2D numpy arrays");
        }
    };

};  // struct RegularSurface

}  // namespace regsurf
namespace cube {

struct Cube
{
    size_t ncol;
    size_t nrow;
    size_t nlay;
    double xori;
    double yori;
    double zori;
    double xinc;
    double yinc;
    double zinc;
    double rotation;
    py::array_t<float> values;

    // Default constructor (deleted)
    Cube() = delete;

    // Constructor that takes a Python object (using a subset of the attributes)
    Cube(const py::object &cube)
    {
        ncol = cube.attr("ncol").cast<size_t>();
        nrow = cube.attr("nrow").cast<size_t>();
        nlay = cube.attr("nlay").cast<size_t>();
        xori = cube.attr("xori").cast<double>();
        yori = cube.attr("yori").cast<double>();
        zori = cube.attr("zori").cast<double>();
        xinc = cube.attr("xinc").cast<double>();
        yinc = cube.attr("yinc").cast<double>();
        zinc = cube.attr("zinc").cast<double>();
        rotation = cube.attr("rotation").cast<double>();

        // Extract the numpy array
        values = cube.attr("values").cast<py::array_t<float>>();

        py::buffer_info buf_info_values = values.request();

        if (buf_info_values.ndim != 3) {
            throw std::runtime_error("Cube values must be a 3D numpy array");
        }
    };
};
}  // namespace cube

}  // namespace xtgeo

#endif  // XTGEO_TYPES_HPP_
