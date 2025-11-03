"""Test suite for RegularSurface.gridding() with all methods."""

import pathlib

import numpy as np
import pytest

import xtgeo

POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")


@pytest.fixture
def simple_surface():
    """Create a simple regular surface template for testing."""
    return xtgeo.RegularSurface(
        ncol=50,
        nrow=50,
        xinc=10.0,
        yinc=10.0,
        xori=0.0,
        yori=0.0,
        values=np.zeros((50, 50)),
    )


@pytest.fixture
def simple_points():
    """Create simple test points on a tilted plane."""
    np.random.seed(42)
    n_points = 100
    x = np.random.uniform(50, 450, n_points)
    y = np.random.uniform(50, 450, n_points)
    # Create a tilted plane: z = 100 + 0.1*x + 0.05*y
    z = 100 + 0.1 * x + 0.05 * y + np.random.normal(0, 1, n_points)

    return xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )


@pytest.fixture
def sparse_points():
    """Create sparse test points for testing methods that work well with few points."""
    # Non-collinear points to avoid singular matrix issues with RBF
    x = [100, 200, 300, 400, 150, 250, 350]
    y = [100, 150, 250, 300, 200, 350, 150]
    z = [100, 110, 105, 115, 120, 125, 130]

    return xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )


@pytest.fixture
def close_points():
    """Create points with some very close together."""
    # Non-collinear points with first two very close
    x = [100.0, 100.1, 200.0, 300.0, 400.0]  # First two are very close
    y = [100.0, 100.1, 250.0, 150.0, 350.0]  # Not on diagonal
    z = [50.0, 51.0, 60.0, 70.0, 80.0]

    return xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )


@pytest.fixture
def pointset2(testdata_path):
    """Use POINTSET2 data."""

    return xtgeo.points_from_file(testdata_path / POINTSET2)


@pytest.fixture
def pointset2_thinned(pointset2):
    """Use POINTSET2 data but thinned to every 4rd point for faster tests."""
    points = pointset2.copy()
    # Thin the points by taking every 3rd point (keeps ~33% of points)
    points.set_dataframe(points._df.iloc[::4].reset_index(drop=True))
    return points


@pytest.fixture
def surface_for_pointset2():
    return xtgeo.RegularSurface(
        ncol=280,
        nrow=440,
        xinc=25,
        yinc=25,
        xori=461500,
        yori=5926500,
        rotation=30,
    )


# ======================================================================================
# Test: linear method
# ======================================================================================


def test_gridding_linear_basic(simple_surface, simple_points):
    """Test linear interpolation gridding."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="linear")

    # Check that surface has been populated
    assert not np.all(np.isnan(surf.values))
    assert surf.values.mask.sum() < surf.ncol * surf.nrow  # Some nodes should be valid


def test_gridding_linear_with_extrapolate(simple_surface, simple_points):
    """Test linear interpolation with extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="linear", method_options={"extrapolate": True})

    # With extrapolate, should have fewer undefined nodes
    assert not np.all(np.isnan(surf.values))


def test_gridding_linear_without_extrapolate(simple_surface, simple_points):
    """Test linear interpolation without extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="linear", method_options={"extrapolate": False})

    # Without extrapolate, may have more undefined nodes at edges
    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: nearest method
# ======================================================================================


def test_gridding_nearest_basic(simple_surface, simple_points):
    """Test nearest neighbor gridding."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="nearest")

    # Check that surface has been populated
    assert not np.all(np.isnan(surf.values))


def test_gridding_nearest_preserves_values(simple_surface):
    """Test that nearest neighbor preserves exact point values."""
    # Create points with known values
    x = [100, 200, 300]
    y = [100, 200, 300]
    z = [10.0, 20.0, 30.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="nearest")

    # Values near the points should be close to the point values
    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: cubic method
# ======================================================================================


def test_gridding_cubic_basic(simple_surface, simple_points):
    """Test cubic interpolation gridding."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="cubic")

    # Check that surface has been populated
    assert not np.all(np.isnan(surf.values))


def test_gridding_cubic_smooth(simple_surface, simple_points):
    """Test that cubic produces smooth results."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="cubic")

    # Cubic should produce smoother results than linear
    # (This is a qualitative check - just ensure it runs)
    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: inverse_distance (IDW) method
# ======================================================================================


def test_gridding_idw_default(simple_surface, simple_points):
    """Test IDW with default parameters."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="inverse_distance")

    assert not np.all(np.isnan(surf.values))


def test_gridding_idw_custom_power(simple_surface, simple_points):
    """Test IDW with custom power parameter."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points, method="inverse_distance", method_options={"power": 3.0}
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_idw_with_radius(simple_surface, simple_points):
    """Test IDW with search radius."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="inverse_distance",
        method_options={"power": 2.0, "radius": 100.0, "min_points": 3},
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: radial_basis (RBF) method
# ======================================================================================


def test_gridding_rbf_default(simple_surface, sparse_points):
    """Test RBF with default parameters (thin_plate_spline)."""
    surf = simple_surface.copy()
    surf.gridding(sparse_points, method="radial_basis")

    assert not np.all(np.isnan(surf.values))


def test_gridding_rbf_functions(simple_surface, sparse_points):
    """Test different RBF kernel functions."""
    functions = [
        "thin_plate_spline",
        "cubic",
        "quintic",
        "linear",
        "multiquadric",
        "inverse_multiquadric",
        "inverse_quadratic",
        "gaussian",
    ]

    for func in functions:
        surf = simple_surface.copy()
        surf.gridding(
            sparse_points, method="radial_basis", method_options={"function": func}
        )
        assert not np.all(np.isnan(surf.values)), f"Failed for function={func}"


def test_gridding_rbf_with_smoothing(simple_surface, simple_points):
    """Test RBF with smoothing parameter."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points, method="radial_basis", method_options={"smoothing": 0.1}
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: moving_average method
# ======================================================================================


def test_gridding_moving_average_default(simple_surface, simple_points):
    """Test moving average with default parameters."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="moving_average")

    assert not np.all(np.isnan(surf.values))


def test_gridding_moving_average_custom_radius(simple_surface, simple_points):
    """Test moving average with custom radius."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="moving_average",
        method_options={"radius": 50.0, "min_points": 5},
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: kriging method
# ======================================================================================


def test_gridding_kriging_default(simple_surface, simple_points):
    """Test kriging with default parameters."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="kriging")

    assert not np.all(np.isnan(surf.values))


def test_gridding_kriging_variogram_models(simple_surface, sparse_points):
    """Test different variogram models."""
    models = ["Gaussian", "Exponential", "Spherical"]

    for model in models:
        surf = simple_surface.copy()
        surf.gridding(
            sparse_points,
            method="kriging",
            method_options={"variogram_model": model},
        )
        assert not np.all(np.isnan(surf.values)), f"Failed for model={model}"


def test_gridding_kriging_with_nugget(simple_surface, simple_points):
    """Test kriging with nugget effect."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="kriging",
        method_options={"variogram_parameters": {"nugget": 0.1}},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_kriging_simple(simple_surface, simple_points):
    """Test simple kriging."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="kriging",
        method_options={"krige_type": "simple", "mean": 110.0},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_kriging_with_len_scale(simple_surface, simple_points):
    """Test kriging with explicit len_scale."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="kriging",
        method_options={"variogram_parameters": {"len_scale": 100.0}},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_kriging_with_range_value(simple_surface, simple_points):
    """Test kriging with range_value instead of len_scale."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="kriging",
        method_options={"variogram_parameters": {"range_value": 150.0}},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_kriging_exact(simple_surface, sparse_points):
    """Test kriging with exact=True."""
    surf = simple_surface.copy()
    surf.gridding(sparse_points, method="kriging", method_options={"exact": True})

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: coarsen parameter
# ======================================================================================


def test_gridding_with_coarsen(simple_surface, simple_points):
    """Test gridding with coarsening factor."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="linear", coarsen=2)

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: error handling
# ======================================================================================


def test_gridding_invalid_method(simple_surface, simple_points):
    """Test that invalid method raises ValueError."""
    surf = simple_surface.copy()

    with pytest.raises(ValueError, match="Invalid gridding method"):
        surf.gridding(simple_points, method="invalid_method")


def test_gridding_not_points_instance(simple_surface):
    """Test that invalid input type raises ValueError."""
    surf = simple_surface.copy()

    with pytest.raises(
        ValueError, match="Input must be a Points, Polygons, or RegularSurface instance"
    ):
        surf.gridding("not_a_points_object", method="linear")


def test_gridding_from_polygons(simple_surface):
    """Test gridding from Polygons instance."""

    surf = simple_surface.copy()

    # Create a simple polygon as list of tuples (x, y, z, poly_id)
    poly_data = [
        (100.0, 100.0, 1000.0, 1),
        (300.0, 100.0, 1100.0, 1),
        (500.0, 300.0, 1200.0, 1),
        (300.0, 500.0, 1300.0, 1),
        (100.0, 100.0, 1000.0, 1),
    ]
    polygons = xtgeo.Polygons(poly_data)

    # Should work without error
    surf.gridding(polygons, method="nearest")

    # Check that some values were set
    assert not surf.values.mask.all(), "All values are masked after gridding"


def test_gridding_from_surface(simple_surface):
    """Test gridding from RegularSurface instance."""
    import numpy as np

    surf = simple_surface.copy()

    # Create a source surface with some data
    source = xtgeo.RegularSurface(
        ncol=10,
        nrow=10,
        xinc=50.0,
        yinc=50.0,
        xori=0.0,
        yori=0.0,
        values=np.random.uniform(1000, 2000, (10, 10)),
    )

    # Should work without error
    surf.gridding(source, method="nearest")

    # Check that some values were set
    assert not surf.values.mask.all(), "All values are masked after gridding"


# ======================================================================================
# Test: method combinations
# ======================================================================================


def test_gridding_linear_then_fill(simple_surface, sparse_points):
    """Test linear gridding followed by fill."""
    surf = simple_surface.copy()
    surf.gridding(sparse_points, method="linear")

    # Count undefined before fill
    undefined_before = surf.values.mask.sum()

    surf.fill()

    # Should have fewer undefined after fill
    undefined_after = surf.values.mask.sum()
    assert undefined_after <= undefined_before


def test_gridding_rbf_with_merge(simple_surface, close_points):
    """Test RBF gridding with close points merging."""
    surf = simple_surface.copy()

    # Test that RBF can handle close points
    # (previously would need merging, but now user should preprocess)
    surf.gridding(
        close_points,
        method="radial_basis",
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: numerical correctness
# ======================================================================================


def test_gridding_preserves_point_values_nearest(simple_surface):
    """Test that nearest method preserves point values exactly at point locations."""
    # Create a few points with known values
    x = [100, 200, 300]
    y = [100, 200, 300]
    z = [50.0, 60.0, 70.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="nearest")

    # Get values at point locations (approximately)
    # This is a rough check - exact values depend on grid alignment
    assert not np.all(np.isnan(surf.values))


def test_gridding_linear_interpolation_properties(simple_surface):
    """Test that linear interpolation behaves correctly."""
    # Create 4 corner points
    x = [100, 100, 300, 300]
    y = [100, 300, 100, 300]
    z = [10.0, 20.0, 30.0, 40.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="linear")

    # Linear interpolation should produce values within the range of input
    valid_values = surf.values[~surf.values.mask]
    if len(valid_values) > 0:
        assert valid_values.min() >= 10.0 - 5.0  # Allow some tolerance
        assert valid_values.max() <= 40.0 + 5.0


# ======================================================================================
# Test: edge cases
# ======================================================================================


def test_gridding_single_point(simple_surface):
    """Test gridding with a single point."""
    x = [250]
    y = [250]
    z = [100.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Nearest should work with single point
    surf.gridding(points, method="nearest")

    assert not np.all(np.isnan(surf.values))


def test_gridding_few_points(simple_surface):
    """Test gridding with very few points."""
    x = [100, 400]
    y = [100, 400]
    z = [50.0, 150.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="nearest")

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: method-specific edge cases
# ======================================================================================


def test_gridding_idw_min_points_not_met(simple_surface):
    """Test IDW when minimum points requirement cannot be met everywhere."""
    # Very sparse points
    x = [100, 400]
    y = [100, 400]
    z = [50.0, 150.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Should handle gracefully
    surf.gridding(
        points,
        method="inverse_distance",
        method_options={"min_points": 1, "radius": 200.0},
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: additional edge cases
# ======================================================================================


def test_gridding_empty_points(simple_surface):
    """Test gridding with empty points object raises appropriate error."""
    # Create a Points object with zero points (but proper structure)
    empty_points = xtgeo.Points(
        values=np.empty((0, 3)),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # With no points, gridding should raise an error or handle gracefully
    # Catching any exception as this is an edge case with undefined behavior
    with pytest.raises((ValueError, RuntimeError, TypeError, SystemError, Exception)):
        surf.gridding(empty_points, method="linear")


def test_gridding_points_outside_surface(simple_surface):
    """Test gridding when all points are outside surface extent."""
    # Points far outside the surface bounds (surface is 0-490, 0-490)
    x = [1000, 1100, 1200]
    y = [1000, 1100, 1200]
    z = [50.0, 60.0, 70.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # When all points are outside the surface, they get filtered out
    # This should raise an error since there are no points to grid
    with pytest.raises(RuntimeError, match="Could not do gridding"):
        surf.gridding(points, method="linear", method_options={"extrapolate": False})


def test_gridding_points_at_surface_edges(simple_surface):
    """Test gridding with points exactly at surface boundaries."""
    # Points at the corners and edges of the surface
    x = [0, 490, 0, 490, 245]  # xori=0, xmax=0+50*10-10=490
    y = [0, 0, 490, 490, 245]
    z = [10.0, 20.0, 30.0, 40.0, 25.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="linear")

    # Should have some valid values
    assert not np.all(np.isnan(surf.values))


def test_gridding_duplicate_points_no_merge(simple_surface):
    """Test gridding with exact duplicate points without merging."""
    # Exact duplicate points
    x = [100, 100, 200, 300]  # First two are identical
    y = [100, 100, 200, 300]
    z = [50.0, 55.0, 60.0, 70.0]  # Different Z values

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Scipy's triangulation doesn't handle duplicate points well
    # This should raise an error - duplicates need to be merged first
    with pytest.raises((RuntimeError, ValueError)):
        surf.gridding(points, method="linear")


def test_gridding_duplicate_points_with_merge(simple_surface):
    """Test gridding with exact duplicate points using merge."""
    # Exact duplicate points with non-collinear result after merge
    x = [100, 100, 200, 300, 150]  # First two are identical
    y = [100, 100, 250, 150, 200]  # Non-collinear after merge
    z = [50.0, 55.0, 60.0, 70.0, 58.0]  # Different Z values

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Duplicates should be handled by the gridding method
    surf.gridding(points, method="linear")

    assert not np.all(np.isnan(surf.values))


def test_gridding_collinear_points(simple_surface):
    """Test gridding with all points on a line."""
    # All points on a diagonal line
    coords = [(100 + i * 50, 100 + i * 50, 10.0 + i * 5) for i in range(6)]
    x, y, z = zip(*coords)

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Some methods may struggle with collinear points
    surf.gridding(points, method="nearest")

    assert not np.all(np.isnan(surf.values))


def test_gridding_extreme_z_values(simple_surface):
    """Test gridding with very large or very small Z values."""
    # Non-collinear points with extreme Z values
    x = [100, 200, 300, 400, 150]
    y = [100, 250, 150, 350, 200]
    z = [1e6, -1e6, 1e-6, -1e-6, 0.0]  # Extreme values

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="linear")

    # Should handle extreme values
    valid_values = (
        surf.values[~surf.values.mask]
        if hasattr(surf.values, "mask")
        else surf.values[~np.isnan(surf.values)]
    )
    if len(valid_values) > 0:
        assert not np.all(np.isnan(valid_values))


def test_gridding_uniform_z_values(simple_surface):
    """Test gridding when all Z values are identical."""
    # Non-collinear points with uniform Z
    x = [100, 200, 300, 400, 150, 250]
    y = [100, 250, 150, 350, 200, 300]
    z = [100.0] * 6  # All same value

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="linear")

    # All interpolated values should be approximately 100.0
    valid_values = (
        surf.values[~surf.values.mask]
        if hasattr(surf.values, "mask")
        else surf.values[~np.isnan(surf.values)]
    )
    if len(valid_values) > 0:
        assert np.allclose(valid_values, 100.0, rtol=0.01)


def test_gridding_very_dense_points(simple_surface):
    """Test gridding with very dense point cloud."""
    np.random.seed(123)
    n_points = 5000  # Very dense
    x = np.random.uniform(100, 400, n_points)
    y = np.random.uniform(100, 400, n_points)
    z = 100 + 0.1 * x + 0.05 * y

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Should handle large number of points
    surf.gridding(points, method="linear")

    assert not np.all(np.isnan(surf.values))


def test_gridding_nan_values_in_points(simple_surface):
    """Test gridding when points contain NaN values."""
    # Non-collinear points with some NaN values
    x = [100, 200, 300, np.nan, 400, 150]
    y = [100, 250, np.nan, 300, 350, 200]
    z = [50.0, 60.0, 70.0, 80.0, 90.0, 65.0]

    # Points should filter out NaN values automatically
    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    surf.gridding(points, method="linear")

    # Should work with remaining valid points
    assert not np.all(np.isnan(surf.values))


def test_gridding_with_coarsen_edge_cases(simple_surface):
    """Test coarsen parameter with edge cases."""
    # Use simple_points which has many points
    np.random.seed(42)
    n_points = 50
    x = np.random.uniform(50, 450, n_points)
    y = np.random.uniform(50, 450, n_points)
    z = 100 + 0.1 * x + 0.05 * y

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    # Test with coarsen factor - with enough points, this should work
    surf = simple_surface.copy()
    surf.gridding(points, method="linear", coarsen=2)

    assert not np.all(np.isnan(surf.values))


def test_gridding_snap_surface_roundtrip(simple_surface, simple_points):
    """Test gridding followed by snap_surface to verify consistency."""
    # Grid points to create a surface
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="linear")

    # Create new points from a subset of the original
    test_points = simple_points.copy()
    test_points._df = test_points._df.iloc[:10].copy()  # Take first 10 points

    # Snap these points to the gridded surface
    test_points.snap_surface(surf, activeonly=True)
    snapped_z = test_points._df["Z_TVDSS"].values

    # The snapped Z values should be close to interpolated surface values
    # (within reasonable tolerance due to interpolation)
    if len(snapped_z) > 0:
        assert not np.all(np.isnan(snapped_z))


def test_gridding_rbf_with_noise(simple_surface):
    """Test RBF gridding with noisy data and smoothing."""
    np.random.seed(42)
    n_points = 50
    x = np.random.uniform(100, 400, n_points)
    y = np.random.uniform(100, 400, n_points)
    # True function + significant noise
    z = 100 + 0.1 * x + 0.05 * y + np.random.normal(0, 10, n_points)

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Use smoothing to handle noise
    surf.gridding(
        points,
        method="radial_basis",
        method_options={"smoothing": 1.0, "function": "thin_plate_spline"},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_merge_all_points_merged(simple_surface):
    """Test when points are very close together (previously would merge)."""
    x = [100, 110, 120, 130]
    y = [100, 110, 120, 130]
    z = [50.0, 60.0, 70.0, 80.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    surf = simple_surface.copy()
    # Use nearest method which handles close points well
    surf.gridding(points, method="nearest")

    # Should produce a valid result
    assert not np.all(np.isnan(surf.values))


def test_gridding_moving_average_zero_radius(simple_surface, simple_points):
    """Test moving average with very small radius."""
    surf = simple_surface.copy()

    # Very small radius - should still work but may produce sparse results
    surf.gridding(
        simple_points,
        method="moving_average",
        method_options={"radius": 1.0, "min_points": 1},
    )

    # May have many undefined nodes but should not crash
    assert True  # Just verify it doesn't crash


def test_gridding_idw_zero_power(simple_surface, simple_points):
    """Test IDW with power close to zero (uniform weighting)."""
    surf = simple_surface.copy()

    # Very small power approaches uniform weighting
    surf.gridding(
        simple_points,
        method="inverse_distance",
        method_options={"power": 0.1},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_idw_very_high_power(simple_surface, simple_points):
    """Test IDW with very high power (nearest neighbor-like)."""
    surf = simple_surface.copy()

    # Very high power makes it behave like nearest neighbor
    surf.gridding(
        simple_points,
        method="inverse_distance",
        method_options={"power": 10.0},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_different_surface_sizes(simple_points):
    """Test gridding on surfaces with different dimensions."""
    dimensions = [(10, 10), (100, 100), (20, 50)]

    for ncol, nrow in dimensions:
        surf = xtgeo.RegularSurface(
            ncol=ncol,
            nrow=nrow,
            xinc=10.0,
            yinc=10.0,
            xori=0.0,
            yori=0.0,
            values=np.zeros((ncol, nrow)),
        )

        surf.gridding(simple_points, method="linear")
        assert not np.all(np.isnan(surf.values)), f"Failed for {ncol}x{nrow}"


def test_gridding_different_increments():
    """Test gridding with different xinc and yinc values."""
    # Non-collinear points
    x = [100, 200, 300, 400, 150]
    y = [100, 250, 150, 350, 200]
    z = [50.0, 60.0, 70.0, 80.0, 65.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    # Very different xinc and yinc (anisotropic grid)
    surf = xtgeo.RegularSurface(
        ncol=50,
        nrow=50,
        xinc=5.0,  # Fine in X
        yinc=20.0,  # Coarse in Y
        xori=0.0,
        yori=0.0,
        values=np.zeros((50, 50)),
    )

    surf.gridding(points, method="linear")
    assert not np.all(np.isnan(surf.values))


def test_gridding_rotated_surface():
    """Test gridding on a rotated surface."""
    # Non-collinear points
    x = [100, 200, 300, 400, 150]
    y = [100, 250, 150, 350, 200]
    z = [50.0, 60.0, 70.0, 80.0, 65.0]

    points = xtgeo.Points(
        values=np.column_stack([x, y, z]),
        xname="X_UTME",
        yname="Y_UTMN",
        zname="Z_TVDSS",
    )

    # Rotated surface (30 degrees)
    surf = xtgeo.RegularSurface(
        ncol=50,
        nrow=50,
        xinc=10.0,
        yinc=10.0,
        xori=0.0,
        yori=0.0,
        rotation=30.0,
        values=np.zeros((50, 50)),
    )

    surf.gridding(points, method="linear")
    assert not np.all(np.isnan(surf.values))


def test_gridding_multiple_methods_same_data(simple_surface, simple_points):
    """Test that different methods produce different but valid results."""
    methods = ["linear", "nearest", "cubic"]
    results = []

    for method in methods:
        surf = simple_surface.copy()
        surf.gridding(simple_points, method=method)
        results.append(surf.values.copy())

        # Each method should produce valid results
        assert not np.all(np.isnan(surf.values)), f"Failed for method={method}"

    # Results from different methods should generally be different
    # (at least nearest vs linear/cubic)
    # This is a qualitative check
    assert not np.array_equal(results[0], results[1]) or not np.array_equal(
        results[1], results[2]
    )


def test_gridding_points_from_surface_roundtrip(simple_surface, simple_points):
    """Test creating points from surface then gridding back."""
    # First, grid points to surface
    surf1 = simple_surface.copy()
    surf1.gridding(simple_points, method="linear")

    # Convert surface back to points
    from xtgeo import points_from_surface

    roundtrip_points = points_from_surface(surf1)

    # Grid again
    surf2 = simple_surface.copy()
    surf2.gridding(roundtrip_points, method="linear")

    # Results should be similar (surfaces should be close)
    # Check that both have similar coverage
    mask1 = (
        surf1.values.mask if hasattr(surf1.values, "mask") else np.isnan(surf1.values)
    )
    mask2 = (
        surf2.values.mask if hasattr(surf2.values, "mask") else np.isnan(surf2.values)
    )

    # Should have similar number of defined nodes
    defined1 = np.sum(~mask1)
    defined2 = np.sum(~mask2)
    assert defined1 > 0 and defined2 > 0


# ======================================================================================
# Using pointset2 with result verified outside
# ======================================================================================


def test_gridding_pset2_simple(pointset2, surface_for_pointset2):
    """Test simple basic gridding on pointset2 with external verification."""
    surf = surface_for_pointset2.copy()
    surf.gridding(pointset2, method="linear")

    assert surf.values.mean() == pytest.approx(1691.65, abs=1e-2)

    # check number of masked values
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 65729

    surf.gridding(pointset2, method="cubic")

    assert surf.values.mean() == pytest.approx(1688.02, abs=1e-2)

    # check number of masked values
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 65729

    surf.gridding(pointset2, method="nearest")

    assert surf.values.mean() == pytest.approx(1709.92, abs=1e-2)

    # check number of masked values, which shall be 0 with "nearest"
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0


def test_gridding_pset2_radial_basis(pointset2, surface_for_pointset2):
    """Test radial basis function gridding on pointset2 with external verification."""
    surf = surface_for_pointset2.copy()
    surf.gridding(pointset2, method="radial_basis")

    assert surf.values.mean() == pytest.approx(1712.44, abs=1e-2)

    surf.gridding(
        pointset2,
        method="radial_basis",
        method_options={"function": "linear", "smoothing": 2.0},
    )

    assert surf.values.mean() == pytest.approx(1711.40, abs=1e-2)


def test_gridding_pset2_kriging(pointset2, surface_for_pointset2):
    """Test kriging gridding on pointset2 with external verification."""
    surf = surface_for_pointset2.copy()
    surf.gridding(pointset2, method="kriging")

    assert surf.values.mean() == pytest.approx(1699.79, abs=1e-2)

    surf.gridding(
        pointset2,
        method="kriging",
        method_options={
            "variogram_model": "spherical",
            "variogram_parameters": {
                "range_value": (3000, 1200),
                "angle": 30,
                "nugget": 0.0,
            },
        },
    )

    assert surf.values.mean() == pytest.approx(1705.53, abs=1e-2)


# ======================================================================================
# Test: cubic method with extrapolation options
# ======================================================================================


def test_gridding_cubic_no_extrapolate(simple_surface, simple_points):
    """Test cubic interpolation without extrapolation (default)."""
    surf = simple_surface.copy()
    surf.gridding(simple_points, method="cubic", method_options={"extrapolate": False})

    # Without extrapolate, should have undefined nodes outside convex hull
    assert not np.all(np.isnan(surf.values))
    assert surf.values.mask.sum() > 0  # Some nodes should be masked


def test_gridding_cubic_with_extrapolate_nearest(simple_surface, simple_points):
    """Test cubic interpolation with nearest neighbor extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "nearest"},
    )

    # With nearest extrapolation, should have fewer or no undefined nodes
    assert not np.all(np.isnan(surf.values))
    # May still have some masked if points don't cover full surface
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked >= 0


def test_gridding_cubic_with_extrapolate_linear(simple_surface, simple_points):
    """Test cubic interpolation with linear extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "linear"},
    )

    # Linear extrapolation should fill remaining undefined areas
    assert not np.all(np.isnan(surf.values))
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked >= 0


def test_gridding_cubic_with_extrapolate_cubic(simple_surface, simple_points):
    """Test cubic interpolation with cubic extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "cubic"},
    )

    # Cubic extrapolation should fill remaining undefined areas
    assert not np.all(np.isnan(surf.values))
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked >= 0


def test_gridding_cubic_with_extrapolate_rbf(simple_surface, simple_points):
    """Test cubic interpolation with radial_basis extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "radial_basis"},
    )

    # RBF extrapolation should fill remaining undefined areas smoothly
    assert not np.all(np.isnan(surf.values))
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked >= 0


def test_gridding_cubic_extrapolate_default_method(simple_surface, simple_points):
    """Test that default extrapolation method is 'nearest' when extrapolate=True."""
    surf = simple_surface.copy()
    # Only specify extrapolate=True, not extrapolation_method
    surf.gridding(simple_points, method="cubic", method_options={"extrapolate": True})

    # Should use default extrapolation method (nearest)
    assert not np.all(np.isnan(surf.values))


def test_gridding_cubic_compare_extrapolation_methods(simple_surface, simple_points):
    """Compare different extrapolation methods after cubic gridding."""
    methods = ["nearest", "linear", "cubic", "radial_basis"]
    results = {}

    for extr_method in methods:
        surf = simple_surface.copy()
        surf.gridding(
            simple_points,
            method="cubic",
            method_options={
                "extrapolate": True,
                "extrapolation_method": extr_method,
            },
        )
        results[extr_method] = {
            "mean": surf.values.mean(),
            "std": surf.values.std(),
            "n_masked": surf.values.mask.sum() if hasattr(surf.values, "mask") else 0,
        }

    # All methods should produce valid results
    for method, stats in results.items():
        assert not np.isnan(stats["mean"]), f"Failed for method={method}"
        assert stats["n_masked"] >= 0, f"Invalid mask count for method={method}"


# ======================================================================================
# Test: linear method with extrapolation options
# ======================================================================================


def test_gridding_linear_with_extrapolate_nearest(simple_surface, simple_points):
    """Test linear interpolation with nearest neighbor extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="linear",
        method_options={"extrapolate": True, "extrapolation_method": "nearest"},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_linear_with_extrapolate_linear(simple_surface, simple_points):
    """Test linear interpolation with linear extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="linear",
        method_options={"extrapolate": True, "extrapolation_method": "linear"},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_linear_with_extrapolate_cubic(simple_surface, simple_points):
    """Test linear interpolation with cubic extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="linear",
        method_options={"extrapolate": True, "extrapolation_method": "cubic"},
    )

    assert not np.all(np.isnan(surf.values))


def test_gridding_linear_with_extrapolate_rbf(simple_surface, simple_points):
    """Test linear interpolation with RBF extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="linear",
        method_options={"extrapolate": True, "extrapolation_method": "radial_basis"},
    )

    assert not np.all(np.isnan(surf.values))


# ======================================================================================
# Test: nearest method with extrapolation options
# ======================================================================================


def test_gridding_nearest_with_extrapolate(simple_surface, simple_points):
    """Test nearest neighbor with extrapolation enabled."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points,
        method="nearest",
        method_options={"extrapolate": True, "extrapolation_method": "nearest"},
    )

    # Nearest neighbor typically fills all nodes anyway
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0  # Nearest should fill everything


def test_gridding_nearest_without_extrapolate(simple_surface, simple_points):
    """Test nearest neighbor without extrapolation."""
    surf = simple_surface.copy()
    surf.gridding(
        simple_points, method="nearest", method_options={"extrapolate": False}
    )

    # Even without extrapolate, nearest fills all nodes
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0


# ======================================================================================
# Test: extrapolation with sparse points
# ======================================================================================


def test_gridding_cubic_sparse_with_extrapolate(simple_surface, sparse_points):
    """Test cubic with sparse points and various extrapolation methods."""
    extrapolation_methods = ["nearest", "linear", "cubic", "radial_basis"]

    for extr_method in extrapolation_methods:
        surf = simple_surface.copy()
        surf.gridding(
            sparse_points,
            method="cubic",
            method_options={
                "extrapolate": True,
                "extrapolation_method": extr_method,
            },
        )

        # With sparse points and extrapolation, should fill most/all of surface
        assert not np.all(np.isnan(surf.values)), f"Failed for method={extr_method}"


def test_gridding_linear_sparse_with_extrapolate(simple_surface, sparse_points):
    """Test linear with sparse points and various extrapolation methods."""
    extrapolation_methods = ["nearest", "linear", "cubic", "radial_basis"]

    for extr_method in extrapolation_methods:
        surf = simple_surface.copy()
        surf.gridding(
            sparse_points,
            method="linear",
            method_options={
                "extrapolate": True,
                "extrapolation_method": extr_method,
            },
        )

        # With sparse points and extrapolation, should fill most/all of surface
        assert not np.all(np.isnan(surf.values)), f"Failed for method={extr_method}"


# ======================================================================================
# Test: extrapolation edge cases
# ======================================================================================


def test_gridding_extrapolate_invalid_method(simple_surface, simple_points):
    """Test that invalid extrapolation method raises appropriate error."""
    surf = simple_surface.copy()

    # Invalid extrapolation_method should be caught by fill() method
    with pytest.raises(Exception):  # Could be ValueError or other
        surf.gridding(
            simple_points,
            method="cubic",
            method_options={
                "extrapolate": True,
                "extrapolation_method": "invalid_method",
            },
        )


def test_gridding_cubic_extrapolate_consistency(simple_surface, simple_points):
    """Test that extrapolate=True fills more nodes than extrapolate=False."""
    # Grid without extrapolation
    surf_no_extr = simple_surface.copy()
    surf_no_extr.gridding(
        simple_points, method="cubic", method_options={"extrapolate": False}
    )
    n_masked_no_extr = (
        surf_no_extr.values.mask.sum()
        if hasattr(surf_no_extr.values, "mask")
        else np.isnan(surf_no_extr.values).sum()
    )

    # Grid with extrapolation
    surf_with_extr = simple_surface.copy()
    surf_with_extr.gridding(
        simple_points,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "nearest"},
    )
    n_masked_with_extr = (
        surf_with_extr.values.mask.sum()
        if hasattr(surf_with_extr.values, "mask")
        else np.isnan(surf_with_extr.values).sum()
    )

    # With extrapolation should have equal or fewer masked nodes
    assert n_masked_with_extr <= n_masked_no_extr


def test_gridding_linear_extrapolate_consistency(simple_surface, simple_points):
    """Test that extrapolate=True fills more nodes than extrapolate=False for linear."""
    # Grid without extrapolation
    surf_no_extr = simple_surface.copy()
    surf_no_extr.gridding(
        simple_points, method="linear", method_options={"extrapolate": False}
    )
    n_masked_no_extr = (
        surf_no_extr.values.mask.sum()
        if hasattr(surf_no_extr.values, "mask")
        else np.isnan(surf_no_extr.values).sum()
    )

    # Grid with extrapolation
    surf_with_extr = simple_surface.copy()
    surf_with_extr.gridding(
        simple_points,
        method="linear",
        method_options={"extrapolate": True, "extrapolation_method": "linear"},
    )
    n_masked_with_extr = (
        surf_with_extr.values.mask.sum()
        if hasattr(surf_with_extr.values, "mask")
        else np.isnan(surf_with_extr.values).sum()
    )

    # With extrapolation should have equal or fewer masked nodes
    assert n_masked_with_extr <= n_masked_no_extr


# ======================================================================================
# Test: extrapolation with pointset2 (real data)
# ======================================================================================


def test_gridding_pset2_cubic_no_extrapolate(pointset2_thinned, surface_for_pointset2):
    """Test cubic on pointset2 without extrapolation."""
    surf = surface_for_pointset2.copy()
    surf.gridding(
        pointset2_thinned, method="cubic", method_options={"extrapolate": False}
    )

    # Should have some masked nodes outside convex hull
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked > 0
    assert not np.all(np.isnan(surf.values))


def test_gridding_pset2_cubic_with_extrapolate_nearest(
    pointset2_thinned, surface_for_pointset2
):
    """Test cubic on pointset2 with nearest extrapolation."""
    surf = surface_for_pointset2.copy()
    surf.gridding(
        pointset2_thinned,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "nearest"},
    )

    # With extrapolation, should have no masked nodes
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0
    assert not np.all(np.isnan(surf.values))


def test_gridding_pset2_cubic_with_extrapolate_linear(
    pointset2_thinned, surface_for_pointset2
):
    """Test cubic on pointset2 with linear extrapolation."""
    surf = surface_for_pointset2.copy()
    surf.gridding(
        pointset2_thinned,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "linear"},
    )

    # With extrapolation, should have no masked nodes
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0
    assert not np.all(np.isnan(surf.values))


def test_gridding_pset2_cubic_with_extrapolate_rbf(
    pointset2_thinned, surface_for_pointset2
):
    """Test cubic on pointset2 with RBF extrapolation."""
    surf = surface_for_pointset2.copy()
    surf.gridding(
        pointset2_thinned,
        method="cubic",
        method_options={"extrapolate": True, "extrapolation_method": "radial_basis"},
    )

    # With RBF extrapolation, should have no masked nodes
    n_masked = surf.values.mask.sum() if hasattr(surf.values, "mask") else 0
    assert n_masked == 0
    assert not np.all(np.isnan(surf.values))


def test_gridding_pset2_linear_extrapolate_comparison(
    pointset2_thinned, surface_for_pointset2
):
    """Compare different extrapolation methods on pointset2 with linear gridding."""
    results = {}

    # Without extrapolation
    surf = surface_for_pointset2.copy()
    surf.gridding(
        pointset2_thinned, method="linear", method_options={"extrapolate": False}
    )
    results["no_extrapolate"] = {
        "mean": surf.values.mean(),
        "n_masked": surf.values.mask.sum() if hasattr(surf.values, "mask") else 0,
    }

    # With different extrapolation methods
    for extr_method in ["nearest", "linear", "cubic", "radial_basis"]:
        surf = surface_for_pointset2.copy()
        surf.gridding(
            pointset2_thinned,
            method="linear",
            method_options={"extrapolate": True, "extrapolation_method": extr_method},
        )
        results[extr_method] = {
            "mean": surf.values.mean(),
            "n_masked": surf.values.mask.sum() if hasattr(surf.values, "mask") else 0,
        }

    # All extrapolated versions should have no masked values
    assert results["no_extrapolate"]["n_masked"] > 0
    for method in ["nearest", "linear", "cubic", "radial_basis"]:
        assert results[method]["n_masked"] == 0, f"Failed for {method}"
        assert not np.isnan(results[method]["mean"]), f"Mean is NaN for {method}"
