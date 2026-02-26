"""Tests for create_synthetic_surface."""

import numpy as np

from xtgeo import create_synthetic_surface


def test_create_synthetic_surface_shape_and_min():
    surface = create_synthetic_surface(
        ncol=4,
        nrow=3,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        dipping=5.0,
        top_depth=0.0,
        centroid=(0.0, 0.0),
        azimuth=0.0,
    )

    assert surface.ncol == 4
    assert surface.nrow == 3
    assert surface.values.shape == (4, 3)
    assert np.isclose(surface.values.min(), 0.0)


def test_create_synthetic_surface_with_rotation():
    surface = create_synthetic_surface(
        ncol=5,
        nrow=5,
        xinc=2.0,
        yinc=2.0,
        rotation=45.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        dipping=0.0,
        top_depth=10.0,
        centroid=(0.0, 0.0),
        azimuth=0.0,
    )
    assert surface.ncol == 5
    assert surface.nrow == 5
    assert surface.values.shape == (5, 5)
    assert np.isclose(surface.values.min(), 10.0)
    # Check that all values are >= 10.0 (since top_depth=10.0, dipping=0)
    assert np.all(surface.values >= 10.0)
    # For zero dipping, all values should be equal to top_depth
    assert np.allclose(surface.values, 10.0)


def test_create_synthetic_surface_with_dipping_and_azimuth():
    """Test with dipping=10 degrees towards East (azimuth=90).

    With azimuth=90 (East), centroid=(0,0), the surface dips in the +X direction.
    At X=0, Z=100 (top_depth). Each unit in X increases Z by tan(10°) ≈ 0.17633.
    """
    surface = create_synthetic_surface(
        ncol=3,
        nrow=3,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        dipping=10.0,
        top_depth=100.0,
        centroid=(0.0, 0.0),
        azimuth=90.0,
    )

    # Expected values: dips in +X direction with tan(10°) ≈ 0.17633
    # X coordinates: 0, 1, 2 (relative to origin at centroid)
    # Z increases by ~0.17633 per unit X
    expected = np.array(
        [
            [100.0, 100.0, 100.0],  # X=0, all Y values
            [100.17633, 100.17633, 100.17633],  # X=1, all Y values
            [100.35266, 100.35266, 100.35266],  # X=2, all Y values
        ]
    )

    assert surface.values.shape == (3, 3)
    assert np.allclose(surface.values, expected, atol=1e-4)


def test_create_synthetic_surface_with_curvature_and_centroid():
    """Test ellipsoidal curvature with non-default centroid.

    Creates a 5x5 surface with ellipsoidal curvature (rx=10, ry=10, rz=5)
    and centroid at (0.5, 0.5), which places the center at (2.5, 2.5).
    The surface forms a downward-facing bowl (ellipsoid cap).
    """
    surface = create_synthetic_surface(
        ncol=5,
        nrow=5,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        rx=10.0,
        ry=10.0,
        rz=5.0,
        dipping=0.0,
        top_depth=1000.0,
        centroid=(0.5, 0.5),
        azimuth=0.0,
    )

    assert surface.values.shape == (5, 5)
    assert np.isclose(surface.values.min(), 1000.0)

    # Center should have the minimum value (top point on the ellipsoid bowl)
    center_val = surface.values[2, 2]
    assert center_val == np.min(surface.values)

    # Values should increase (get deeper) as we move away from center
    assert surface.values[0, 2] > center_val  # Edge is deeper than center
    assert surface.values[2, 0] > center_val
    assert surface.values[1, 2] > center_val

    # Corners should be even higher
    assert surface.values[0, 0] > surface.values[1, 1]


def test_create_synthetic_surface_with_curvature_and_offset_centroid():
    """Test with curvature and centroid at corner (0, 0).

    With centroid at (0,0), the origin is at the grid corner, which is
    the top point of the ellipsoid bowl. Z increases away from origin.
    """
    surface = create_synthetic_surface(
        ncol=3,
        nrow=3,
        xinc=2.0,
        yinc=2.0,
        rotation=0.0,
        rx=8.0,
        ry=8.0,
        rz=4.0,
        dipping=0.0,
        top_depth=500.0,
        centroid=(0.0, 0.0),
        azimuth=0.0,
    )

    assert surface.values.shape == (3, 3)
    assert np.isclose(surface.values.min(), 500.0)

    # With centroid at (0,0), corner (0,0) should have the lowest value
    assert surface.values[0, 0] == np.min(surface.values)
    assert np.isclose(surface.values[0, 0], 500.0)

    # Values should increase as we move away from origin
    assert surface.values[0, 0] < surface.values[1, 1]
    assert surface.values[1, 1] < surface.values[2, 2]

    # Far corner should be the deepest
    assert surface.values[2, 2] == np.max(surface.values)


def test_rotation_affects_azimuth_dipping():
    """Test that rotation correctly adjusts azimuth direction.

    With rotation, the azimuth should be measured in the rotated coordinate system.
    Without rotation, azimuth=0 means dip towards North (+Y).
    With 90° rotation, azimuth=0 should mean dip towards East (+X) in the global frame,
    because in the rotated frame, the original Y direction is now along the X axis.
    """
    # Create surface without rotation: azimuth=0 dips towards +Y
    surf_no_rot = create_synthetic_surface(
        ncol=5,
        nrow=5,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        dipping=10.0,
        top_depth=100.0,
        centroid=(0.0, 0.0),
        azimuth=0.0,
    )

    # Create surface with 90° rotation: azimuth=0 should dip towards +X
    surf_rot90 = create_synthetic_surface(
        ncol=5,
        nrow=5,
        xinc=1.0,
        yinc=1.0,
        rotation=90.0,
        rx=0.0,
        ry=0.0,
        rz=0.0,
        dipping=10.0,
        top_depth=100.0,
        centroid=(0.0, 0.0),
        azimuth=0.0,
    )

    # In the unrotated surface, values increase along Y direction (row index)
    # Compare corner (0,0) vs (0,4): values should increase along Y
    no_rot_y_increase = surf_no_rot.values[0, 4] > surf_no_rot.values[0, 0]
    assert no_rot_y_increase, "Without rotation, dip should increase along Y"

    # In the rotated surface, values should increase along X direction (col index)
    # Compare corner (0,0) vs (4,0): values should increase along X
    rot_x_increase = surf_rot90.values[4, 0] > surf_rot90.values[0, 0]
    assert rot_x_increase, "With 90° rotation, dip should increase along X"

    # The dipping gradient should be approximately equal
    # For unrotated: gradient along Y axis
    y_gradient = (surf_no_rot.values[0, 4] - surf_no_rot.values[0, 0]) / 4.0
    # For rotated: gradient along X axis
    x_gradient = (surf_rot90.values[4, 0] - surf_rot90.values[0, 0]) / 4.0

    assert np.isclose(y_gradient, x_gradient, rtol=1e-4), (
        f"Dipping gradients should be equal: y_gradient={y_gradient}, "
        f"x_gradient={x_gradient}"
    )
