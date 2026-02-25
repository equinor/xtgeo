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
    and centroid at (0.5, 0.5), which for a node-based 5x5 grid with
    xinc=yinc=1.0 places the center at (2.0, 2.0) in local coordinates.
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


def test_azimuth_is_global_independent_of_rotation():
    """Test that azimuth is interpreted in global coordinates.

    For azimuth=0 (global North), moving northward in global coordinates should
    increase depth, while moving eastward should keep approximately the same depth,
    regardless of map rotation.
    """
    for rotation in (0.0, 45.0, 90.0):
        surface = create_synthetic_surface(
            ncol=101,
            nrow=101,
            xinc=1.0,
            yinc=1.0,
            rotation=rotation,
            rx=0.0,
            ry=0.0,
            rz=0.0,
            dipping=10.0,
            top_depth=100.0,
            centroid=(0.5, 0.5),
            azimuth=0.0,
        )

        rot_rad = np.radians(rotation)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        dx = 0.5 * (surface.ncol - 1) * surface.xinc
        dy = 0.5 * (surface.nrow - 1) * surface.yinc
        x_center = surface.xori + dx * cos_r - dy * sin_r
        y_center = surface.yori + dx * sin_r + dy * cos_r

        center = surface.get_value_from_xy((x_center, y_center))
        north = surface.get_value_from_xy((x_center, y_center + 10.0))
        east = surface.get_value_from_xy((x_center + 10.0, y_center))

        assert north > center
        assert np.isclose(east, center, atol=1e-6)


def test_curvature_with_dipping_and_azimuth():
    """Test ellipsoidal curvature combined with dipping in azimuth direction.

    Creates a surface with both curvature (ellipsoid bowl) and dipping.
    Verifies that moving in the azimuth direction increases depth more than
    moving perpendicular to it.
    """
    # Create surface with curvature and dipping towards East (azimuth=90)
    surface = create_synthetic_surface(
        ncol=51,
        nrow=51,
        xinc=1.0,
        yinc=1.0,
        rotation=0.0,
        rx=50.0,
        ry=50.0,
        rz=10.0,
        dipping=5.0,
        top_depth=1000.0,
        centroid=(0.5, 0.5),
        azimuth=90.0,
    )

    # Get center point in world coordinates
    dx = 0.5 * (surface.ncol - 1) * surface.xinc
    dy = 0.5 * (surface.nrow - 1) * surface.yinc
    x_center = surface.xori + dx
    y_center = surface.yori + dy

    center = surface.get_value_from_xy((x_center, y_center))

    # Move 10 units East (azimuth direction): should increase depth significantly
    east = surface.get_value_from_xy((x_center + 10.0, y_center))
    east_delta = east - center

    # Move 10 units North (perpendicular): should increase depth only from curvature
    north = surface.get_value_from_xy((x_center, y_center + 10.0))
    north_delta = north - center

    # Dipping gradient (5°) should dominate in azimuth direction
    assert east_delta > north_delta, (
        f"Dipping in azimuth=90 (East) should increase depth more than North: "
        f"east_delta={east_delta:.4f}, north_delta={north_delta:.4f}"
    )
    # Sanity check: both should be positive (curvature and/or dipping)
    assert east_delta > 0.0
    assert north_delta > 0.0
