"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Fccus here on hexahedron functions
"""

import pytest
from hypothesis import given, strategies as st

import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.xyz import Point  # type: ignore
from xtgeo.common.log import null_logger

logger = null_logger(__name__)


@pytest.fixture
def hcorners_normal():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 1.0),
        Point(100.0, 0.0, 1.0),
        Point(100.0, 100.0, 1.0),
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_concave():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 1.0),
        Point(100.0, 0.0, 1.0),
        Point(10.0, 10.0, 1.0),  # Concave point
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(10.0, 10.0, 0.0),  # Concave point
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_concave_top():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 2.0),
        Point(100.0, 0.0, 1.0),
        Point(100.0, 100.0, 2.0),
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_very_concave_top():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 90.0),
        Point(100.0, 0.0, 1.0),
        Point(100.0, 100.0, 90.0),
        Point(0.0, 100.0, 1.0),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


@pytest.fixture
def hcorners_thin():
    return _internal.geometry.HexahedronCorners(
        Point(0.0, 0.0, 0.001),
        Point(100.0, 0.0, 0.001),
        Point(100.0, 100.0, 0.001),
        Point(0.0, 100.0, 0.001),
        Point(0.0, 0.0, 0.0),
        Point(100.0, 0.0, 0.0),
        Point(100.0, 100.0, 0.0),
        Point(0.0, 100.0, 0.0),
    )


THIN_RATIO = 1e-6  # this is typical a cell thickness 0.01 m and a cell size of 100 m


def test_is_hexahedron_thin(hcorners_normal, hcorners_thin):
    """Test the is_hexahedron_thin function."""

    # Create a regular hexahedron (not thin)
    assert not _internal.geometry.is_hexahedron_thin(hcorners_normal, THIN_RATIO)

    assert _internal.geometry.is_hexahedron_thin(hcorners_thin, THIN_RATIO)


def test_is_hexahedron_concave_projected(
    hcorners_normal, hcorners_concave, hcorners_concave_top
):
    """Test the is_hexahedron_concave_projected function."""
    # Create a regular convex hexahedron
    assert not _internal.geometry.is_hexahedron_concave_projected(hcorners_normal)
    assert not _internal.geometry.is_hexahedron_concave_projected(hcorners_concave_top)

    assert _internal.geometry.is_hexahedron_concave_projected(hcorners_concave)


def test_is_hexahedron_non_convert_distorted(
    hcorners_normal,
    hcorners_concave,
    hcorners_thin,
    hcorners_concave_top,
    hcorners_very_concave_top,
):
    """Test methods that determine if a cell is non_convext, etc"""
    assert _internal.geometry.is_hexahedron_severely_distorted(hcorners_thin)
    assert not _internal.geometry.is_hexahedron_severely_distorted(hcorners_normal)
    assert _internal.geometry.is_hexahedron_severely_distorted(hcorners_concave)

    assert not _internal.geometry.is_hexahedron_non_convex(hcorners_thin)
    assert not _internal.geometry.is_hexahedron_non_convex(hcorners_normal)
    assert _internal.geometry.is_hexahedron_non_convex(hcorners_concave)

    # hcorners with concave top but not severely distorted
    assert _internal.geometry.is_hexahedron_non_convex(hcorners_concave_top)
    assert not _internal.geometry.is_hexahedron_severely_distorted(hcorners_concave_top)

    # hcorners with very concave top will be severy distorted also
    assert _internal.geometry.is_hexahedron_non_convex(hcorners_very_concave_top)
    assert _internal.geometry.is_hexahedron_severely_distorted(
        hcorners_very_concave_top
    )


@given(
    base_x=st.floats(0, 50),
    base_y=st.floats(0, 50),
    height=st.floats(1, 10),
    width=st.floats(10, 100),
    depth=st.floats(10, 100),
)
def test_hexahedron_volume_with_randomized_corners(
    base_x, base_y, height, width, depth
):
    """Test the hexahedron volume function with randomized but logical corners."""
    # Define corners based on the base position, width, depth, and height
    corners = [
        (base_x, base_y, 0.0),  # Lower SW
        (base_x + width, base_y, 0.0),  # Lower SE
        (base_x, base_y + depth, 0.0),  # Lower NW
        (base_x + width, base_y + depth, 0.0),  # Lower NE
        (base_x, base_y, height),  # Upper SW
        (base_x + width, base_y, height),  # Upper SE
        (base_x, base_y + depth, height),  # Upper NW
        (base_x + width, base_y + depth, height),  # Upper NE
    ]

    # Convert the list of tuples to CellCorners
    cell_corners = _internal.grid3d.CellCorners(*[Point(*corner) for corner in corners])

    # Calculate the volume
    volume = _internal.geometry.hexahedron_volume(
        cell_corners, _internal.geometry.HexVolumePrecision.P4
    )

    # Assert that the volume is non-negative
    assert volume >= 0, f"Volume should be non-negative, got {volume}"
