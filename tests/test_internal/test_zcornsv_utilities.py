"""
Tests for zcornsv utility functions.

This module tests the zcornsv_pillar_to_cell and zcornsv_cell_to_pillar
utility functions that convert Z corner values between two different formats:

1. Pillar format: (ncol+1, nrow+1, nlay+1, 4) - Z values per pillar
2. Cell format: (ncol, nrow, nlay+1, 4) - Z values per cell

The mapping follows the convention:
- zcornsv_pillar(i, j, k, 3) (southwest corner) -> zcornsv_cell(i, j, k, 0)
- zcornsv_pillar(i+1, j, k, 2) (southeast corner) -> zcornsv_cell(i, j, k, 1)
- zcornsv_pillar(i, j+1, k, 1) (northwest corner) -> zcornsv_cell(i, j, k, 2)
- zcornsv_pillar(i+1, j+1, k, 0) (northeast corner) -> zcornsv_cell(i, j, k, 3)
"""

import numpy as np
import pytest
from xtgeo._internal import grid3d

from xtgeo.common.log import functimer


def test_zcornsv_pillar_to_cell_roundtrip():
    """Test roundtrip conversion between pillar and cell formats."""
    # Create a test grid
    ncol, nrow, nlay = 3, 2, 2

    # Create pillar format with random values
    np.random.seed(12345)
    zcornsv_pillar = (
        np.random.rand(ncol + 1, nrow + 1, nlay + 1, 4).astype(np.float32) * 100
    )

    # Test expected shapes
    assert zcornsv_pillar.shape == (ncol + 1, nrow + 1, nlay + 1, 4)

    # Convert from pillar to cell format
    zcornsv_cell = grid3d.zcornsv_pillar_to_cell(zcornsv_pillar)
    assert zcornsv_cell.shape == (ncol, nrow, nlay + 1, 4)

    # Convert back from cell to pillar format
    zcornsv_pillar_back = grid3d.zcornsv_cell_to_pillar(zcornsv_cell)
    assert zcornsv_pillar_back.shape == (ncol + 1, nrow + 1, nlay + 1, 4)

    # Test roundtrip accuracy: only interior pillar values should be perfectly preserved
    # Boundary values are filled with averages and won't match original random values

    # Test specific interior pillars that should be exactly preserved
    # For a 3x2x2 grid (ncol=3, nrow=2, nlay=2), interior pillars are (1,1)
    # in the i,j plane
    if ncol > 1 and nrow > 1:
        # Interior pillar (1,1) should be exactly preserved
        for k in range(nlay + 1):
            for corner in range(4):
                original_val = zcornsv_pillar[1, 1, k, corner]
                reconstructed_val = zcornsv_pillar_back[1, 1, k, corner]

                # These should be exactly equal (no averaging involved)
                assert abs(original_val - reconstructed_val) < 1e-6, (
                    f"Interior pillar ({1},{1},{k},{corner}) not preserved: "
                    f"original={original_val}, reconstructed={reconstructed_val}"
                )

    # Verify conversion preserves cell values exactly
    zcornsv_cell_check = grid3d.zcornsv_pillar_to_cell(zcornsv_pillar_back)
    np.testing.assert_array_almost_equal(
        zcornsv_cell,
        zcornsv_cell_check,
        decimal=6,
        err_msg="Cell values should be preserved through pillar roundtrip",
    )


def test_zcornsv_corner_mapping():
    """Test that specific corner mappings are correct."""
    # Create a small test case with known values
    ncol, nrow, nlay = 2, 2, 1

    # Create pillar format with specific test values
    zcornsv_pillar = np.zeros((ncol + 1, nrow + 1, nlay + 1, 4), dtype=np.float32)

    # Set specific values to test corner mapping
    # Cell (0,0) corners should come from:
    # SW: pillar (0,0) corner 3 -> cell (0,0) corner 0
    # SE: pillar (1,0) corner 2 -> cell (0,0) corner 1
    # NW: pillar (0,1) corner 1 -> cell (0,0) corner 2
    # NE: pillar (1,1) corner 0 -> cell (0,0) corner 3

    zcornsv_pillar[0, 0, 0, 3] = 100.0  # SW of cell (0,0)
    zcornsv_pillar[1, 0, 0, 2] = 110.0  # SE of cell (0,0)
    zcornsv_pillar[0, 1, 0, 1] = 120.0  # NW of cell (0,0)
    zcornsv_pillar[1, 1, 0, 0] = 130.0  # NE of cell (0,0)

    # Convert to cell format
    zcornsv_cell = grid3d.zcornsv_pillar_to_cell(zcornsv_pillar)

    # Verify the mapping for cell (0,0)
    assert zcornsv_cell[0, 0, 0, 0] == 100.0  # SW
    assert zcornsv_cell[0, 0, 0, 1] == 110.0  # SE
    assert zcornsv_cell[0, 0, 0, 2] == 120.0  # NW
    assert zcornsv_cell[0, 0, 0, 3] == 130.0  # NE


def test_boundary_filling():
    """Test boundary filling functionality."""
    # Create cell view with specific values
    zcornsv_cell = np.array(
        [
            [
                [[100, 110, 120, 130]],  # Cell (0,0), layer 0
                [[200, 210, 220, 230]],  # Cell (0,1), layer 0
            ],
            [
                [[300, 310, 320, 330]],  # Cell (1,0), layer 0
                [[400, 410, 420, 430]],  # Cell (1,1), layer 0
            ],
        ],
        dtype=np.float32,
    )

    # Test shapes
    assert zcornsv_cell.shape == (2, 2, 1, 4)

    # Test with boundary filling (default)
    pillar_view_filled = grid3d.zcornsv_cell_to_pillar(zcornsv_cell, fill_boundary=True)
    assert pillar_view_filled.shape == (3, 3, 1, 4)

    # Test without boundary filling
    pillar_view_no_fill = grid3d.zcornsv_cell_to_pillar(
        zcornsv_cell, fill_boundary=False
    )
    assert pillar_view_no_fill.shape == (3, 3, 1, 4)

    # Check that some boundary values are NaN when fill_boundary=False
    has_nan_no_fill = np.any(np.isnan(pillar_view_no_fill))
    assert has_nan_no_fill, "Expected NaN values with fill_boundary=False"

    # Check that no values are NaN when fill_boundary=True
    has_nan_filled = np.any(np.isnan(pillar_view_filled))
    assert not has_nan_filled, "No NaN values expected with fill_boundary=True"

    # Verify that interior pillars have the same values regardless of fill_boundary
    # Interior pillar (1,1) should be the same in both cases
    np.testing.assert_array_equal(
        pillar_view_filled[1, 1, 0, :],
        pillar_view_no_fill[1, 1, 0, :],
        "Interior pillars should be identical regardless of boundary filling",
    )


def test_realistic_use_case():
    """Test with realistic pillar data from a grid."""
    # Create some example pillar Z-values (normally you'd get this from a grid)
    zcornsv_pillar = np.array(
        [
            [
                [
                    [1000, 1010, 1020, 1030],  # Pillar (0,0), layer 0
                    [1001, 1011, 1021, 1031],
                ],  # Pillar (0,0), layer 1
                [
                    [1100, 1110, 1120, 1130],  # Pillar (0,1), layer 0
                    [1101, 1111, 1121, 1131],
                ],
            ],  # Pillar (0,1), layer 1
            [
                [
                    [2000, 2010, 2020, 2030],  # Pillar (1,0), layer 0
                    [2001, 2011, 2021, 2031],
                ],  # Pillar (1,0), layer 1
                [
                    [2100, 2110, 2120, 2130],  # Pillar (1,1), layer 0
                    [2101, 2111, 2121, 2131],
                ],
            ],  # Pillar (1,1), layer 1
            [
                [
                    [3000, 3010, 3020, 3030],  # Pillar (2,0), layer 0
                    [3001, 3011, 3021, 3031],
                ],  # Pillar (2,0), layer 1
                [
                    [3100, 3110, 3120, 3130],  # Pillar (2,1), layer 0
                    [3101, 3111, 3121, 3131],
                ],
            ],  # Pillar (2,1), layer 1
        ],
        dtype=np.float32,
    )

    # Verify initial shape
    assert zcornsv_pillar.shape == (3, 2, 2, 4)

    # Convert to cell-based format for easier processing
    zcornsv_cell = grid3d.zcornsv_pillar_to_cell(zcornsv_pillar)
    assert zcornsv_cell.shape == (2, 1, 2, 4)

    # Convert back to pillar format (boundary filling is default)
    zcornsv_pillar_result = grid3d.zcornsv_cell_to_pillar(zcornsv_cell)
    assert zcornsv_pillar_result.shape == (3, 2, 2, 4)

    # Verify that interior values are preserved exactly
    # Cell (0,0) corners should be preserved when converting back
    cell_00_corners = zcornsv_cell[0, 0, 0, :]

    # These should match the original pillar values for this cell
    expected_sw = zcornsv_pillar[0, 0, 0, 3]  # SW corner
    expected_se = zcornsv_pillar[1, 0, 0, 2]  # SE corner
    expected_nw = zcornsv_pillar[0, 1, 0, 1]  # NW corner
    expected_ne = zcornsv_pillar[1, 1, 0, 0]  # NE corner

    assert cell_00_corners[0] == expected_sw
    assert cell_00_corners[1] == expected_se
    assert cell_00_corners[2] == expected_nw
    assert cell_00_corners[3] == expected_ne


@pytest.mark.bigtest
def test_big_zcornsv_conversion():
    """Test with a large grid to evaluate performance by inspecting test logs."""
    # Create a large grid of Z-values, resembles a 300 million cell grid
    zcornsv_pillar = np.random.rand(1001, 1001, 301, 4).astype(np.float32)

    # Convert to cell-based format
    @functimer
    def pillar_to_cell(p):
        return grid3d.zcornsv_pillar_to_cell(p)

    @functimer
    def cell_to_pillar(zcornsv_cell):
        return grid3d.zcornsv_cell_to_pillar(zcornsv_cell)

    c = pillar_to_cell(zcornsv_pillar)
    p = cell_to_pillar(c)

    # Verify that the shapes are consistent
    assert zcornsv_pillar.shape == p.shape

    # Verify that interior values are preserved
    np.testing.assert_array_equal(
        zcornsv_pillar[1:99, 1:99, 1:9, :],
        p[1:99, 1:99, 1:9, :],
        "Interior pillars should be identical regardless of boundary filling",
    )
