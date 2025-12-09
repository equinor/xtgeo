"""Tests for BlockedWellData dataclass."""

from __future__ import annotations

import numpy as np
import pytest

from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellLog


def test_blockedwell_creation():
    """Test creating a BlockedWellData object."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, 21.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    well = BlockedWellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    assert well.name == "TestWell"
    assert well.n_records == n
    assert len(well.i_index) == n
    assert len(well.j_index) == n
    assert len(well.k_index) == n
    np.testing.assert_array_equal(well.i_index, i_index)
    np.testing.assert_array_equal(well.j_index, j_index)
    np.testing.assert_array_equal(well.k_index, k_index)


def test_blockedwell_with_logs():
    """Test BlockedWellData with well logs."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, 21.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    gr_log = WellLog(name="GR", values=np.random.rand(n))
    phit_log = WellLog(name="PHIT", values=np.random.rand(n))

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log),
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    assert len(well.logs) == 2
    assert well.log_names == ("GR", "PHIT")


def test_blockedwell_missing_indices():
    """Test that BlockedWellData requires all index arrays."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, 21.0, 21.0, 22.0])

    # Missing k_index - should raise TypeError at construction
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        BlockedWellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            i_index=i_index,
            j_index=j_index,
        )


def test_blockedwell_index_length_mismatch():
    """Test that BlockedWellData validates index array lengths."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0])  # Wrong length
    j_index = np.array([20.0, 20.0, 21.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError, match="i_index has 3 values, but survey has 5"):
        BlockedWellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            i_index=i_index,
            j_index=j_index,
            k_index=k_index,
        )


def test_blockedwell_j_index_length_mismatch():
    """Ensure mismatched j_index length raises ValueError."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, 21.0])  # Wrong length
    k_index = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(ValueError, match="j_index has 3 values, but survey has 5"):
        BlockedWellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            i_index=i_index,
            j_index=j_index,
            k_index=k_index,
        )


def test_blockedwell_k_index_length_mismatch():
    """Ensure mismatched k_index length raises ValueError."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 10.0, 11.0, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, 21.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0])  # Wrong length

    with pytest.raises(ValueError, match="k_index has 3 values, but survey has 5"):
        BlockedWellData(
            name="TestWell",
            xpos=0.0,
            ypos=0.0,
            zpos=0.0,
            survey_x=survey_x,
            survey_y=survey_y,
            survey_z=survey_z,
            i_index=i_index,
            j_index=j_index,
            k_index=k_index,
        )


def test_blockedwell_with_nan_indices():
    """Test BlockedWellData with undefined (NaN) grid indices."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    # Some points are outside the grid
    i_index = np.array([10.0, 10.0, np.nan, 11.0, 12.0])
    j_index = np.array([20.0, 20.0, np.nan, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    assert well.n_records == n
    assert well.n_blocked_cells == 4  # One point has NaN indices
    assert well.has_valid_indices


def test_blockedwell_all_nan_indices():
    """Test BlockedWellData with all NaN indices."""
    n = 3
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    # All points are outside the grid
    i_index = np.array([np.nan, np.nan, np.nan])
    j_index = np.array([np.nan, np.nan, np.nan])
    k_index = np.array([np.nan, np.nan, np.nan])

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    assert well.n_blocked_cells == 0
    assert not well.has_valid_indices


def test_blockedwell_get_cell_indices():
    """Test getting cell indices at specific survey points."""
    n = 5
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    j_index = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
    k_index = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    # Get indices at point 0
    i, j, k = well.get_cell_indices(0)
    assert i == 10.0
    assert j == 20.0
    assert k == 1.0

    # Get indices at point 2
    i, j, k = well.get_cell_indices(2)
    assert i == 12.0
    assert j == 22.0
    assert k == 3.0


def test_blockedwell_get_cell_indices_out_of_bounds():
    """Test that get_cell_indices raises IndexError for invalid indices."""
    n = 3
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 11.0, 12.0])
    j_index = np.array([20.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0])

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    # Test negative index
    with pytest.raises(IndexError, match="Index -1 out of bounds"):
        well.get_cell_indices(-1)

    # Test index too large
    with pytest.raises(IndexError, match="Index 10 out of bounds"):
        well.get_cell_indices(10)


def test_blockedwell_inheritance():
    """Test that BlockedWellData properly inherits from WellData."""
    n = 3
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 11.0, 12.0])
    j_index = np.array([20.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0])

    gr_log = WellLog(name="GR", values=np.random.rand(n))

    well = BlockedWellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log,),
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    # Test inherited properties
    assert well.n_records == n
    assert well.log_names == ("GR",)
    assert well.get_log("GR") is not None
    assert well.get_continuous_logs() == (gr_log,)


def test_blockedwell_immutability():
    """Test that BlockedWellData is immutable."""
    n = 3
    survey_x = np.linspace(0, 100, n)
    survey_y = np.linspace(0, 200, n)
    survey_z = np.linspace(-50, -150, n)

    i_index = np.array([10.0, 11.0, 12.0])
    j_index = np.array([20.0, 21.0, 22.0])
    k_index = np.array([1.0, 2.0, 3.0])

    well = BlockedWellData(
        name="TestWell",
        xpos=0.0,
        ypos=0.0,
        zpos=0.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    # Try to modify - should fail
    with pytest.raises(Exception):  # FrozenInstanceError
        well.name = "NewName"

    with pytest.raises(Exception):
        well.i_index = np.array([1.0, 2.0, 3.0])
