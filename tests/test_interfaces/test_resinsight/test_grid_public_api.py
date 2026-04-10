"""Tests for the public xtgeo grid ↔ ResInsight API.

Covers:
- ``xtgeo.grid_from_resinsight``
- ``xtgeo.Grid.to_resinsight``

These tests require a live ResInsight instance and the ``rips`` package.
They are run via the shared ``resinsight_instance`` fixture defined in
``conftest.py``, which loads two cases both named "EXAMPLE" so that the
``find_last`` behaviour can be exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

import xtgeo

pytestmark = pytest.mark.requires_resinsight


# ---------------------------------------------------------------------------
# grid_from_resinsight
# ---------------------------------------------------------------------------


def test_grid_from_resinsight_returns_grid(resinsight_instance):
    """grid_from_resinsight should return an xtgeo.Grid instance."""
    grid = xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE")
    assert isinstance(grid, xtgeo.Grid)


def test_grid_from_resinsight_find_last_true(resinsight_instance):
    """With find_last=True (default) the last loaded case is selected.

    The conftest loads DROGON first and EMERALD second, both renamed "EXAMPLE",
    so the last match is the Emerald grid (4 x 4 x 3).
    """
    grid = xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE", find_last=True)
    assert grid.ncol == 4
    assert grid.nrow == 4
    assert grid.nlay == 3


def test_grid_from_resinsight_find_last_false(resinsight_instance):
    """With find_last=False the first loaded case is selected.

    The first match is the Drogon grid (92 x 146 x 67).
    """
    grid = xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE", find_last=False)
    assert grid.ncol == 92
    assert grid.nrow == 146
    assert grid.nlay == 67


def test_grid_from_resinsight_no_matching_case_raises(resinsight_instance):
    """grid_from_resinsight should raise RuntimeError for an unknown case name."""
    with pytest.raises(RuntimeError, match="Cannot find any case with name"):
        xtgeo.grid_from_resinsight(resinsight_instance, "NON_EXISTENT_CASE")


def test_grid_from_resinsight_auto_discover(resinsight_instance):
    """Passing None lets ResInsight auto-discover the running instance."""
    grid = xtgeo.grid_from_resinsight(None, "EXAMPLE")
    assert isinstance(grid, xtgeo.Grid)
    assert grid.ncol == 4
    assert grid.nrow == 4
    assert grid.nlay == 3


# ---------------------------------------------------------------------------
# Grid.to_resinsight
# ---------------------------------------------------------------------------


def test_to_resinsight_creates_new_case(resinsight_instance):
    """to_resinsight should create a new case in ResInsight."""
    grid = xtgeo.create_box_grid((3, 3, 2), increment=(10.0, 10.0, 5.0))
    grid.to_resinsight(resinsight_instance, gname="GRID_NEW")

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_NEW")
    assert reloaded.ncol == grid.ncol
    assert reloaded.nrow == grid.nrow
    assert reloaded.nlay == grid.nlay


def test_to_resinsight_replaces_existing_case(resinsight_instance):
    """to_resinsight should replace an existing case when called with the same name."""
    grid_a = xtgeo.create_box_grid((2, 2, 2))
    grid_a.to_resinsight(resinsight_instance, gname="GRID_REPLACE")

    grid_b = xtgeo.create_box_grid((5, 4, 3))
    grid_b.to_resinsight(resinsight_instance, gname="GRID_REPLACE")

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_REPLACE")
    assert reloaded.ncol == grid_b.ncol
    assert reloaded.nrow == grid_b.nrow
    assert reloaded.nlay == grid_b.nlay


# ---------------------------------------------------------------------------
# Full roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_from_resinsight_to_resinsight(resinsight_instance):
    """A grid read from ResInsight should survive a write → read roundtrip unchanged."""
    original = xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE")

    original.to_resinsight(resinsight_instance, gname="GRID_ROUNDTRIP")

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_ROUNDTRIP")

    assert reloaded.ncol == original.ncol
    assert reloaded.nrow == original.nrow
    assert reloaded.nlay == original.nlay
    assert np.array_equal(reloaded.get_actnum().values, original.get_actnum().values), (
        "Active cell mask should be identical after roundtrip"
    )


def test_roundtrip_box_grid(resinsight_instance):
    """A synthetic box grid should round-trip through ResInsight without loss."""
    original = xtgeo.create_box_grid((4, 3, 2), increment=(5.0, 5.0, 2.0))

    original.to_resinsight(resinsight_instance, gname="GRID_BOX_ROUNDTRIP")

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_BOX_ROUNDTRIP")

    assert reloaded.ncol == original.ncol
    assert reloaded.nrow == original.nrow
    assert reloaded.nlay == original.nlay
    assert np.array_equal(reloaded.get_actnum().values, original.get_actnum().values), (
        "Active cell mask should be identical after roundtrip"
    )
