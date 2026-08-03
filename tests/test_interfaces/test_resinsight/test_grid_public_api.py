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

import logging

import numpy as np
import pytest

import xtgeo

pytestmark = [
    pytest.mark.requires_resinsight,
    pytest.mark.xdist_group(name="resinsight"),
]  # Avoid running multiple tests in parallel against the same ResInsight instance


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


def test_grid_from_resinsight_case_object(resinsight_instance):
    """A rips case object can be passed directly instead of a case name.

    This disambiguates the two cases both named "EXAMPLE": here we explicitly
    pick the first one (the Drogon grid, 92 x 146 x 67).
    """
    cases = [c for c in resinsight_instance.project.cases() if c.name == "EXAMPLE"]
    assert len(cases) >= 2, "conftest should load two cases named EXAMPLE"

    grid = xtgeo.grid_from_resinsight(resinsight_instance, cases[0])
    assert isinstance(grid, xtgeo.Grid)
    assert grid.ncol == 92
    assert grid.nrow == 146
    assert grid.nlay == 67


# ---------------------------------------------------------------------------
# Grid.to_resinsight
# ---------------------------------------------------------------------------


def test_to_resinsight_creates_new_case(resinsight_instance):
    """to_resinsight should create a new case in ResInsight and return it."""
    grid = xtgeo.create_box_grid((3, 3, 2), increment=(10.0, 10.0, 5.0))
    case = grid.to_resinsight(resinsight_instance, case="GRID_NEW")

    assert case is not None, "to_resinsight should return the created case"
    assert case.name == "GRID_NEW"

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_NEW")
    assert reloaded.ncol == grid.ncol
    assert reloaded.nrow == grid.nrow
    assert reloaded.nlay == grid.nlay


def test_to_resinsight_replaces_existing_case(resinsight_instance):
    """to_resinsight should replace an existing case when called with the same name."""
    grid_a = xtgeo.create_box_grid((2, 2, 2))
    grid_a.to_resinsight(resinsight_instance, case="GRID_REPLACE")

    grid_b = xtgeo.create_box_grid((5, 4, 3))
    case = grid_b.to_resinsight(resinsight_instance, case="GRID_REPLACE")

    assert case is not None, "to_resinsight should return the replaced case"
    assert case.name == "GRID_REPLACE"

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_REPLACE")
    assert reloaded.ncol == grid_b.ncol
    assert reloaded.nrow == grid_b.nrow
    assert reloaded.nlay == grid_b.nlay


def test_to_resinsight_replaces_case_object(resinsight_instance):
    """to_resinsight should replace a case given directly as a rips case object."""
    grid_a = xtgeo.create_box_grid((2, 2, 2))
    case = grid_a.to_resinsight(resinsight_instance, case="GRID_OBJ_REPLACE")

    grid_b = xtgeo.create_box_grid((6, 5, 4))
    replaced = grid_b.to_resinsight(resinsight_instance, case=case)

    assert replaced is not None
    assert replaced.name == "GRID_OBJ_REPLACE"

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, replaced)
    assert reloaded.ncol == grid_b.ncol
    assert reloaded.nrow == grid_b.nrow
    assert reloaded.nlay == grid_b.nlay


# ---------------------------------------------------------------------------
# Full roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_from_resinsight_to_resinsight(resinsight_instance):
    """A grid read from ResInsight should survive a write → read roundtrip unchanged."""
    original = xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE")

    original.to_resinsight(resinsight_instance, case="GRID_ROUNDTRIP")

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

    original.to_resinsight(resinsight_instance, case="GRID_BOX_ROUNDTRIP")

    reloaded = xtgeo.grid_from_resinsight(resinsight_instance, "GRID_BOX_ROUNDTRIP")

    assert reloaded.ncol == original.ncol
    assert reloaded.nrow == original.nrow
    assert reloaded.nlay == original.nlay
    assert np.array_equal(reloaded.get_actnum().values, original.get_actnum().values), (
        "Active cell mask should be identical after roundtrip"
    )


# ---------------------------------------------------------------------------
# Deprecated argument aliases (backward compatibility)
# ---------------------------------------------------------------------------


def test_grid_from_resinsight_case_name_alias_deprecated(resinsight_instance):
    """The deprecated 'case_name' alias still works but emits a warning."""
    with pytest.warns(DeprecationWarning, match="case_name"):
        grid = xtgeo.grid_from_resinsight(resinsight_instance, case_name="EXAMPLE")
    assert isinstance(grid, xtgeo.Grid)

    # Passing both the new name and the deprecated alias is an error.
    with pytest.raises(TypeError, match="only 'case'"):
        xtgeo.grid_from_resinsight(resinsight_instance, "EXAMPLE", case_name="EXAMPLE")


def test_to_resinsight_gname_alias_deprecated(resinsight_instance):
    """The deprecated 'gname' alias for Grid.to_resinsight still works but warns."""
    grid = xtgeo.create_box_grid((2, 2, 2))
    with pytest.warns(DeprecationWarning, match="gname"):
        case = grid.to_resinsight(resinsight_instance, gname="GRID_GNAME_ALIAS")
    assert case.name == "GRID_GNAME_ALIAS"


# ---------------------------------------------------------------------------
# SUBGRIDS input property
# ---------------------------------------------------------------------------


def test_to_resinsight_exports_subgrids_property(resinsight_instance):
    """Grid.to_resinsight should write a SUBGRIDS input property when subgrids exist."""
    grid = xtgeo.create_box_grid((4, 3, 6))
    grid.set_subgrids({"Upper": 2, "Middle": 2, "Lower": 2})

    orig_propnames = list(grid.propnames or [])
    rips_case = grid.to_resinsight(resinsight_instance, case="GRID_SUBGRIDS_TEST")

    # Verify that grid props were restored to their original state (no side-effect).
    assert list(grid.propnames or []) == orig_propnames

    subgrids_prop = xtgeo.gridproperty_from_resinsight(
        resinsight_instance,
        rips_case,
        property_type="INPUT_PROPERTY",
        property_name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
    )
    assert subgrids_prop is not None
    assert subgrids_prop.isdiscrete

    # 1-based index; Upper=1, Middle=2, Lower=3
    vals = subgrids_prop.values.compressed()  # only unmasked (active) cells
    assert set(vals) == {1, 2, 3}
    assert subgrids_prop.codes == {1: "Upper", 2: "Middle", 3: "Lower"}


def test_to_resinsight_no_subgrids_no_property(resinsight_instance):
    """Grid.to_resinsight should not write SUBGRIDS when subgrids are not defined."""
    grid = xtgeo.create_box_grid((3, 3, 4))
    assert grid.subgrids is None

    rips_case = grid.to_resinsight(resinsight_instance, case="GRID_NO_SUBGRIDS_TEST")

    with pytest.raises(Exception):
        xtgeo.gridproperty_from_resinsight(
            resinsight_instance,
            rips_case,
            property_type="INPUT_PROPERTY",
            property_name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
        )


# ---------------------------------------------------------------------------
# SUBGRIDS round-trip import
# ---------------------------------------------------------------------------


def test_grid_from_resinsight_imports_subgrids(resinsight_instance):
    """grid_from_resinsight should populate grid.subgrids from a SUBGRIDS property."""
    original = xtgeo.create_box_grid((4, 3, 6))
    original.set_subgrids({"Top": 2, "Mid": 2, "Base": 2})

    rips_case = original.to_resinsight(
        resinsight_instance, case="GRID_SUBGRIDS_IMPORT_TEST"
    )

    imported = xtgeo.grid_from_resinsight(resinsight_instance, rips_case)

    assert imported.subgrids is not None, (
        "subgrids should be populated from SUBGRIDS property"
    )
    assert imported.get_subgrids() == {"Top": 2, "Mid": 2, "Base": 2}


def test_grid_from_resinsight_ignores_nondiscrete_subgrids_property(
    resinsight_instance,
):
    """A non-discrete SUBGRIDS property should be ignored on import."""
    grid = xtgeo.create_box_grid((4, 3, 6))
    rips_case = grid.to_resinsight(
        resinsight_instance, case="GRID_SUBGRIDS_IMPORT_NONDISCRETE_TEST"
    )

    nondiscrete_subgrids = xtgeo.GridProperty(
        grid,
        name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
        values=1.5,
        discrete=False,
    )
    nondiscrete_subgrids.to_resinsight(
        resinsight_instance,
        case=rips_case,
        property_name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
        property_type="INPUT_PROPERTY",
    )

    imported = xtgeo.grid_from_resinsight(resinsight_instance, rips_case)

    assert imported.subgrids is None


def test_grid_from_resinsight_subgrids_zero_value_excluded(resinsight_instance, caplog):
    """Value 0 in SUBGRIDS should be excluded and subgrid reconstruction skipped."""
    grid = xtgeo.create_box_grid((4, 3, 6))
    rips_case = grid.to_resinsight(
        resinsight_instance, case="GRID_SUBGRIDS_IMPORT_ZERO_EXCLUDED_TEST"
    )

    values = np.ones((grid.ncol, grid.nrow, grid.nlay), dtype=np.int32)
    values[:, :, :2] = 0
    subgrids_with_zero = xtgeo.GridProperty(
        grid,
        name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
        values=values,
        discrete=True,
        codes={0: "NoSubgrid", 1: "Main"},
    )
    subgrids_with_zero.to_resinsight(
        resinsight_instance,
        case=rips_case,
        property_name=xtgeo.interfaces.resinsight.SUBGRIDS_PROPERTY_NAME,
        property_type="INPUT_PROPERTY",
    )

    with caplog.at_level(logging.WARNING, logger="xtgeo.grid3d.grid"):
        imported = xtgeo.grid_from_resinsight(resinsight_instance, rips_case)

    assert imported.subgrids is None
    assert any(
        "Failed to reconstruct subgrids from SUBGRIDS property" in record.message
        and "grid.subgrids will not be set." in record.message
        for record in caplog.records
    )


def test_grid_from_resinsight_subgrids_inactive_boundary_layers_warns_and_skips(
    resinsight_instance, caplog
):
    """SUBGRIDS import should warn and skip when inactive layers break boundaries."""
    grid = xtgeo.create_box_grid((4, 3, 10))
    grid.set_subgrids({"Top": 4, "Middle": 3, "Bottom": 3})

    # Make full zone-boundary layers inactive (boundary between Top and Middle).
    grid._actnumsv[:, :, 3] = 0
    grid._actnumsv[:, :, 4] = 0

    rips_case = grid.to_resinsight(
        resinsight_instance, case="GRID_SUBGRIDS_IMPORT_INACTIVE_BOUNDARY_TEST"
    )

    with caplog.at_level(logging.WARNING, logger="xtgeo.grid3d.grid"):
        imported = xtgeo.grid_from_resinsight(resinsight_instance, rips_case)

    assert imported.subgrids is None
    assert any(
        "Failed to reconstruct subgrids from SUBGRIDS property" in record.message
        and "grid.subgrids will not be set." in record.message
        for record in caplog.records
    )


def test_grid_from_resinsight_no_subgrids_property(resinsight_instance):
    """grid_from_resinsight on a case without SUBGRIDS leaves grid.subgrids as None."""
    grid = xtgeo.create_box_grid((3, 3, 4))
    rips_case = grid.to_resinsight(
        resinsight_instance, case="GRID_NO_SUBGRIDS_IMPORT_TEST"
    )

    imported = xtgeo.grid_from_resinsight(resinsight_instance, rips_case)

    assert imported.subgrids is None
