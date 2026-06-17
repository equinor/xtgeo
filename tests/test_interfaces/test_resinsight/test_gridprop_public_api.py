"""Tests for the public xtgeo grid property ↔ ResInsight API.

Covers:
- ``xtgeo.gridproperty_from_resinsight``
- ``xtgeo.GridProperty.to_resinsight``

These tests require a live ResInsight instance and the ``rips`` package.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo

pytestmark = pytest.mark.requires_resinsight


def _write_generated_property(resinsight_instance, case_name="EXAMPLE"):
    """Write a GENERATED continuous property to the case, return the created data."""
    from xtgeo.interfaces.resinsight._grid_property import (
        GridPropertyDataResInsight,
        GridPropertyWriter,
        _read_actnum,
    )
    from xtgeo.interfaces.resinsight._resinsight_base import _BaseResInsightDataRW

    base = _BaseResInsightDataRW(resinsight_instance)
    case = base.get_case(case_name=case_name, find_last=True)
    dims = case.grids()[0].dimensions()
    nx, ny, nz = dims.i, dims.j, dims.k
    total = nx * ny * nz
    actnum = _read_actnum(case, total)

    rng = np.random.default_rng(42)
    values = rng.random(total).astype(np.float64) * 0.4

    data = GridPropertyDataResInsight(
        name="SYNTH_PORO",
        nx=nx,
        ny=ny,
        nz=nz,
        values=values,
        actnumsv=actnum,
        property_type="GENERATED",
        time_step_index=0,
        discrete=False,
        codes={},
        filesrc="",
    )
    GridPropertyWriter(resinsight_instance).save(data, case_name)
    return data


# ---------------------------------------------------------------------------
# gridproperty_from_resinsight
# ---------------------------------------------------------------------------


def test_gridproperty_from_resinsight_returns_valid_gridproperty(resinsight_instance):
    """gridproperty_from_resinsight should return an xtgeo.GridProperty instance."""
    data = _write_generated_property(resinsight_instance)
    prop = xtgeo.gridproperty_from_resinsight(
        resinsight_instance, "EXAMPLE", "SYNTH_PORO", property_type="GENERATED"
    )

    assert isinstance(prop, xtgeo.GridProperty)
    assert not prop.isdiscrete
    assert prop.name == "SYNTH_PORO"
    assert prop.ncol == data.nx and prop.nrow == data.ny and prop.nlay == data.nz
    active_vals = prop.values.compressed()
    assert active_vals.min() >= 0.0
    assert active_vals.max() <= 0.4


def test_gridproperty_from_resinsight_no_case_raises(resinsight_instance):
    """Should raise RuntimeError for an unknown case name."""
    with pytest.raises(RuntimeError, match="Cannot find any case with name"):
        xtgeo.gridproperty_from_resinsight(
            resinsight_instance, "NON_EXISTENT_CASE", "ANY"
        )


# ---------------------------------------------------------------------------
# GridProperty.to_resinsight
# ---------------------------------------------------------------------------


def test_to_resinsight_continuous(resinsight_instance):
    """A continuous property should survive a write → read roundtrip."""
    _write_generated_property(resinsight_instance)
    original = xtgeo.gridproperty_from_resinsight(
        resinsight_instance, "EXAMPLE", "SYNTH_PORO", property_type="GENERATED"
    )

    original.to_resinsight(
        resinsight_instance,
        case_name="EXAMPLE",
        property_name="SYNTH_PORO_COPY",
        property_type="GENERATED",
    )

    reloaded = xtgeo.gridproperty_from_resinsight(
        resinsight_instance,
        "EXAMPLE",
        "SYNTH_PORO_COPY",
        property_type="GENERATED",
    )

    assert_allclose(
        original.values.compressed(), reloaded.values.compressed(), atol=1e-6
    )


def test_to_resinsight_discrete(resinsight_instance):
    """A discrete property should roundtrip correctly."""
    _write_generated_property(resinsight_instance)
    base_prop = xtgeo.gridproperty_from_resinsight(
        resinsight_instance, "EXAMPLE", "SYNTH_PORO", property_type="GENERATED"
    )

    # Create a discrete property from the continuous values
    active_vals = base_prop.values.compressed()
    region_vals = np.asarray([int(v * 100) % 4 for v in active_vals], dtype=np.int32)

    total = base_prop.ncol * base_prop.nrow * base_prop.nlay
    full_vals = np.zeros(total, dtype=np.int32)
    active_mask = ~np.ma.getmaskarray(base_prop.values).ravel()
    full_vals[active_mask] = region_vals

    masked = np.ma.MaskedArray(
        full_vals.reshape(base_prop.ncol, base_prop.nrow, base_prop.nlay),
        mask=np.ma.getmaskarray(base_prop.values),
    )
    region_prop = xtgeo.GridProperty(
        ncol=base_prop.ncol,
        nrow=base_prop.nrow,
        nlay=base_prop.nlay,
        values=masked,
        name="REGION_TEST",
        discrete=True,
        codes={0: "Sand", 1: "Shale", 2: "Coal", 3: "Limestone"},
    )

    region_prop.to_resinsight(
        resinsight_instance,
        case_name="EXAMPLE",
        property_type="GENERATED",
    )

    reloaded = xtgeo.gridproperty_from_resinsight(
        resinsight_instance,
        "EXAMPLE",
        "REGION_TEST",
        property_type="GENERATED",
    )

    assert np.array_equal(region_prop.values.compressed(), reloaded.values.compressed())


# ---------------------------------------------------------------------------
# Full roundtrip: box grid property
# ---------------------------------------------------------------------------


def test_roundtrip_box_grid_property(resinsight_instance):
    """A synthetic box grid + property should roundtrip through ResInsight."""
    grid = xtgeo.create_box_grid((4, 3, 2), increment=(5.0, 5.0, 2.0))
    grid.to_resinsight(resinsight_instance, gname="PROP_TEST_GRID")

    prop = xtgeo.GridProperty(grid, name="SYNTH", values=0.42, discrete=False)
    prop.to_resinsight(
        resinsight_instance,
        case_name="PROP_TEST_GRID",
        property_name="SYNTH",
        property_type="GENERATED",
    )

    reloaded = xtgeo.gridproperty_from_resinsight(
        resinsight_instance,
        "PROP_TEST_GRID",
        "SYNTH",
        property_type="GENERATED",
    )
    assert reloaded.ncol == 4
    assert reloaded.nrow == 3
    assert reloaded.nlay == 2
    assert_allclose(reloaded.values.compressed(), 0.42, atol=1e-6)
