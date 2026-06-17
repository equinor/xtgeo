"""Unit tests for GridPropertyDataResInsight.

Tests the data container's validation, XTGeo ↔ ResInsight roundtrip logic,
and mask/actnum handling.  Mirrors the patterns in test_resinsight_grid.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.common.constants import UNDEF, UNDEF_INT
from xtgeo.interfaces.resinsight._grid_property import (
    GridPropertyDataResInsight,
    GridPropertyWriter,
    _read_actnum,
    _validate_property_type,
)

pytestmark = pytest.mark.requires_resinsight


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_array_size():
    """__post_init__ rejects arrays whose sizes don't match nx*ny*nz."""
    correct = 24  # 4*3*2

    with pytest.raises(ValueError, match="values should have length 24"):
        GridPropertyDataResInsight(
            name="T",
            nx=4,
            ny=3,
            nz=2,
            values=np.zeros(10, dtype=np.float64),
            actnumsv=np.ones(correct, dtype=np.int32),
            property_type="STATIC_NATIVE",
            time_step_index=0,
            discrete=False,
            codes={},
            filesrc="",
        )

    with pytest.raises(ValueError, match="actnumsv should have length 24"):
        GridPropertyDataResInsight(
            name="T",
            nx=4,
            ny=3,
            nz=2,
            values=np.zeros(correct, dtype=np.float64),
            actnumsv=np.ones(10, dtype=np.int32),
            property_type="STATIC_NATIVE",
            time_step_index=0,
            discrete=False,
            codes={},
            filesrc="",
        )


def test_validate_property_type_invalid_string():
    """_validate_property_type raises ValueError on unknown string."""
    with pytest.raises(ValueError, match="Invalid property_type"):
        _validate_property_type("NOT_A_REAL_TYPE")


# ---------------------------------------------------------------------------
# Roundtrip: XTGeo GridProperty ↔ GridPropertyDataResInsight
# ---------------------------------------------------------------------------


def test_roundtrip_continuous():
    """Continuous property values survive from_xtgeo → to_xtgeo roundtrip."""
    rng = np.random.default_rng(42)
    vals = np.ma.array(rng.random((4, 3, 2)))

    original = xtgeo.GridProperty(ncol=4, nrow=3, nlay=2, values=vals, name="PORO")
    data = GridPropertyDataResInsight.from_xtgeo_gridproperty(
        original, property_type="STATIC_NATIVE"
    )
    restored = data.to_xtgeo_gridproperty()

    assert restored.ncol == 4 and restored.nrow == 3 and restored.nlay == 2
    assert restored.name == "PORO"
    assert not restored.isdiscrete
    assert_allclose(restored.values.compressed(), original.values.compressed())


def test_roundtrip_discrete_with_codes():
    """Discrete property with code labels survives roundtrip."""
    vals = np.ma.array(
        np.array([[[1, 2], [0, 1], [2, 0]], [[0, 2], [1, 0], [2, 1]]], dtype=np.int32),
    )
    codes = {0: "Sand", 1: "Shale", 2: "Coal"}
    original = xtgeo.GridProperty(
        ncol=2,
        nrow=3,
        nlay=2,
        values=vals,
        name="FACIES",
        discrete=True,
        codes=codes,
    )
    data = GridPropertyDataResInsight.from_xtgeo_gridproperty(
        original, property_type="STATIC_NATIVE"
    )
    restored = data.to_xtgeo_gridproperty()

    assert restored.isdiscrete
    assert np.array_equal(original.values, restored.values)
    assert restored.codes == codes


def test_roundtrip_preserves_inactive_cells():
    """Masked (inactive) cells survive the roundtrip with correct fill values."""
    vals = np.ma.MaskedArray(
        np.arange(24, dtype=np.float64).reshape(4, 3, 2),
        mask=False,
    )
    vals[0, 0, 0] = np.ma.masked
    vals[3, 2, 1] = np.ma.masked

    original = xtgeo.GridProperty(ncol=4, nrow=3, nlay=2, values=vals, name="PROP")
    data = GridPropertyDataResInsight.from_xtgeo_gridproperty(
        original, property_type="STATIC_NATIVE"
    )

    # Inactive cells should have actnum 0 and the UNDEF sentinel
    idx = np.ravel_multi_index((0, 0, 0), (4, 3, 2), order="F")
    assert data.actnumsv[idx] == 0
    assert data.values[idx] == UNDEF

    restored = data.to_xtgeo_gridproperty()
    assert np.ma.is_masked(restored.values[0, 0, 0])
    assert np.ma.is_masked(restored.values[3, 2, 1])
    assert_allclose(restored.values.compressed(), original.values.compressed())


def test_roundtrip_discrete_inactive_uses_undef_int():
    """Inactive cells in discrete properties use UNDEF_INT, not UNDEF."""
    vals = np.ma.MaskedArray(
        np.ones((2, 2, 2), dtype=np.int32),
        mask=False,
    )
    vals[0, 0, 0] = np.ma.masked

    original = xtgeo.GridProperty(
        ncol=2,
        nrow=2,
        nlay=2,
        values=vals,
        name="ZONE",
        discrete=True,
        codes={1: "Upper"},
    )
    data = GridPropertyDataResInsight.from_xtgeo_gridproperty(
        original, property_type="STATIC_NATIVE"
    )

    idx = np.ravel_multi_index((0, 0, 0), (2, 2, 2), order="F")
    assert data.values[idx] == UNDEF_INT
    assert data.values.dtype == np.int32


def test_from_xtgeo_with_grid_actnum_intersection():
    """When grid is provided, actnum is intersection of prop mask and grid actnum."""
    grid = xtgeo.create_box_grid((4, 3, 2))
    actnum = grid.get_actnum()
    actnum.values[1, 1, 0] = 0
    grid.set_actnum(actnum)

    prop = xtgeo.GridProperty(ncol=4, nrow=3, nlay=2, values=0.5, name="P")
    data = GridPropertyDataResInsight.from_xtgeo_gridproperty(
        prop, property_type="STATIC_NATIVE", grid=grid
    )

    idx = np.ravel_multi_index((1, 1, 0), (4, 3, 2), order="F")
    assert data.actnumsv[idx] == 0
    assert data.values[idx] == UNDEF


# ---------------------------------------------------------------------------
# __eq__
# ---------------------------------------------------------------------------


def _make_data(**overrides):
    """Create a GridPropertyDataResInsight with sensible defaults."""
    kw = {
        "name": "P",
        "nx": 2,
        "ny": 2,
        "nz": 2,
        "values": np.ones(8, dtype=np.float64),
        "actnumsv": np.ones(8, dtype=np.int32),
        "property_type": "STATIC_NATIVE",
        "time_step_index": 0,
        "discrete": False,
        "codes": {},
        "filesrc": "",
    }
    kw.update(overrides)
    return GridPropertyDataResInsight(**kw)


def test_eq():
    """__eq__ compares all fields; returns NotImplemented for foreign types."""
    a = _make_data()
    assert a == _make_data()
    assert a != _make_data(name="Q")
    assert a.__eq__("not a data object") is NotImplemented


# ---------------------------------------------------------------------------
# _read_actnum failure path
# ---------------------------------------------------------------------------


def test_read_actnum_raises_on_impossible_size(resinsight_instance):
    """_read_actnum raises RuntimeError when expected_size doesn't match grid."""
    from xtgeo.interfaces.resinsight._resinsight_base import _BaseResInsightDataRW

    base = _BaseResInsightDataRW(resinsight_instance)
    case = base.get_case(case_name="EXAMPLE", find_last=True)

    with pytest.raises(RuntimeError, match="Could not read ACTNUM"):
        _read_actnum(case, expected_size=7)


# ---------------------------------------------------------------------------
# GridPropertyWriter.save dimension guard
# ---------------------------------------------------------------------------


def test_writer_rejects_dimension_mismatch(resinsight_instance):
    """save raises ValueError when property dims don't match case grid."""
    data = _make_data(
        nx=99,
        ny=99,
        nz=99,
        values=np.ones(99**3, dtype=np.float64),
        actnumsv=np.ones(99**3, dtype=np.int32),
    )
    writer = GridPropertyWriter(instance_or_port=resinsight_instance)
    with pytest.raises(ValueError, match="dimensions"):
        writer.save(data, case_name="EXAMPLE")


def test_writer_rejects_nonexistent_case(resinsight_instance):
    """save raises RuntimeError when case name is not found."""
    data = _make_data()
    writer = GridPropertyWriter(instance_or_port=resinsight_instance)
    with pytest.raises(RuntimeError, match="Cannot find any case with name"):
        writer.save(data, case_name="NONEXISTENT_CASE")
