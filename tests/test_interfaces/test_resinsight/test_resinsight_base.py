"""Unit tests for the shared ResInsight read/write base helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.resinsight._resinsight_base import (
    _BaseResInsightDataRW,
    validate_case,
)


def test_resolve_case_returns_case_object_unchanged():
    base = _BaseResInsightDataRW(instance_or_port=None)
    case = SimpleNamespace(name="KEEP")
    assert base.resolve_case(case) is case


@pytest.mark.parametrize("bad", [123, object(), None, SimpleNamespace(name=42)])
def test_resolve_case_rejects_invalid_argument(bad):
    base = _BaseResInsightDataRW(instance_or_port=None)
    with pytest.raises(TypeError, match="case must be a case name"):
        base.resolve_case(bad)


@pytest.mark.parametrize("good", ["MYCASE", SimpleNamespace(name="FC")])
def test_validate_case_accepts_valid_argument(good):
    assert validate_case(good) is None


@pytest.mark.parametrize("bad", [123, object(), None, SimpleNamespace(name=42)])
def test_validate_case_rejects_invalid_argument(bad):
    with pytest.raises(TypeError, match="case must be a case name"):
        validate_case(bad)


# ---------------------------------------------------------------------------
# Early validation at the public API boundary (before expensive extraction)
# ---------------------------------------------------------------------------


def test_grid_to_resinsight_rejects_invalid_case_early():
    grd = xtgeo.create_box_grid((2, 2, 2))
    with pytest.raises(TypeError, match="case must be a case name"):
        grd.to_resinsight(5000, case=123)


def test_gridproperty_to_resinsight_rejects_invalid_case_early():
    gprop = xtgeo.GridProperty(ncol=2, nrow=2, nlay=2, values=np.ones((2, 2, 2)))
    with pytest.raises(TypeError, match="case must be a case name"):
        gprop.to_resinsight(5000, case=123)
