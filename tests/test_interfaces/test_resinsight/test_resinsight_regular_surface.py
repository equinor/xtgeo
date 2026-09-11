"""Tests for the ResInsight regular-surface reader/writer and data container.

Only the tests marked ``requires_resinsight`` need a running ResInsight.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.interfaces.resinsight._regular_surface import (
    _NAME_ATTR,
    RegularSurfaceDataResInsight,
    RegularSurfaceReader,
    RegularSurfaceWriter,
    _find_regular_surface,
)
from xtgeo.interfaces.resinsight._resinsight_base import resolve_folder

if TYPE_CHECKING:
    from xtgeo.interfaces.resinsight._rips_package import RipsInstanceType


@pytest.fixture
def make_data():
    """Factory for ``RegularSurfaceDataResInsight`` with overridable defaults.

    ``values`` defaults to ``arange(ncol * nrow)`` so its size always matches
    the geometry unless an explicit array is supplied.
    """

    def _make(**overrides) -> RegularSurfaceDataResInsight:
        params: dict = {
            "name": "S1",
            "ncol": 2,
            "nrow": 3,
            "xori": 100.0,
            "yori": 200.0,
            "xinc": 25.0,
            "yinc": 50.0,
            "rotation": 10.0,
        }
        params.update(overrides)
        params.setdefault(
            "values", np.arange(params["ncol"] * params["nrow"], dtype=np.float64)
        )
        return RegularSurfaceDataResInsight(**params)

    return _make


@pytest.fixture
def reader(resinsight_instance: RipsInstanceType) -> RegularSurfaceReader:
    return RegularSurfaceReader(resinsight_instance)


@pytest.fixture
def writer(resinsight_instance: RipsInstanceType) -> RegularSurfaceWriter:
    return RegularSurfaceWriter(resinsight_instance)


# ---------------------------------------------------------------------------
# Conversion XTGeo RegularSurface <-> data container (no ResInsight needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ncol, nrow, rotation", [(5, 4, 0.0), (6, 5, 45.0), (3, 3, 90.0)]
)
def test_roundtrip_xtgeo_to_data_to_xtgeo(ncol, nrow, rotation):
    """Geometry, values and metadata survive the conversion roundtrip."""
    original = xtgeo.RegularSurface(
        ncol=ncol,
        nrow=nrow,
        xinc=25.0,
        yinc=30.0,
        xori=100.0,
        yori=200.0,
        rotation=rotation,
        values=np.random.default_rng(42).random((ncol, nrow)),
    )

    data = RegularSurfaceDataResInsight.from_xtgeo_surface(original, name="ROUNDTRIP")
    assert data.values.size == ncol * nrow

    restored = data.to_xtgeo_surface()

    assert (restored.ncol, restored.nrow) == (ncol, nrow)
    assert restored.rotation == pytest.approx(rotation)
    assert restored.name == "ROUNDTRIP"
    assert_allclose(restored.values, original.values, atol=1e-10)


def test_values_are_flattened_in_x_fastest_order():
    """Values are flattened in Fortran (X-fastest) order, masks become NaN."""
    values = np.ma.array(
        [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],  # shape (ncol=2, nrow=3)
        mask=[[False, False, True], [False, False, False]],
    )
    surf = xtgeo.RegularSurface(ncol=2, nrow=3, xinc=5.0, yinc=5.0, values=values)

    data = RegularSurfaceDataResInsight.from_xtgeo_surface(surf, name="ASYM")

    expected = np.array([10.0, 40.0, 20.0, 50.0, np.nan, 60.0])
    assert_allclose(data.values, expected)
    assert_allclose(data.to_xtgeo_surface().values, values, atol=1e-10)


def test_values_size_must_match_geometry(make_data):
    """A values array not matching ncol * nrow is rejected."""
    with pytest.raises(ValueError, match="values should have length 6"):
        make_data(values=np.zeros(5))


def test_equality_compares_values_not_identity(make_data):
    """Containers compare by content; NaN in the same position counts as equal."""
    assert make_data() == make_data()
    assert make_data() != make_data(xori=999.0)
    assert make_data(values=np.full(6, np.nan)) == make_data(values=np.full(6, np.nan))
    assert make_data() != "not a container"


@pytest.mark.parametrize("rotation", [0.0, 30.0, 90.0])
def test_righthanded_surface_normalizes_without_shifting_geometry(rotation):
    """A make_righthanded() surface round-trips through the data container.

    This mirrors, without needing a live ResInsight, the normalisation done
    in ``RegularSurface.to_resinsight()``: ``make_righthanded()`` leaves the
    surface with ``yflip=-1`` and a negative ``yinc``, and calling
    ``make_lefthanded()`` on a copy must undo that transform exactly, not
    apply a further shift.
    """
    original = xtgeo.RegularSurface(
        ncol=3,
        nrow=4,
        xinc=10.0,
        yinc=20.0,
        xori=100.0,
        yori=200.0,
        rotation=rotation,
        values=np.arange(12, dtype=np.float64).reshape(3, 4),
    )
    original.make_righthanded()
    assert original.yflip == -1
    assert original.yinc < 0

    use_srf = original.copy()
    use_srf.make_lefthanded()
    assert use_srf.yflip == 1
    assert use_srf.yinc > 0

    data = RegularSurfaceDataResInsight.from_xtgeo_surface(use_srf, name="YFLIP_SURF")
    reloaded = data.to_xtgeo_surface()

    for icol in range(1, original.ncol + 1):
        for jrow in range(1, original.nrow + 1):
            assert_allclose(
                reloaded.get_xy_value_from_ij(icol, original.nrow + 1 - jrow),
                original.get_xy_value_from_ij(icol, jrow),
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# Integration tests (require running ResInsight)
# ---------------------------------------------------------------------------


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_reader_handles_constant_depth_surface(
    resinsight_instance: RipsInstanceType, reader
):
    """A ResInsight surface with a constant depth has no value array."""
    surface = resinsight_instance.project.surface_folder().new_regular_surface(
        name="FIXED_DEPTH_SURF",
        nx=3,
        ny=4,
        origin_x=0.0,
        origin_y=0.0,
        increment_x=10.0,
        increment_y=10.0,
        depth=123.5,
    )
    surface.update()

    data = reader.load("FIXED_DEPTH_SURF")

    assert data.values.size == 12
    assert_allclose(data.values, 123.5)


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_geometry_matches_resinsight_float32_coordinates(
    resinsight_instance: RipsInstanceType, make_data
):
    """Geometry matching accepts ResInsight's float32 coordinate precision."""
    data = make_data(name="PRECISION_TEST", xori=6788228.1234, yori=6123456.9876)
    surface = resinsight_instance.project.surface_folder().new_regular_surface(
        name=data.name,
        nx=data.ncol,
        ny=data.nrow,
        origin_x=data.xori,
        origin_y=data.yori,
        increment_x=data.xinc,
        increment_y=data.yinc,
        rotation=data.rotation,
    )
    surface.update()

    assert RegularSurfaceWriter._regular_surface_geometry_matches(surface, data)


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_resolve_surface_folder(resinsight_instance: RipsInstanceType):
    """Nested folder paths are created once, reused after, else resolve None."""
    root = resinsight_instance.project.surface_folder()

    created = resolve_folder(root, "PATH_A/PATH_B", _NAME_ATTR, create=True)
    assert created.surface_user_description == "PATH_B"

    assert resolve_folder(root, "PATH_A/PATH_B", _NAME_ATTR, create=True) is not None
    parent = resolve_folder(root, "PATH_A", _NAME_ATTR)
    names = [f.surface_user_description for f in parent.sub_collections()]
    assert names.count("PATH_B") == 1

    assert resolve_folder(root, "NO_SUCH_FOLDER", _NAME_ATTR) is None


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_surface_lookup_is_not_recursive(
    resinsight_instance: RipsInstanceType, writer, make_data
):
    """A surface in a sub-folder is not found from the parent folder."""
    root = resinsight_instance.project.surface_folder()
    writer.save(make_data(), surface_name="NESTED_SURF", folder_name="NESTED_LOOKUP")

    assert _find_regular_surface(root, "NESTED_SURF") is None

    found = _find_regular_surface(
        resolve_folder(root, "NESTED_LOOKUP", _NAME_ATTR), "NESTED_SURF"
    )
    assert found.surface_user_description == "NESTED_SURF"


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_writer_overwrites_same_named_non_regular_surface(
    resinsight_instance: RipsInstanceType, writer, make_data
):
    """A name taken by another surface type is only overwritten with replace=True.

    ResInsight keeps all surface types in one namespace, but the regular-surface
    lookup does not see them, so the conflict surfaces from ResInsight itself.
    """
    folder = resinsight_instance.project.surface_folder()
    other = folder.new_surface(case=resinsight_instance.project.cases()[0], k_index=0)
    other.surface_user_description = "TYPE_CLASH"
    other.update()

    data = make_data(name="TYPE_CLASH")
    with pytest.raises(RuntimeError, match="already exists"):
        writer.save(data, surface_name="TYPE_CLASH")

    writer.save(data, surface_name="TYPE_CLASH", replace=True)
    assert _find_regular_surface(folder, "TYPE_CLASH") is not None
