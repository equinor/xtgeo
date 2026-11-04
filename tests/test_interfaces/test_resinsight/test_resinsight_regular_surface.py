"""Tests for the ResInsight regular-surface reader/writer and data container.

Only the tests marked ``requires_resinsight`` need a running ResInsight.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.interfaces.resinsight import _regular_surface
from xtgeo.interfaces.resinsight._regular_surface import (
    RegularSurfaceDataResInsight,
    RegularSurfaceReader,
    RegularSurfaceWriter,
    _find_regular_surface,
)
from xtgeo.interfaces.resinsight._resinsight_base import resolve_surface_folder

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


@pytest.fixture
def offline_writer(monkeypatch):
    writer = RegularSurfaceWriter()
    monkeypatch.setattr(_regular_surface, "require_rips", lambda: SimpleNamespace())
    monkeypatch.setattr(
        writer,
        "get_project",
        lambda: SimpleNamespace(surface_folder=lambda: object()),
    )
    monkeypatch.setattr(
        _regular_surface,
        "resolve_surface_folder",
        lambda root, folder_path, create: object(),
    )
    return writer


def test_writer_wraps_rips_errors(make_data, monkeypatch, offline_writer):
    """ResInsight failures are presented as the writer's documented error type."""

    class FakeRipsError(Exception):
        pass

    def raise_rips_error(*args, **kwargs):
        raise FakeRipsError("name collision")

    writer = offline_writer
    monkeypatch.setattr(
        _regular_surface,
        "require_rips",
        lambda: SimpleNamespace(RipsError=FakeRipsError),
    )
    monkeypatch.setattr(writer, "_get_or_create_regular_surface", raise_rips_error)

    with pytest.raises(RuntimeError, match="Failed to save ResInsight") as exc_info:
        writer.save(make_data(), surface_name="COLLISION")

    assert isinstance(exc_info.value.__cause__, FakeRipsError)


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
@pytest.mark.parametrize(
    "operation", ["_create_regular_surface", "_set_regular_surface_property"]
)
def test_writer_wraps_operational_errors(
    make_data, monkeypatch, offline_writer, error_type, operation
):
    error = error_type("operation failed")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(_regular_surface, "_find_regular_surface", lambda *args: None)
    monkeypatch.setattr(
        offline_writer, "_create_regular_surface", lambda **kwargs: object()
    )
    monkeypatch.setattr(offline_writer, operation, fail)

    with pytest.raises(RuntimeError, match="Failed to save ResInsight") as exc_info:
        offline_writer.save(make_data(), surface_name="TEST")

    assert type(exc_info.value) is RuntimeError
    assert exc_info.value.__cause__ is error


def test_writer_wraps_value_conversion_errors(make_data, monkeypatch, offline_writer):
    data = make_data(values=np.full(6, "not-a-number"))
    monkeypatch.setattr(
        offline_writer,
        "_get_or_create_regular_surface",
        lambda *args, **kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="Failed to save ResInsight") as exc_info:
        offline_writer.save(data, surface_name="TEST")

    assert isinstance(exc_info.value.__cause__, ValueError)


@pytest.mark.parametrize("folder_name", ["", "A/B"])
def test_writer_preserves_geometry_mismatch_error(
    make_data, monkeypatch, offline_writer, folder_name
):
    monkeypatch.setattr(
        _regular_surface, "_find_regular_surface", lambda *args: object()
    )
    monkeypatch.setattr(
        offline_writer, "_regular_surface_geometry_matches", lambda *args: False
    )

    def unexpected_operation(*args, **kwargs):
        pytest.fail("Geometry mismatch must fail before creation or property writes")

    monkeypatch.setattr(offline_writer, "_create_regular_surface", unexpected_operation)
    monkeypatch.setattr(
        offline_writer, "_set_regular_surface_property", unexpected_operation
    )

    with pytest.raises(RuntimeError) as exc_info:
        offline_writer.save(make_data(), surface_name="TEST", folder_name=folder_name)

    assert str(exc_info.value) == (
        f"A surface named 'TEST' already exists in folder '{folder_name or '<root>'}' "
        "with a different geometry. Pass replace=True to overwrite it."
    )
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("has_existing_surface", "geometry_matches", "replace"),
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_writer_logs_create_reuse_and_replace_decisions(
    has_existing_surface, geometry_matches, replace, monkeypatch
):
    """Successful saves log whether a surface is reused or created."""
    existing_surface = object() if has_existing_surface else None
    created_surface = object()
    debug_calls = []
    writer = RegularSurfaceWriter()

    monkeypatch.setattr(
        _regular_surface,
        "_find_regular_surface",
        lambda folder, name: existing_surface,
    )
    monkeypatch.setattr(
        writer,
        "_regular_surface_geometry_matches",
        lambda surface, data: geometry_matches,
    )
    monkeypatch.setattr(
        writer,
        "_create_regular_surface",
        lambda **kwargs: created_surface,
    )
    monkeypatch.setattr(
        _regular_surface.logger,
        "debug",
        lambda *args: debug_calls.append(args),
    )

    surface = writer._get_or_create_regular_surface(
        target_folder=object(),
        surface_name="SURFACE",
        data=object(),
        folder_name=None,
        replace=replace,
    )

    expected_surface = existing_surface if geometry_matches else created_surface
    assert surface is expected_surface
    assert debug_calls == [
        ("Reusing existing surface '%s' ", "SURFACE")
        if geometry_matches
        else (
            "Creating new surface '%s' (overwrite=%s)",
            "SURFACE",
            replace,
        )
    ]


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
    """Nested paths are created once, reused, and missing folders raise."""
    root = resinsight_instance.project.surface_folder()

    created = resolve_surface_folder(root, "PATH_A/PATH_B", create=True)
    assert created.surface_user_description == "PATH_B"

    assert resolve_surface_folder(root, "PATH_A/PATH_B", create=True) is not None
    parent = resolve_surface_folder(root, "PATH_A")
    names = [f.surface_user_description for f in parent.sub_collections()]
    assert names.count("PATH_B") == 1

    with pytest.raises(RuntimeError, match="Cannot find surface folder"):
        resolve_surface_folder(root, "NO_SUCH_FOLDER")


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
        resolve_surface_folder(root, "NESTED_LOOKUP"), "NESTED_SURF"
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
    # The live project can retain surfaces from earlier test runs
    surface_name = f"TYPE_CLASH_{uuid4().hex}"
    other = folder.new_surface(case=resinsight_instance.project.cases()[0], k_index=0)
    other.surface_user_description = surface_name
    other.update()

    data = make_data(name=surface_name)
    with pytest.raises(RuntimeError, match="already exists"):
        writer.save(data, surface_name=surface_name)

    writer.save(data, surface_name=surface_name, replace=True)
    assert _find_regular_surface(folder, surface_name) is not None
