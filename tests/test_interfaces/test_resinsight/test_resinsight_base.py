"""Unit tests for the shared ResInsight read/write base helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import xtgeo
from xtgeo.interfaces.resinsight import _resinsight_base
from xtgeo.interfaces.resinsight._resinsight_base import (
    _BaseResInsightDataRW,
    resolve_folder,
    resolve_polygon_folder,
    validate_case,
)
from xtgeo.interfaces.resinsight._rips_package import require_rips


class _SurfaceFolder:
    def __init__(self, name: str):
        self.surface_user_description = name
        self._children: list[_SurfaceFolder] = []

    def sub_collections(self):
        return self._children

    def add_folder(self, folder_name, on_name_conflict):
        child = _SurfaceFolder(folder_name)
        self._children.append(child)
        return child


class _PolygonFolder:
    def __init__(self, name: str):
        self.polygon_collection_name = name
        self._children: list[_PolygonFolder] = []

    def children(self, child_field, class_definition):
        return [
            child for child in self._children if isinstance(child, class_definition)
        ]

    def add_folder(self, folder_name, on_name_conflict):
        child = _PolygonFolder(folder_name)
        self._children.append(child)
        return child


def _fake_rips():
    return SimpleNamespace(
        PolygonCollection=_PolygonFolder,
        RimPolygonContainer=_PolygonFolder,
        SurfaceCollection=_SurfaceFolder,
    )


def test_resolve_surface_folder_adapter_creates_and_reuses_nested_path():
    root = _SurfaceFolder("root")

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
    ):
        created = resolve_folder(root, "A/B", "surface_user_description", create=True)
        found = resolve_folder(root, "A/B", "surface_user_description")

    assert created is found
    assert created.surface_user_description == "B"
    assert len(root._children) == 1


def test_resolve_polygon_folder_adapter_creates_and_reuses_nested_path():
    root = _PolygonFolder("root")

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
    ):
        created = resolve_polygon_folder(root, "A/B", create=True)
        found = resolve_polygon_folder(root, "A/B")

    assert created is found
    assert created.polygon_collection_name == "B"
    assert len(root._children) == 1


def test_resolve_polygon_folder_ignores_non_container_children():
    root = _PolygonFolder("root")
    non_container = SimpleNamespace(polygon_collection_name="TARGET")
    # Invalid child and valid child to test filtering logic.
    root._children.extend([non_container, _PolygonFolder("TARGET")])

    with patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()):
        found = resolve_polygon_folder(root, "TARGET")

    assert found is root._children[1]


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_resolve_polygon_folder_with_rips_creates_nested_path(resinsight_instance):
    rips = require_rips()
    root = resinsight_instance.project.descendants(rips.PolygonCollection)[0]

    created = resolve_polygon_folder(
        root, "XTGEO_CREATE_PATH_A/XTGEO_CREATE_PATH_B", create=True
    )
    assert isinstance(created, rips.RimPolygonContainer)
    assert created.polygon_collection_name == "XTGEO_CREATE_PATH_B"


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_resolve_polygon_folder_with_rips_reuses_existing_path(resinsight_instance):
    rips = require_rips()
    root = resinsight_instance.project.descendants(rips.PolygonCollection)[0]

    path = "XTGEO_REUSE_PATH_A/XTGEO_REUSE_PATH_B"
    # The first call creates the path; the second must reuse it without duplication.
    resolve_polygon_folder(root, path, create=True)
    resolve_polygon_folder(root, path, create=True)
    parent = resolve_polygon_folder(root, "XTGEO_REUSE_PATH_A")
    child_names = [
        child.polygon_collection_name
        for child in parent.children("SubCollections", rips.RimPolygonContainer)
    ]
    assert child_names.count("XTGEO_REUSE_PATH_B") == 1


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_resolve_polygon_folder_with_rips_raises_for_missing_path(
    resinsight_instance,
):
    rips = require_rips()
    root = resinsight_instance.project.descendants(rips.PolygonCollection)[0]

    with pytest.raises(RuntimeError, match="Cannot find polygon folder"):
        resolve_polygon_folder(root, "NO_SUCH_FOLDER")


@pytest.mark.parametrize(
    "resolver, root, expected",
    [
        (
            lambda root: resolve_folder(root, "MISSING", "surface_user_description"),
            _SurfaceFolder("root"),
            "Cannot find surface folder",
        ),
        (
            lambda root: resolve_polygon_folder(root, "MISSING"),
            _PolygonFolder("root"),
            "Cannot find polygon folder",
        ),
    ],
)
def test_folder_adapters_raise_for_missing_path(resolver, root, expected):
    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        pytest.raises(RuntimeError, match=expected),
    ):
        resolver(root)


@pytest.mark.parametrize(
    "resolver, root, expected",
    [
        (
            lambda root: resolve_folder(
                root, "INVALID", "surface_user_description", create=True
            ),
            _SurfaceFolder("root"),
            "invalid surface folder type",
        ),
        (
            lambda root: resolve_polygon_folder(root, "INVALID", create=True),
            _PolygonFolder("root"),
            "invalid polygon folder type",
        ),
    ],
)
def test_folder_adapters_reject_invalid_created_type(resolver, root, expected):
    root.add_folder = lambda folder_name, on_name_conflict: object()

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
        pytest.raises(RuntimeError, match=expected),
    ):
        resolver(root)


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
