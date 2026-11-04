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
    resolve_polygon_folder,
    resolve_surface_folder,
    select_by_name,
    validate_case,
)
from xtgeo.interfaces.resinsight._rips_package import require_rips


class _SurfaceContainer:
    def __init__(self, name: str):
        self.surface_user_description = name


class _SurfaceFolder(_SurfaceContainer):
    def __init__(self, name: str):
        super().__init__(name)
        self._children: list[_SurfaceFolder] = []

    def children(self, child_field, class_definition):
        assert child_field == "SubCollections"
        assert class_definition is _SurfaceFolder
        return self._children

    def add_folder(self, folder_name, on_name_conflict):
        child = _SurfaceFolder(folder_name)
        self._children.append(child)
        return child


class _PolygonContainer:
    def __init__(self, name: str):
        self.polygon_collection_name = name


class _PolygonFolder(_PolygonContainer):
    def __init__(self, name: str):
        super().__init__(name)
        self._children: list[_PolygonFolder] = []

    def children(self, child_field, class_definition):
        assert child_field == "SubCollections"
        assert class_definition is _PolygonFolder
        return self._children

    def add_folder(self, folder_name, on_name_conflict):
        child = _PolygonFolder(folder_name)
        self._children.append(child)
        return child


def _fake_rips():
    return SimpleNamespace(
        PolygonCollection=_PolygonFolder,
        SurfaceCollection=_SurfaceFolder,
    )


@pytest.mark.parametrize(
    "resolver, folder_type, name_attr",
    [
        (resolve_surface_folder, _SurfaceFolder, "surface_user_description"),
        (resolve_polygon_folder, _PolygonFolder, "polygon_collection_name"),
    ],
)
def test_folder_resolvers_create_and_reuse_nested_path(
    resolver, folder_type, name_attr
):
    root = folder_type("root")

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
    ):
        created = resolver(root, "A/B", create=True)
        found = resolver(root, "A/B")

    assert created is found
    assert getattr(created, name_attr) == "B"
    assert len(root._children) == 1


@pytest.mark.parametrize(
    "resolver, folder_type",
    [
        (resolve_surface_folder, _SurfaceFolder),
        (resolve_polygon_folder, _PolygonFolder),
    ],
)
@pytest.mark.parametrize("use_subclass", [False, True])
@pytest.mark.parametrize("create", [False, True])
def test_folder_resolvers_include_exact_type_and_subclasses(
    resolver, folder_type, use_subclass, create
):
    class SpecialFolder(folder_type):
        pass

    root = folder_type("root")
    child_type = SpecialFolder if use_subclass else folder_type
    matching_child = child_type("TARGET")
    root._children.extend([matching_child, folder_type("TARGET")])

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(root, "add_folder", wraps=root.add_folder) as add_folder,
    ):
        found = resolver(root, "TARGET", create=create)

    assert found is matching_child
    add_folder.assert_not_called()


@pytest.mark.parametrize(
    "resolver, folder_type, container_type, name_attr, folder_kind",
    [
        (
            resolve_surface_folder,
            _SurfaceFolder,
            _SurfaceContainer,
            "surface_user_description",
            "surface",
        ),
        (
            resolve_polygon_folder,
            _PolygonFolder,
            _PolygonContainer,
            "polygon_collection_name",
            "polygon",
        ),
    ],
)
@pytest.mark.parametrize("child_kind", ["base", "sibling", "unrelated"])
@pytest.mark.parametrize("outcome", ["existing", "missing", "create"])
def test_folder_resolvers_exclude_other_wrapper_types(
    resolver, folder_type, container_type, name_attr, folder_kind, child_kind, outcome
):
    class SiblingContainer(container_type):
        pass

    root = folder_type("root")
    if child_kind == "base":
        excluded = container_type("TARGET")
    elif child_kind == "sibling":
        excluded = SiblingContainer("TARGET")
    else:
        excluded = SimpleNamespace(**{name_attr: "TARGET"})
    root._children.append(excluded)
    if outcome == "existing":
        root._children.append(folder_type("TARGET"))

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
        patch.object(root, "add_folder", wraps=root.add_folder) as add_folder,
    ):
        if outcome == "missing":
            with pytest.raises(
                RuntimeError, match=f"Cannot find {folder_kind} folder 'TARGET'"
            ):
                resolver(root, "TARGET")
        else:
            found = resolver(root, "TARGET", create=outcome == "create")
            assert found is root._children[1]
            assert isinstance(found, folder_type)

    assert root._children[0] is excluded
    if outcome == "create":
        add_folder.assert_called_once_with(
            folder_name="TARGET", on_name_conflict="FAIL"
        )
    else:
        add_folder.assert_not_called()


@pytest.mark.parametrize(
    "resolver, folder_type",
    [
        (resolve_surface_folder, _SurfaceFolder),
        (resolve_polygon_folder, _PolygonFolder),
    ],
)
def test_folder_resolvers_include_already_wrapped_fallback(resolver, folder_type):
    """Model the result of fallback wrapping, not the rips conversion itself."""
    root = folder_type("root")
    fallback_child = folder_type("TARGET")
    server_object = SimpleNamespace(class_keyword="UnknownCollection")
    fallback_child.pb2_object = lambda: server_object
    root._children.append(fallback_child)

    with patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()):
        found = resolver(root, "TARGET")

    assert found is fallback_child
    assert found.pb2_object().class_keyword == "UnknownCollection"


@pytest.mark.parametrize(
    "resolver, folder_type",
    [
        (resolve_surface_folder, _SurfaceFolder),
        (resolve_polygon_folder, _PolygonFolder),
    ],
)
@pytest.mark.parametrize("create", [False, True])
def test_folder_resolvers_return_root_for_empty_path(resolver, folder_type, create):
    root = folder_type("root")

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(root, "children", wraps=root.children) as children,
        patch.object(root, "add_folder", wraps=root.add_folder) as add_folder,
    ):
        assert resolver(root, "", create=create) is root

    children.assert_not_called()
    add_folder.assert_not_called()


@pytest.mark.parametrize(
    "resolver, folder_type, folder_kind",
    [
        (resolve_surface_folder, _SurfaceFolder, "surface"),
        (resolve_polygon_folder, _PolygonFolder, "polygon"),
    ],
)
@pytest.mark.parametrize("create", [False, True])
def test_folder_resolvers_handle_partially_existing_path(
    resolver, folder_type, folder_kind, create
):
    root = folder_type("root")
    parent = folder_type("A")
    retained = folder_type("KEEP")
    root._children.append(parent)
    parent._children.append(retained)

    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
        patch.object(root, "add_folder", wraps=root.add_folder) as root_add,
        patch.object(parent, "add_folder", wraps=parent.add_folder) as parent_add,
    ):
        if create:
            created = resolver(root, "A/B", create=True)
            assert parent._children == [retained, created]
            assert resolver(root, "A/B") is created
            parent_add.assert_called_once_with(folder_name="B", on_name_conflict="FAIL")
        else:
            with pytest.raises(
                RuntimeError, match=f"Cannot find {folder_kind} folder 'B'"
            ):
                resolver(root, "A/B")
            assert parent._children == [retained]
            parent_add.assert_not_called()

    assert root._children == [parent]
    root_add.assert_not_called()


@pytest.mark.requires_resinsight
@pytest.mark.xdist_group(name="resinsight")
def test_resolve_polygon_folder_with_rips_creates_nested_path(resinsight_instance):
    rips = require_rips()
    root = resinsight_instance.project.descendants(rips.PolygonCollection)[0]

    path = "XTGEO_CREATE_PATH_A/XTGEO_CREATE_PATH_B"
    created = resolve_polygon_folder(root, path, create=True)
    assert isinstance(created, rips.PolygonCollection)
    assert created.polygon_collection_name == "XTGEO_CREATE_PATH_B"
    found = resolve_polygon_folder(root, path)
    assert isinstance(found, rips.PolygonCollection)
    assert found.address() == created.address()


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
        for child in parent.children("SubCollections", rips.PolygonCollection)
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
            lambda root: resolve_surface_folder(root, "MISSING"),
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
def test_folder_resolvers_raise_for_missing_path(resolver, root, expected):
    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        pytest.raises(RuntimeError, match=expected),
    ):
        resolver(root)


@pytest.mark.parametrize(
    "resolver, root",
    [
        (
            lambda root: resolve_surface_folder(root, "INVALID", create=True),
            _SurfaceFolder("root"),
        ),
        (
            lambda root: resolve_polygon_folder(root, "INVALID", create=True),
            _PolygonFolder("root"),
        ),
    ],
)
def test_folder_resolvers_create_without_overwriting(resolver, root):
    with (
        patch.object(_resinsight_base, "require_rips", return_value=_fake_rips()),
        patch.object(
            _resinsight_base, "NameConflictPolicy", SimpleNamespace(FAIL="FAIL")
        ),
        patch.object(root, "add_folder", wraps=root.add_folder) as add_folder,
    ):
        created = resolver(root)

    assert created is root._children[0]
    add_folder.assert_called_once_with(folder_name="INVALID", on_name_conflict="FAIL")


@pytest.mark.parametrize("find_last, expected_index", [(False, 0), (True, 2)])
def test_select_by_name_selects_first_or_last_match(find_last, expected_index):
    items = [SimpleNamespace(name=name) for name in ["MATCH", "OTHER", "MATCH"]]

    assert (
        select_by_name(iter(items), "MATCH", find_last=find_last)
        is items[expected_index]
    )
    assert select_by_name(iter(items), "MATCH") is items[2]


@pytest.mark.parametrize("items", [[], [object(), SimpleNamespace(name="OTHER")]])
def test_select_by_name_returns_none_without_match(items):
    assert select_by_name(iter(items), "MISSING") is None


def test_select_by_name_uses_custom_name_attribute():
    item = SimpleNamespace(name="OTHER", label="MATCH")

    assert select_by_name([object(), item], "MATCH", name_attr="label") is item


@pytest.mark.parametrize("find_last, expected_index", [(False, 0), (True, 2)])
def test_get_case_selects_first_or_last_match(find_last, expected_index):
    base = _BaseResInsightDataRW()
    cases = [SimpleNamespace(name=name) for name in ["MATCH", "OTHER", "MATCH"]]
    project = SimpleNamespace(cases=lambda: cases)

    with patch.object(base, "get_project", return_value=project):
        assert base.get_case("MATCH", find_last=find_last) is cases[expected_index]
        assert base.get_case("MATCH") is cases[2]


@pytest.mark.parametrize("cases", [[], [SimpleNamespace(name="OTHER")]])
def test_get_case_returns_none_without_match(cases):
    base = _BaseResInsightDataRW()
    project = SimpleNamespace(cases=lambda: cases)

    with patch.object(base, "get_project", return_value=project):
        assert base.get_case("MISSING") is None


@pytest.mark.parametrize("find_last", [False, True])
@pytest.mark.parametrize("case", [None, SimpleNamespace(name="MATCH")])
def test_resolve_case_forwards_name_and_selection_policy(find_last, case):
    base = _BaseResInsightDataRW()

    with patch.object(base, "get_case", return_value=case) as get_case:
        assert base.resolve_case("MATCH", find_last=find_last) is case

    get_case.assert_called_once_with(case_name="MATCH", find_last=find_last)


@pytest.mark.parametrize("instance_or_port", [None, 5000, object()])
def test_ripsapi_utils_is_created_lazily_and_cached(instance_or_port):
    with patch.object(_resinsight_base, "RipsApiUtils") as make_utils:
        base = _BaseResInsightDataRW(instance_or_port)
        make_utils.assert_not_called()

        assert base.get_ripsapi_utils() is make_utils.return_value
        assert base.get_ripsapi_utils() is make_utils.return_value
        assert base.get_instance() is make_utils.return_value.instance
        assert base.get_project() is make_utils.return_value.project

    make_utils.assert_called_once_with(instance_or_port)


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
