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
    select_by_name,
    validate_case,
)


class FakeFolder:
    """Minimal rips folder-collection test double used by folder helper tests.

    The object models only the collection operations required by
    ``resolve_folder``. It keeps an in-memory child hierarchy and records folder
    creation calls so tests can assert both the result and the interaction.
    """

    def __init__(self, name, children=None):
        """Create a named folder with optional pre-existing child folders."""
        self.folder_name = name
        self.children = [] if children is None else children
        self.add_calls = []

    def sub_collections(self):
        return self.children

    def add_folder(self, folder_name, on_name_conflict):
        """Create a child folder and record the requested conflict policy."""
        self.add_calls.append((folder_name, on_name_conflict))
        child = FakeFolder(folder_name)
        self.children.append(child)
        return child


def test_select_by_name_supports_first_last_and_missing_matches():
    """Select the first or last matching item and return None when absent."""
    first = SimpleNamespace(label="MATCH")
    last = SimpleNamespace(label="MATCH")
    items = [first, SimpleNamespace(other="MATCH"), last]

    assert select_by_name(items, "MATCH", find_last=False, name_attr="label") is first
    assert select_by_name(items, "MATCH", name_attr="label") is last
    assert select_by_name(items, "MISSING", name_attr="label") is None


def test_resolve_folder_handles_empty_existing_and_missing_paths():
    """Resolve empty, existing, and missing paths without creating folders."""
    leaf = FakeFolder("leaf")
    parent = FakeFolder("parent", [leaf])
    root = FakeFolder("root", [parent])

    assert resolve_folder(root, "", "folder_name") is root
    assert resolve_folder(root, "/parent//leaf/", "folder_name") is leaf
    assert resolve_folder(root, "parent/missing", "folder_name") is None


def test_resolve_folder_creates_missing_segments():
    """Create missing path segments using the fail-on-conflict policy."""
    conflict_policy = SimpleNamespace(FAIL=object())
    root = FakeFolder("root")

    with (
        patch.object(_resinsight_base, "require_rips") as require_rips,
        patch.object(_resinsight_base, "NameConflictPolicy", conflict_policy),
    ):
        leaf = resolve_folder(root, "parent/leaf", "folder_name", create=True)

    require_rips.assert_called()
    assert leaf.folder_name == "leaf"
    assert root.add_calls == [("parent", conflict_policy.FAIL)]
    assert root.children[0].add_calls == [("leaf", conflict_policy.FAIL)]


def test_resolve_case_returns_case_object_unchanged():
    """Return an already resolved case object unchanged."""
    base = _BaseResInsightDataRW(instance_or_port=None)
    case = SimpleNamespace(name="KEEP")
    assert base.resolve_case(case) is case


@pytest.mark.parametrize("bad", [123, object(), None, SimpleNamespace(name=42)])
def test_resolve_case_rejects_invalid_argument(bad):
    """Reject unsupported case arguments before attempting resolution."""
    base = _BaseResInsightDataRW(instance_or_port=None)
    with pytest.raises(TypeError, match="case must be a case name"):
        base.resolve_case(bad)


@pytest.mark.parametrize("good", ["MYCASE", SimpleNamespace(name="FC")])
def test_validate_case_accepts_valid_argument(good):
    """Accept a case name or a case-like object with a string name."""
    assert validate_case(good) is None


@pytest.mark.parametrize("bad", [123, object(), None, SimpleNamespace(name=42)])
def test_validate_case_rejects_invalid_argument(bad):
    """Reject unsupported case argument forms with a TypeError."""
    with pytest.raises(TypeError, match="case must be a case name"):
        validate_case(bad)


# ---------------------------------------------------------------------------
# Early validation at the public API boundary (before expensive extraction)
# ---------------------------------------------------------------------------


def test_grid_to_resinsight_rejects_invalid_case_early():
    """Reject an invalid grid-export case before expensive extraction."""
    grd = xtgeo.create_box_grid((2, 2, 2))
    with pytest.raises(TypeError, match="case must be a case name"):
        grd.to_resinsight(5000, case=123)


def test_gridproperty_to_resinsight_rejects_invalid_case_early():
    """Reject an invalid property-export case before expensive extraction."""
    gprop = xtgeo.GridProperty(ncol=2, nrow=2, nlay=2, values=np.ones((2, 2, 2)))
    with pytest.raises(TypeError, match="case must be a case name"):
        gprop.to_resinsight(5000, case=123)
