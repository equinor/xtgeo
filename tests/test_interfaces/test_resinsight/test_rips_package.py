"""Tests for _rips_package version gating (no ResInsight required).

These tests patch the version metadata and module-level state
(via ``unittest.mock.patch``) to exercise the error paths in _check_rips_version() and
require_rips() without needing an actual rips installation or ResInsight executable.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from xtgeo.interfaces.resinsight import _rips_package


# Tests for _check_rips_version()
def test_check_rips_version_missing_metadata_raises():
    with (
        patch.object(
            _rips_package, "version", side_effect=PackageNotFoundError("rips")
        ),
        pytest.raises(RuntimeError, match="version metadata is missing"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_invalid_version_string_raises():
    with (
        patch.object(_rips_package, "version", return_value="not-a-version!"),
        pytest.raises(RuntimeError, match="not a valid PEP 440"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_old_version_raises():
    with (
        patch.object(_rips_package, "version", return_value="2020.1"),
        pytest.raises(RuntimeError, match="Please upgrade"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_sufficient_version_passes():
    with patch.object(
        _rips_package, "version", return_value=_rips_package.MIN_RIPS_VERSION
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_newer_version_passes():
    with patch.object(_rips_package, "version", return_value="9999.1"):
        _rips_package._check_rips_version()


# Tests for require_rips()


def test_require_rips_raises_when_rips_is_none():
    with (
        patch.object(_rips_package, "rips", None),
        patch.object(_rips_package, "_rips_import_error", None),
        pytest.raises(RuntimeError, match="not available"),
    ):
        _rips_package.require_rips()


def test_require_rips_raises_with_import_error_message():
    msg = "missing symbols (Case, Instance, Project)"
    with (
        patch.object(_rips_package, "rips", None),
        patch.object(_rips_package, "_rips_import_error", msg),
        pytest.raises(RuntimeError, match="missing symbols"),
    ):
        _rips_package.require_rips()


def test_require_rips_delegates_to_version_check():
    sentinel = object()
    with (
        patch.object(_rips_package, "rips", sentinel),
        patch.object(_rips_package, "version", return_value="2020.1"),
        pytest.raises(RuntimeError, match="Please upgrade"),
    ):
        _rips_package.require_rips()


def test_require_rips_passes_when_rips_available_and_version_ok():
    sentinel = object()
    with (
        patch.object(_rips_package, "rips", sentinel),
        patch.object(
            _rips_package, "version", return_value=_rips_package.MIN_RIPS_VERSION
        ),
    ):
        _rips_package.require_rips()
