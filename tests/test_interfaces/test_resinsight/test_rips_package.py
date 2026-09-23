"""Tests for _rips_package version gating (no ResInsight required).

These tests patch the version metadata and module-level state
(via ``unittest.mock.patch``) to exercise the error paths in _check_rips_version() and
require_rips() without needing an actual rips installation or ResInsight executable.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from xtgeo.interfaces.resinsight import _rips_package


def test_load_package_returns_available_module():
    """Return the requested package when it can be imported."""
    package = ModuleType("rips")
    with patch.object(_rips_package.importlib, "import_module", return_value=package):
        assert _rips_package._load_package("rips") is package


def test_load_package_returns_none_when_package_is_missing():
    """Treat only the requested package being absent as optional."""
    error = ModuleNotFoundError("No module named 'rips'", name="rips")
    with patch.object(_rips_package.importlib, "import_module", side_effect=error):
        assert _rips_package._load_package("rips") is None


def test_load_package_propagates_missing_transitive_dependency():
    """Expose a missing dependency from inside an installed package."""
    error = ModuleNotFoundError("No module named 'grpc'", name="grpc")
    with (
        patch.object(_rips_package.importlib, "import_module", side_effect=error),
        pytest.raises(ModuleNotFoundError, match="grpc"),
    ):
        _rips_package._load_package("rips")


def test_load_package_propagates_internal_import_error():
    """Expose other import failures raised by an installed package."""
    error = ImportError("cannot import name 'Client' from 'grpc'")
    with (
        patch.object(_rips_package.importlib, "import_module", side_effect=error),
        pytest.raises(ImportError, match="cannot import name 'Client'"),
    ):
        _rips_package._load_package("rips")


def test_load_rips_api_returns_required_symbols():
    """Return every required symbol from a complete rips-like module."""
    symbols = {name: object() for name in _rips_package._RIPS_API_SYMBOLS}

    assert _rips_package._load_rips_api(SimpleNamespace(**symbols)) == symbols


def test_initialize_rips_api_handles_unavailable_package():
    """Represent an unavailable package with empty, error-free state."""
    assert _rips_package._initialize_rips_api(None) == (None, None, {})


def test_initialize_rips_api_accepts_complete_package():
    """Accept a package that exposes the complete API required by xtgeo."""
    symbols = {name: object() for name in _rips_package._RIPS_API_SYMBOLS}
    package = SimpleNamespace(**symbols)

    assert _rips_package._initialize_rips_api(package) == (package, None, symbols)


@pytest.mark.parametrize("missing_symbol", ["RegularSurface", "RipsError"])
def test_initialize_rips_api_rejects_missing_symbol(missing_symbol):
    """Reject an incomplete package and identify the missing API symbol."""
    package = SimpleNamespace(
        **{
            name: object()
            for name in _rips_package._RIPS_API_SYMBOLS
            if name != missing_symbol
        }
    )

    loaded, error, symbols = _rips_package._initialize_rips_api(package)

    assert loaded is None
    assert missing_symbol in error
    assert symbols == {}


def test_initialize_rips_api_generates_required_symbol_list():
    """Keep the incompatibility message synchronized with the API contract."""
    with patch.object(_rips_package, "_RIPS_API_SYMBOLS", ("First", "Second")):
        _, error, _ = _rips_package._initialize_rips_api(SimpleNamespace())

    assert "required API symbols (First, Second)" in error


def test_check_rips_version_missing_metadata_raises():
    """Report missing version metadata as an actionable RuntimeError."""
    with (
        patch.object(
            _rips_package, "version", side_effect=PackageNotFoundError("rips")
        ),
        pytest.raises(RuntimeError, match="version metadata is missing"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_invalid_version_string_raises():
    """Reject rips metadata that is not a valid PEP 440 version."""
    with (
        patch.object(_rips_package, "version", return_value="not-a-version!"),
        pytest.raises(RuntimeError, match="not a valid PEP 440"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_old_version_raises():
    """Reject a valid rips version below the minimum supported release."""
    with (
        patch.object(_rips_package, "version", return_value="2020.1"),
        pytest.raises(RuntimeError, match="Please upgrade"),
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_sufficient_version_passes():
    """Accept a rips version exactly equal to the minimum requirement."""
    with patch.object(
        _rips_package, "version", return_value=_rips_package.MIN_RIPS_VERSION
    ):
        _rips_package._check_rips_version()


def test_check_rips_version_newer_version_passes():
    """Accept a valid rips version newer than the minimum requirement."""
    with patch.object(_rips_package, "version", return_value="9999.1"):
        _rips_package._check_rips_version()


def test_require_rips_raises_when_rips_is_none():
    """Report standard installation guidance when rips is unavailable."""
    with (
        patch.object(_rips_package, "rips", None),
        patch.object(_rips_package, "_rips_import_error", None),
        pytest.raises(RuntimeError, match="not available"),
    ):
        _rips_package.require_rips()


def test_require_rips_raises_with_import_error_message():
    """Preserve the initialization error for an incompatible package."""
    msg = "missing symbols (Case, Instance, Project)"
    with (
        patch.object(_rips_package, "rips", None),
        patch.object(_rips_package, "_rips_import_error", msg),
        pytest.raises(RuntimeError, match="missing symbols"),
    ):
        _rips_package.require_rips()


def test_require_rips_delegates_to_version_check():
    """Validate the version even when the rips module is already loaded."""
    sentinel = object()
    with (
        patch.object(_rips_package, "rips", sentinel),
        patch.object(_rips_package, "version", return_value="2020.1"),
        pytest.raises(RuntimeError, match="Please upgrade"),
    ):
        _rips_package.require_rips()


def test_require_rips_passes_when_rips_available_and_version_ok():
    """Allow a loaded rips module that meets the version requirement."""
    sentinel = object()
    with (
        patch.object(_rips_package, "rips", sentinel),
        patch.object(
            _rips_package, "version", return_value=_rips_package.MIN_RIPS_VERSION
        ),
    ):
        assert _rips_package.require_rips() is sentinel
