"""Tests integration with dependencies that may have conflicts.

These conflicts are typically a result of compiled dependencies."""

from pathlib import Path

import pytest


def test_openvds_works_in_same_environment(testdata_path: str) -> None:
    """Tests that OpenVDS installed in the same environment causes no issues.

    This previous caused segfaults on Window due to OpenMP not being statically
    compiled."""
    openvds = pytest.importorskip("openvds")
    openvds
    import xtgeo

    xtgeo.grid_from_file(Path(testdata_path) / "3dgrids/eme/1/emerald_hetero_grid.roff")
    # Does not segfault
