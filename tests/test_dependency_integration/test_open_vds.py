from pathlib import Path

import pytest

import xtgeo


def test_openvds_works_in_same_environment(testdata_path: Path) -> None:
    """Tests that OpenVDS installed in the same environment causes no issues."""
    openvds = pytest.importorskip("openvds")
    openvds
    return xtgeo.grid_from_file(
        testdata_path / "3dgrids/eme/1/emerald_hetero_grid.roff"
    )
    # Does not segfault
