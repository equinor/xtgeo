"""Testing some simulator results that can be regarded as corner cases

* BENCH_SPE9: Older SPE 9 test case; have subtle differences in e.g. restart files
* IX_HIST: Intersect results

The tests where made on resolving issues #873, #874
"""

import pathlib

import pytest

import xtgeo

SPE9 = pathlib.Path("3dgrids/bench_spe9")
SPE9_ROOT = SPE9 / "BENCH_SPE9"
SPE9_INITS = ["PORO", "PERMX"]
SPE9_RESTARTS = ["PRESSURE", "SWAT", "SOIL"]
SPE9_DATES = [19901028, 19901117]

IXH = pathlib.Path("3dgrids/ix_hist")
IXH_ROOT = IXH / "IX_HIST"
IXH_INITS = ["PORO", "PERMX"]
IXH_RESTARTS = ["PRESSURE", "SWAT", "SOIL"]
IXH_DATES = [19980201, 19980301]


def test_grid_gridprops_spe9(testdata_path):
    """Test BENCH_SPE9, which has restart that does not start with SEQNUM."""
    grd = xtgeo.grid_from_file(
        testdata_path / SPE9_ROOT,
        fformat="eclipserun",
        initprops=SPE9_INITS,
        restartprops=SPE9_RESTARTS,
        restartdates=SPE9_DATES,
    )

    dataframe = grd.get_dataframe()
    assert dataframe.loc[0, "PORO"] == pytest.approx(0.087)
    assert dataframe.loc[8999, "PERMX"] == pytest.approx(47.053421)


def test_grid_gridprops_ixh(testdata_path):
    """Test IX_HIST, which may have issues (as most IX cases...)."""
    grd = xtgeo.grid_from_file(
        testdata_path / IXH_ROOT,
        fformat="eclipserun",
        initprops=IXH_INITS,
        restartprops=IXH_RESTARTS,
        restartdates=IXH_DATES,
    )

    dataframe = grd.get_dataframe()
    assert dataframe.loc[0, "PORO"] == pytest.approx(0.106738)
    assert dataframe.loc[46612, "PERMX"] == pytest.approx(1551.469971)
