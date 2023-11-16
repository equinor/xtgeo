# -*- coding: utf-8 -*-


from os.path import join

import pytest

import xtgeo
from xtgeo.well import BlockedWells


@pytest.fixture(name="testblockedwells")
def fixture_testblockedwells(testpath):
    well_files = [join(testpath, "wells", "reek", "1", "OP_1.bw")]
    bws = xtgeo.blockedwells_from_files(well_files)
    return bws


def test_blockedwells_empty_is_none():
    assert BlockedWells().wells is None


def test_blockedwells_non_well_setter_errors():
    with pytest.raises(ValueError, match="not valid Well"):
        BlockedWells().wells = [None]


def test_blockedwells_copy_same_names(testblockedwells):
    assert testblockedwells.copy().names == testblockedwells.names


def test_blockedwells_setter_same_names(testblockedwells):
    blockedwells2 = BlockedWells()
    blockedwells2.wells = testblockedwells.wells
    assert blockedwells2.names == testblockedwells.names


def test_well_names(testblockedwells):
    for well in testblockedwells.wells:
        assert well.name in testblockedwells.names
        assert testblockedwells.get_blocked_well(well.name) is well


def test_get_dataframe_allblockedwells(testblockedwells, snapshot, helpers):
    """Get a single dataframe for all blockedwells"""
    snapshot.assert_match(
        helpers.df2csv(testblockedwells.get_dataframe(filled=True).head(50).round()),
        "blockedwells.csv",
    )


def test_quickplot_blockedwells(tmpdir, testblockedwells, generate_plot):
    """Import blockedwells from file to BlockedWells and quick plot."""
    if not generate_plot:
        pytest.skip()
    testblockedwells.quickplot(filename=join(tmpdir, "quickblockedwells.png"))


def test_wellintersections(snapshot, testblockedwells, helpers):
    """Find well crossing"""
    snapshot.assert_match(
        helpers.df2csv(testblockedwells.wellintersections().head(50).round()),
        "blockedwells_intersections.csv",
    )


def test_wellintersections_tvdrange_nowfilter(snapshot, testblockedwells, helpers):
    """Find well crossing using coarser sampling to Fence"""
    testblockedwells.limit_tvd(1300, 1400)
    testblockedwells.downsample(interval=6)

    snapshot.assert_match(
        helpers.df2csv(testblockedwells.wellintersections().head(50).round()),
        "tvdrange_blockedwells_intersections.csv",
    )


def test_wellintersections_tvdrange_wfilter(snapshot, testblockedwells, helpers):
    """Find well crossing using coarser sampling to Fence, with
    wfilter settings.
    """

    wfilter = {
        "parallel": {"xtol": 4.0, "ytol": 4.0, "ztol": 2.0, "itol": 10, "atol": 5.0}
    }

    testblockedwells.limit_tvd(1300, 1400)
    testblockedwells.downsample(interval=6)

    snapshot.assert_match(
        helpers.df2csv(
            testblockedwells.wellintersections(wfilter=wfilter).head(50).round()
        ),
        "filtered_blockedwells_intersections.csv",
    )
