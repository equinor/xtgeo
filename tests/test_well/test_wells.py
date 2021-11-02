# -*- coding: utf-8 -*-


from os.path import join

import pytest

from xtgeo.common import XTGeoDialog
from xtgeo.well import Wells

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit


@pytest.fixture(name="testwells")
def fixture_testwells(testpath):
    w_names = [
        "WELL29",
        "WELL14",
        "WELL30",
        "WELL27",
        "WELL23",
        "WELL32",
        "WELL22",
        "WELL35",
        "WELLX",
    ]
    well_files = [
        join(testpath, "wells", "battle", "1", wn + ".rmswell") for wn in w_names
    ]
    return Wells(well_files, fformat="rms_ascii")


def test_wells_empty_is_none():
    assert Wells().wells is None


def test_wells_non_well_setter_errors():
    with pytest.raises(ValueError, match="not valid Well"):
        Wells().wells = [None]


def test_wells_copy_same_names(testwells):
    assert testwells.copy().names == testwells.names


def test_wells_setter_same_names(testwells):
    wells2 = Wells()
    wells2.wells = testwells.wells
    assert wells2.names == testwells.names


def test_well_names(testwells):
    for well in testwells.wells:
        assert well.name in testwells.names
        assert testwells.get_well(well.name) is well


def test_get_dataframe_allwells(testwells, snapshot):
    """Get a single dataframe for all wells"""
    snapshot.assert_match(
        testwells.get_dataframe(filled=True)
        .head(50)
        .round()
        .to_csv(line_terminator="\n"),
        "wells.csv",
    )


def test_quickplot_wells(tmpdir, testwells, generate_plot):
    """Import wells from file to Wells and quick plot."""
    if not generate_plot:
        pytest.skip()
    testwells.quickplot(filename=join(tmpdir, "quickwells.png"))


def test_wellintersections(tmpdir, testwells, snapshot):
    """Find well crossing"""
    snapshot.assert_match(
        testwells.wellintersections().head(50).round().to_csv(line_terminator="\n"),
        "wellintersections.csv",
    )


def test_wellintersections_tvdrange_nowfilter(tmpdir, snapshot, testwells):
    """Find well crossing using coarser sampling to Fence"""
    testwells.limit_tvd(1300, 1400)
    testwells.downsample(interval=6)

    snapshot.assert_match(
        testwells.wellintersections().head(50).round().to_csv(line_terminator="\n"),
        "tvd_wellintersections.csv",
    )


def test_wellintersections_tvdrange_wfilter(tmpdir, snapshot, testwells):
    """Find well crossing using coarser sampling to Fence, with
    wfilter settings.
    """

    wfilter = {
        "parallel": {"xtol": 4.0, "ytol": 4.0, "ztol": 2.0, "itol": 10, "atol": 5.0}
    }

    testwells.limit_tvd(1300, 1400)
    testwells.downsample(interval=6)

    snapshot.assert_match(
        testwells.wellintersections(wfilter=wfilter)
        .head(50)
        .round()
        .to_csv(line_terminator="\n"),
        "filtered_wellintersections.csv",
    )
