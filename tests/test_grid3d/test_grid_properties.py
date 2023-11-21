# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""


import io
import sys

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import GridProperties, GridProperty

from .grid_generator import xtgeo_grids
from .gridprop_generator import grid_properties as gridproperties_elements
from .gridprop_generator import keywords

xtg = XTGeoDialog()

if not xtg.testsetup():
    sys.exit(-9)

TPATH = xtg.testpathobj

logger = xtg.basiclogger(__name__)

GFILE1 = TPATH / "3dgrids/reek/REEK.EGRID"
IFILE1 = TPATH / "3dgrids/reek/REEK.INIT"
RFILE1 = TPATH / "3dgrids/reek/REEK.UNRST"
RFILE2 = TPATH / "3dgrids/simpleb8/E100_3LETTER_TRACER.UNRST"  # has kword with spaces

XFILE2 = TPATH / "3dgrids/reek/reek_grd_w_props.roff"

# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name


@st.composite
def gridproperties(draw):
    grid = draw(xtgeo_grids)
    names = draw(st.lists(keywords, unique=True))
    return GridProperties(
        props=[
            draw(gridproperties_elements(name=st.just(n), grid=st.just(grid)))
            for n in names
        ]
    )


@given(gridproperties(), st.text())
def test_gridproperties_get_prop_by_name_not_exists(gps, name):
    assume(name not in gps.names)

    assert gps.get_prop_by_name(name, raiseserror=False) is None

    with pytest.raises(ValueError, match="Cannot find"):
        gps.get_prop_by_name(name)

    with pytest.raises(KeyError, match="does not exist"):
        gps[name]


def test_gridproperties_import_date_does_not_exist(tmp_path):
    tmpfile = tmp_path / "TEST.UNRST"
    tmpfile.write_text("")
    with pytest.raises(ValueError, match="dates are either"):
        xtgeo.gridproperties_from_file(
            tmpfile,
            names="all",
            dates=["12.12.1"],
            fformat="unrst",
            grid=xtgeo.create_box_grid((2, 2, 2)),
        )


def test_gridproperties_iter():
    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    gps = xtgeo.gridproperties_from_file(
        IFILE1, fformat="init", names=["PORO", "PORV"], grid=g
    )

    count = 0
    for _ in gps:
        for _ in gps:
            count += 1

    assert count == 4


@given(gridproperties())
def test_gridproperties_copy(grid_properties):
    grid_properties_copy = grid_properties.copy()

    assert grid_properties_copy.dates == grid_properties.dates
    assert grid_properties_copy.names == grid_properties.names

    assume(grid_properties.props)
    assume(grid_properties.props[0].date != 19990101)
    grid_properties_copy.props[0].date = 19990101
    assert grid_properties.props[0].date != 19990101


@given(gridproperties_elements(), gridproperties_elements())
def test_consistency_check(gridproperty1, gridproperty2):
    assume(gridproperty1.dimensions != gridproperty2.dimensions)
    with pytest.raises(ValueError, match="Mismatching dimensions"):
        GridProperties(props=[gridproperty1, gridproperty2])


@given(gridproperties_elements())
def test_gridproperties_from_roff(grid_property):
    buff = io.BytesIO()
    grid_property.to_file(buff, fformat="roff")
    buff.seek(0)
    props = xtgeo.gridproperties_from_file(
        buff, fformat="roff", names=[grid_property.name]
    )

    assert props.names == [grid_property.name]


@given(gridproperties_elements())
def test_gridproperties_invalid_format(grid_property):
    buff = io.BytesIO()
    grid_property.to_file(buff, fformat="roff")
    with pytest.raises(ValueError, match="Invalid file format"):
        xtgeo.gridproperties_from_file(buff, fformat="segy")


def test_scan_dates():
    """A static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(RFILE1)  # no need to make instance
    t2 = xtg.timer(t1)
    print(f"Dates scanned in {t2} seconds")

    assert dl[2][1] == 20000201


def test_scan_dates_invalid_file():
    """Raise an error before trying to scan a non-existent file."""
    with pytest.raises(ValueError, match="does not exist"):
        GridProperties.scan_dates(TPATH / "notafile.UNRST")


def test_dates_from_restart():
    """A simpler static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(RFILE1, datesonly=True)  # no need to make instance
    t2 = xtg.timer(t1)
    print(f"Dates scanned in {t2} seconds")

    assert dl[4] == 20000401


def test_scan_keywords():
    """A static method to scan quickly keywords in a RESTART/INIT/*GRID file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(RFILE1, dataframe=True)
    t2 = xtg.timer(t1)
    logger.info("Dates scanned in %s seconds", t2)
    logger.info(df)

    assert df.loc[12, "KEYWORD"] == "SWAT"  # pylint: disable=no-member


def test_scan_ecl_keywords_with_spaces():
    """Allow and preserve spacing in keywords from Eclipse RESTART file"""
    df = GridProperties.scan_keywords(RFILE2, dataframe=True)

    assert df.loc[12, "KEYWORD"] == "W2 F"  # pylint: disable=no-member


def test_scan_keywords_invalid_file():
    """Raise an error before trying to scan a non-existent file."""
    with pytest.raises(ValueError, match="does not exist"):
        GridProperties.scan_keywords(TPATH / "notafile.UNRST")


def test_scan_keywords_roff_as_tuple_list():
    """A static method to scan quickly keywords in a ROFF file"""
    t1 = xtg.timer()
    keywords = GridProperties.scan_keywords(XFILE2, fformat="roff")
    t2 = xtg.timer(t1)
    logger.info("Keywords scanned in %s seconds", t2)
    assert keywords[0] == ("filedata!byteswaptest", "int", 1, 111)
    assert keywords[-1] == ("parameter!data", "int", 35840, 806994)
    logger.info(keywords)


def test_scan_keywords_roff():
    """A static method to scan quickly keywords in a ROFF file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(XFILE2, dataframe=True, fformat="roff")
    t2 = xtg.timer(t1)
    logger.info("Keywords scanned in %s seconds", t2)
    assert tuple(df.iloc[0]) == ("filedata!byteswaptest", "int", 1, 111)
    assert tuple(df.iloc[-1]) == ("parameter!data", "int", 35840, 806994)
    logger.info(df)


def test_get_dataframe():
    """Get a Pandas dataframe from the gridproperties"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x = xtgeo.gridproperties_from_file(
        RFILE1, fformat="unrst", names=names, dates=dates, grid=g
    )
    df = x.get_dataframe(activeonly=True, ijk=True, xyz=False)

    print(df.head())

    assert df["SWAT_19991201"].mean() == pytest.approx(0.87802, abs=0.001)
    assert df["PRESSURE_19991201"].mean() == pytest.approx(334.523, abs=0.005)


def test_get_dataframe_active_only():
    """Get a Pandas dataframe from the gridproperties"""

    grid = xtgeo.grid_from_file(GFILE1, fformat="egrid")
    gps = xtgeo.gridproperties_from_file(
        RFILE1,
        fformat="unrst",
        names=["SOIL", "SWAT", "PRESSURE"],
        dates=[19991201],
        grid=grid,
    )

    df = gps.get_dataframe(activeonly=True, ijk=True, xyz=False)
    assert len(df.index == grid.nactive)

    df2 = xtgeo.gridproperties_dataframe(gps, activeonly=True, ijk=True, xyz=False)
    assert (df == df2).all().all()


def test_gridproperties_all_roff():
    """Read all gridproperties from ROFF binary format."""

    gps = xtgeo.gridproperties_from_file(
        XFILE2,
        fformat="roff",
        names="all",
    )

    for name in ("PORV", "PORO", "EQLNUM", "FIPNUM"):
        assert name in gps.names


def test_gridproperties_read_roff_missing_name():
    """Read gridproperties from ROFF binary format, with one key not present."""

    with pytest.raises(ValueError):
        gps = xtgeo.gridproperties_from_file(
            XFILE2,
            fformat="roff",
            names=["PORO", "EQLNUM", "NOTPRESENT"],
            strict=True,
        )

    gps = xtgeo.gridproperties_from_file(
        XFILE2,
        fformat="roff",
        names=["PORO", "EQLNUM", "NOTPRESENT"],
        strict=False,
    )

    assert "PORO" in gps.names
    assert "NOTPRESENT" not in gps.names


@given(gridproperties())
def test_get_dataframe_no_grid(gridproperties):
    with pytest.raises(ValueError, match="no Grid is present"):
        xtgeo.gridproperties_dataframe(gridproperties, ijk=True, activeonly=False)

    with pytest.raises(ValueError, match="no Grid is present"):
        xtgeo.gridproperties_dataframe(
            gridproperties, ijk=True, activeonly=True, xyz=True
        )


@given(gridproperties())
def test_get_dataframe_filled(gridproperties):
    gridproperties_list = list(gridproperties)
    assume(len(gridproperties_list) > 0)
    df = xtgeo.gridproperties_dataframe(gridproperties, ijk=False, activeonly=False)
    assert (
        len(df.index) == gridproperties.ncol * gridproperties.nrow * gridproperties.nlay
    )


def test_props_set_get() -> None:
    gp = GridProperties()
    assert gp.props is None

    props = [GridProperty()]
    gp = GridProperties()
    gp.props = props
    assert gp.props == props

    props = [GridProperty()]
    gp = GridProperties(props=props)
    assert gp.props == props
