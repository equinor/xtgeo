# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""


import sys

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import GridProperties

from .ecl_run_fixtures import *  # noqa: F401, F403
from .gridprop_generator import grid_properties as gridproperties_elements

xtg = XTGeoDialog()

if not xtg.testsetup():
    sys.exit(-9)

TPATH = xtg.testpathobj

logger = xtg.basiclogger(__name__)

GFILE1 = TPATH / "3dgrids/reek/REEK.EGRID"
IFILE1 = TPATH / "3dgrids/reek/REEK.INIT"
RFILE1 = TPATH / "3dgrids/reek/REEK.UNRST"

XFILE2 = TPATH / "3dgrids/reek/reek_grd_w_props.roff"

# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name


@st.composite
def gridproperties(draw):
    gps = GridProperties()
    gps._props = []
    gps.append_props(draw(st.lists(elements=gridproperties_elements())))
    return gps


@given(gridproperties(), st.text())
def test_gridproperties_get_prop_by_name_not_exists(gps, name):
    assume(name not in gps.names)

    assert gps.get_prop_by_name(name, raiseserror=False) is None

    with pytest.raises(ValueError, match="Cannot find"):
        gps.get_prop_by_name(name)

    with pytest.raises(KeyError, match="does not exist"):
        gps[name]


@given(gridproperties())
def test_gridproperties_iter(gps):
    count = 0
    for _ in gps:
        for _ in gps:
            count += 1

    if gps.props is None:
        num_props = 0
    else:
        num_props = len(gps.props)

    assert count == num_props ** 2


def test_restart_name_style():
    grid = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    names = ["PRESSURE", "SWAT"]
    dates = [19991201, 20010101]

    gps = GridProperties()
    gps.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=grid)
    assert gps.names == [p.name for p in gps]

    gps2 = GridProperties()
    gps2.from_file(
        RFILE1, fformat="unrst", names=names, dates=dates, grid=grid, namestyle=1
    )
    assert gps2.names == [p.name for p in gps2]


def test_import_restart_gull():
    """Import Restart Reek"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["PRESSURE", "SWAT"]
    dates = [19991201]
    x.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=g)

    # get the object
    pr = x.get_prop_by_name("PRESSURE_19991201")

    swat = x.get_prop_by_name("SWAT_19991201")

    logger.info(x.names)

    logger.info(swat.values3d.mean())
    logger.info(pr.values3d.mean())

    # .assertAlmostEqual(pr.values.mean(), 332.54578,
    #                        places=4, msg='Average PRESSURE_19991201')
    # .assertAlmostEqual(swat.values.mean(), 0.87,
    #                        places=2, msg='Average SWAT_19991201')

    # pr = x.get_prop_by_name('PRESSURE_20010101')
    # logger.info(pr.values3d.mean())
    # .assertAlmostEqual(pr.values.mean(), 331.62,
    #                        places=2, msg='Average PRESSURE_20010101')


def test_import_soil():
    """SOIL need to be computed in code from SWAT and SGAS"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=g)

    logger.info(x.names)

    # get the object instance
    soil = x.get_prop_by_name("SOIL_19991201")
    logger.info(soil.values3d.mean())

    logger.debug(x.names)
    txt = "Average SOIL_19850101"
    assert soil.values.mean() == pytest.approx(0.121977, abs=0.001), txt


def test_scan_dates():
    """A static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(RFILE1)  # no need to make instance
    t2 = xtg.timer(t1)
    print("Dates scanned in {} seconds".format(t2))

    assert dl[2][1] == 20000201


def test_dates_from_restart():
    """A simpler static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(RFILE1, datesonly=True)  # no need to make instance
    t2 = xtg.timer(t1)
    print("Dates scanned in {} seconds".format(t2))

    assert dl[4] == 20000401


def test_scan_keywords():
    """A static method to scan quickly keywords in a RESTART/INIT/*GRID file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(RFILE1, dataframe=True)
    t2 = xtg.timer(t1)
    logger.info("Dates scanned in {} seconds".format(t2))
    logger.info(df)

    assert df.loc[12, "KEYWORD"] == "SWAT"  # pylint: disable=no-member


def test_scan_keywords_roff():
    """A static method to scan quickly keywords in a ROFF file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(XFILE2, dataframe=True, fformat="roff")
    t2 = xtg.timer(t1)
    logger.info("Dates scanned in {} seconds".format(t2))
    logger.info(df)
