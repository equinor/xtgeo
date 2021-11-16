# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""


import sys

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import GridProperties

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


def test_gridproperties_iter():
    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    gps = GridProperties()
    gps.from_file(IFILE1, fformat="init", names=["PORO", "PORV"], grid=g)

    count = 0
    for _ in gps:
        for _ in gps:
            count += 1

    assert count == 4


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


#    assert df.loc[12, 'KEYWORD'] == 'SWAT'


def test_get_dataframe():
    """Get a Pandas dataframe from the gridproperties"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=g)
    df = x.dataframe(activeonly=True, ijk=True, xyz=False)

    print(df.head())

    assert df["SWAT_19991201"].mean() == pytest.approx(0.87802, abs=0.001)
    assert df["PRESSURE_19991201"].mean() == pytest.approx(334.523, abs=0.005)


#    df = x.dataframe(activeonly=True, ijk=True, xyz=True)
