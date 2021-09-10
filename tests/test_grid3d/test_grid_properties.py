# -*- coding: utf-8 -*-
"""Testing: test_grid_operations"""


import sys
import warnings

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid, GridProperties

warnings.filterwarnings("ignore")

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


def test_import_init():
    """Import INIT Reek"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["PORO", "PORV"]
    x.from_file(IFILE1, fformat="init", names=names, grid=g)

    # get the object
    poro = x.get_prop_by_name("PORO")
    logger.info("PORO avg {}".format(poro.values.mean()))

    porv = x.get_prop_by_name("PORV")
    logger.info("PORV avg {}".format(porv.values.mean()))
    assert poro.values.mean() == pytest.approx(0.1677402, abs=0.00001)


def test_gridproperties_iter():
    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    gps = GridProperties()
    gps.from_file(IFILE1, fformat="init", names=["PORO", "PORV"], grid=g)

    count = 0
    for _ in gps:
        for _ in gps:
            count += 1

    assert count == 4


def test_import_should_fail():
    """Import INIT and UNRST Reek but ask for wrong name or date"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["PORO", "NOSUCHNAME"]
    with pytest.raises(ValueError) as e_info:
        logger.warning(e_info)
        x.from_file(IFILE1, fformat="init", names=names, grid=g)

    rx = GridProperties()
    names = ["PRESSURE"]
    dates = [19991201, 19991212]  # last date does not exist

    rx.from_file(
        RFILE1, fformat="unrst", names=names, dates=dates, grid=g, strict=(True, False)
    )

    with pytest.raises(ValueError) as e_info:
        rx.from_file(
            RFILE1,
            fformat="unrst",
            names=names,
            dates=dates,
            grid=g,
            strict=(True, True),
        )


def test_import_should_pass():
    """Import INIT and UNRST but ask for wrong name or date , using strict=False"""
    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    rx = GridProperties()
    names = ["PRESSURE", "DUMMY"]  # dummy should exist
    dates = [19991201, 19991212]  # last date does not exist

    rx.from_file(
        RFILE1, fformat="unrst", names=names, dates=dates, grid=g, strict=(False, False)
    )

    assert "PRESSURE_19991201" in rx
    assert "PRESSURE_19991212" not in rx
    assert "DUMMY_19991201" not in rx


def test_import_restart():
    """Import Restart"""

    g = xtgeo.grid_from_file(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["PRESSURE", "SWAT"]
    dates = [19991201, 20010101]
    x.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=g)

    # get the object
    pr = x.get_prop_by_name("PRESSURE_19991201")

    swat = x.get_prop_by_name("SWAT_19991201")

    logger.info(x.names)

    logger.info(swat.values3d.mean())
    logger.info(pr.values3d.mean())

    txt = "Average PRESSURE_19991201"
    assert pr.values.mean() == pytest.approx(334.52327, abs=0.0001), txt

    txt = "Average SWAT_19991201"
    assert swat.values.mean() == pytest.approx(0.87, abs=0.01), txt

    pr = x.get_prop_by_name("PRESSURE_20010101")
    logger.info(pr.values3d.mean())
    txt = "Average PRESSURE_20010101"
    assert pr.values.mean() == pytest.approx(304.897, abs=0.01), txt


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


#    assert df.loc[12, 'KEYWORD'] == 'SWAT'


def test_get_dataframe():
    """Get a Pandas dataframe from the gridproperties"""

    g = Grid(GFILE1, fformat="egrid")

    x = GridProperties()

    names = ["SOIL", "SWAT", "PRESSURE"]
    dates = [19991201]
    x.from_file(RFILE1, fformat="unrst", names=names, dates=dates, grid=g)
    df = x.dataframe(activeonly=True, ijk=True, xyz=False)

    print(df.head())

    assert df["SWAT_19991201"].mean() == pytest.approx(0.87802, abs=0.001)
    assert df["PRESSURE_19991201"].mean() == pytest.approx(334.523, abs=0.005)


#    df = x.dataframe(activeonly=True, ijk=True, xyz=True)
