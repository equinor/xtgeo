import pytest
import os
import sys
import logging

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperties
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# set default level
xtg = XTGeoDialog()

logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)

logger = logging.getLogger(__name__)

gfile1 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID'
ifile1 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.INIT'
rfile1 = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS.UNRST'

gfile2 = '../xtgeo-testdata/3dgrids/gfb/ECLIPSE.EGRID'
ifile2 = '../xtgeo-testdata/3dgrids/gfb/ECLIPSE.INIT'
rfile2 = '../xtgeo-testdata/3dgrids/gfb/ECLIPSE.UNRST'


def test_import_init():
    """Import INIT Gullfaks"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PORO', 'PORV']
    x.from_file(ifile1, fformat="init",
                names=names, grid=g)

    # get the object
    poro = x.get_prop_by_name('PORO')
    logger.info("PORO avg {}".format(poro.values.mean()))

    porv = x.get_prop_by_name('PORV')
    logger.info("PORV avg {}".format(porv.values.mean()))
    assert poro.values.mean() == pytest.approx(0.261157, abs=0.00001)


def test_import_should_fail():
    """Import INIT and UNRST Gullfaks but ask for wrong name or date"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PORO', 'NOSUCHNAME']
    with pytest.raises(ValueError) as e_info:
        logger.warning(e_info)
        x.from_file(ifile1, fformat="init", names=names, grid=g)

    rx = GridProperties()
    names = ['PRESSURE']
    dates = [19851001, 19870799]  # last date does not exist

    with pytest.raises(RuntimeWarning) as e_info:
        logger.warning(e_info)
        rx.from_file(rfile2, fformat='unrst', names=names, dates=dates, grid=g)


def test_import_restart():
    """Import Restart"""

    g = Grid()
    g.from_file(gfile2, fformat="egrid")

    x = GridProperties()

    names = ['PRESSURE', 'SWAT']
    dates = [19851001, 19870701]
    x.from_file(rfile2,
                fformat="unrst", names=names, dates=dates,
                grid=g)

    # get the object
    pr = x.get_prop_by_name('PRESSURE_19851001')

    swat = x.get_prop_by_name('SWAT_19851001')

    logger.info(x.names)

    logger.info(swat.values3d.mean())
    logger.info(pr.values3d.mean())

    txt = 'Average PRESSURE_19851001'
    assert pr.values.mean() == pytest.approx(332.54578, abs=0.0001), txt

    txt = 'Average SWAT_19851001'
    assert swat.values.mean() == pytest.approx(0.87, abs=0.01), txt

    pr = x.get_prop_by_name('PRESSURE_19870701')
    logger.info(pr.values3d.mean())
    txt = 'Average PRESSURE_19870701'
    assert pr.values.mean() == pytest.approx(331.62, abs=0.01), txt


def test_import_restart_gull():
    """Import Restart Gullfaks"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PRESSURE', 'SWAT']
    dates = [19851001]
    x.from_file(rfile1,
                fformat="unrst", names=names, dates=dates,
                grid=g)

    # get the object
    pr = x.get_prop_by_name('PRESSURE_19851001')

    swat = x.get_prop_by_name('SWAT_19851001')

    logger.info(x.names)

    logger.info(swat.values3d.mean())
    logger.info(pr.values3d.mean())

    # .assertAlmostEqual(pr.values.mean(), 332.54578,
    #                        places=4, msg='Average PRESSURE_19851001')
    # .assertAlmostEqual(swat.values.mean(), 0.87,
    #                        places=2, msg='Average SWAT_19851001')

    # pr = x.get_prop_by_name('PRESSURE_19870701')
    # logger.info(pr.values3d.mean())
    # .assertAlmostEqual(pr.values.mean(), 331.62,
    #                        places=2, msg='Average PRESSURE_19870701')


def test_import_soil():
    """SOIL need to be computed in code from SWAT and SGAS"""

    g = Grid()
    g.from_file(gfile2, fformat="egrid")

    x = GridProperties()

    names = ['SOIL']
    dates = [19851001]
    x.from_file(rfile2, fformat="unrst", names=names, dates=dates, grid=g)

    # get the object instance
    soil = x.get_prop_by_name('SOIL_19851001')
    logger.info(soil.values3d.mean())

    logger.debug(x.names)
    txt = 'Average SOIL_19850101'
    assert soil.values.mean() == pytest.approx(0.1246, abs=0.001), txt


def test_scan_dates():
    """A static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(rfile2)  # no need to make instance
    t2 = xtg.timer(t1)
    print('Dates scanned in {} seconds'.format(t2))

    assert dl[2][1] == 19870101


def test_scan_keywords():
    """A static method to scan quickly keywords in a RESTART/INIT/*GRID file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(rfile2, dataframe=True)
    t2 = xtg.timer(t1)
    logger.info('Dates scanned in {} seconds'.format(t2))
    logger.info(df)

    assert df.loc[19, 'KEYWORD'] == 'ICON'
