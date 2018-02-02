import sys
import warnings

import pytest

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperties
from xtgeo.common import XTGeoDialog

warnings.filterwarnings("ignore")

xtg = XTGeoDialog()

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

logger = xtg.basiclogger(__name__)

gfile1 = '../xtgeo-testdata/3dgrids/reek/REEK.EGRID'
ifile1 = '../xtgeo-testdata/3dgrids/reek/REEK.INIT'
rfile1 = '../xtgeo-testdata/3dgrids/reek/REEK.UNRST'

apiver = 2


def test_import_init():
    """Import INIT Reek"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PORO', 'PORV']
    x.from_file(ifile1, fformat="init",
                names=names, grid=g, apiversion=apiver)

    # get the object
    poro = x.get_prop_by_name('PORO')
    logger.info("PORO avg {}".format(poro.values.mean()))

    porv = x.get_prop_by_name('PORV')
    logger.info("PORV avg {}".format(porv.values.mean()))
    assert poro.values.mean() == pytest.approx(0.1677402, abs=0.00001)


def test_import_should_fail():
    """Import INIT and UNRST Reek but ask for wrong name or date"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PORO', 'NOSUCHNAME']
    with pytest.raises(ValueError) as e_info:
        logger.warning(e_info)
        x.from_file(ifile1, fformat="init", names=names, grid=g,
                    apiversion=apiver)

    rx = GridProperties()
    names = ['PRESSURE']
    dates = [19991201, 19991212]  # last date does not exist

    with pytest.warns(RuntimeWarning) as e_info:
        logger.warning(e_info)
        rx.from_file(rfile1, fformat='unrst', names=names, dates=dates, grid=g,
                     apiversion=apiver)


def test_import_should_warn():
    """Import INIT and UNRST Reek but ask for wrong name or date"""
    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    rx = GridProperties()
    names = ['PRESSURE']
    dates = [19991201, 19991212]  # last date does not exist

    rx.from_file(rfile1, fformat='unrst', names=names, dates=dates, grid=g,
                 apiversion=apiver)


def test_import_restart():
    """Import Restart"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PRESSURE', 'SWAT']
    dates = [19991201, 20010101]
    x.from_file(rfile1,
                fformat="unrst", names=names, dates=dates,
                grid=g, apiversion=apiver)

    # get the object
    pr = x.get_prop_by_name('PRESSURE_19991201')

    swat = x.get_prop_by_name('SWAT_19991201')

    logger.info(x.names)

    logger.info(swat.values3d.mean())
    logger.info(pr.values3d.mean())

    txt = 'Average PRESSURE_19991201'
    assert pr.values.mean() == pytest.approx(334.52327, abs=0.0001), txt

    txt = 'Average SWAT_19991201'
    assert swat.values.mean() == pytest.approx(0.87, abs=0.01), txt

    pr = x.get_prop_by_name('PRESSURE_20010101')
    logger.info(pr.values3d.mean())
    txt = 'Average PRESSURE_20010101'
    assert pr.values.mean() == pytest.approx(304.897, abs=0.01), txt


def test_import_restart_gull():
    """Import Restart Reek"""

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['PRESSURE', 'SWAT']
    dates = [19991201]
    x.from_file(rfile1,
                fformat="unrst", names=names, dates=dates,
                grid=g, apiversion=apiver)

    # get the object
    pr = x.get_prop_by_name('PRESSURE_19991201')

    swat = x.get_prop_by_name('SWAT_19991201')

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

    g = Grid()
    g.from_file(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['SOIL', 'SWAT', 'PRESSURE']
    dates = [19991201]
    x.from_file(rfile1, fformat="unrst", names=names, dates=dates, grid=g,
                apiversion=apiver)

    logger.info(x.names)

    # get the object instance
    soil = x.get_prop_by_name('SOIL_19991201')
    logger.info(soil.values3d.mean())

    logger.debug(x.names)
    txt = 'Average SOIL_19850101'
    assert soil.values.mean() == pytest.approx(0.121977, abs=0.001), txt


def test_scan_dates():
    """A static method to scan dates in a RESTART file"""
    t1 = xtg.timer()
    dl = GridProperties.scan_dates(rfile1)  # no need to make instance
    t2 = xtg.timer(t1)
    print('Dates scanned in {} seconds'.format(t2))

    assert dl[2][1] == 20000201


def test_scan_keywords():
    """A static method to scan quickly keywords in a RESTART/INIT/*GRID file"""
    t1 = xtg.timer()
    df = GridProperties.scan_keywords(rfile1, dataframe=True)
    t2 = xtg.timer(t1)
    logger.info('Dates scanned in {} seconds'.format(t2))
    logger.info(df)

    assert df.loc[12, 'KEYWORD'] == 'SWAT'


def test_get_dataframe():
    """Get a Pandas dataframe from the gridproperties"""

    g = Grid(gfile1, fformat="egrid")

    x = GridProperties()

    names = ['SOIL', 'SWAT', 'PRESSURE']
    dates = [19991201]
    x.from_file(rfile1, fformat="unrst", names=names, dates=dates, grid=g,
                apiversion=apiver)
    df = x.dataframe(activeonly=False, ijk=True, xyz=True)
    print(repr(df))
    print(df.dtypes)
