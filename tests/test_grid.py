#!/usr/bin/env python -u

import unittest
import os
import sys
import logging

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# set default level
xtg = XTGeoDialog()


# =============================================================================
# Do tests
# =============================================================================
emegfile = '../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff'


class TestGrid(unittest.TestCase):
    """Testing suite for 3D grid geometry"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_import_wrong(self):
        """Importing wrong fformat, etc"""

        self.getlogger('test_import_wrong')

        with self.assertRaises(ValueError):
            g = Grid().from_file(emegfile, fformat='roffdum')
            self.assertEqual(g.nx, 70)

    def test_import_guess(self):
        """Import with guessing fformat"""

        self.getlogger('test_import_guess')

        g = Grid().from_file(emegfile)

        print(g.nx)
        self.assertEqual(g.nx, 70)

    def test_roffbin_import0(self):
        """Import ROFF on the form Grid().from_file and Grid(..)"""

        self.getlogger('test_roffbin_import0')

        g = Grid().from_file(emegfile, fformat="roff")

        print(g)
        self.assertIsInstance(g, Grid)

        g = Grid(emegfile, fformat="roff")

        print(g)
        self.assertIsInstance(g, Grid)

    def test_roffbin_import1(self):

        self.getlogger('test_roffbin_import1')

        self.logger.info('Name is {}'.format(__name__))
        g = Grid()
        self.logger.info("Import roff...")
        g.from_file(emegfile, fformat="roff")

        self.assertEqual(g.nx, 70, 'Grid NX Emerald')
        self.assertEqual(g.nz, 46, 'Grid NZ Emerald')

        # extract ACTNUM parameter as a property instance (a GridProperty)
        act = g.get_actnum()

        self.logger.info('ACTNUM is {}'.format(act))
        self.logger.debug('ACTNUM values are \n{}'.format(act.values[888:999]))

        # get dZ...
        dz = g.get_dz()

        self.logger.info('DZ is {}'.format(act))
        self.logger.info('DZ values are \n{}'.format(dz.values[888:999]))

        dzval = dz.values3d
        # get the value is cell 32 73 1 shall be 2.761
        mydz = float(dzval[31:32, 72:73, 0:1])
        self.assertAlmostEqual(mydz, 2.761, places=3,
                               msg='Grid DZ Emerald')

        # get X Y Z coordinates (as GridProperty objects) in one go
        self.logger.info('Get X Y Z...')
        x, y, z = g.get_xyz(names=['xxx', 'yyy', 'zzz'])

        self.logger.info('X is {}'.format(act))
        self.logger.debug('X values are \n{}'.format(x.values[888:999]))

        self.assertEqual(x.name, 'xxx', 'Name of X coord')
        x.name = 'Xerxes'

        self.logger.info('X name is now {}'.format(x.name))

        self.logger.info('Y is {}'.format(act))
        self.logger.debug('Y values are \n{}'.format(y.values[888:999]))

        # attach some properties to grid
        g.props = [x, y]

        self.logger.info(g.props)
        g.props = [z]

        self.logger.info(g.props)

        g.props.append(x)
        self.logger.info(g.propnames)

        # get the property of name Xerxes
        # myx = g.get_prop_by_name('Xerxes')
        # if  myx != None:
        #     self.logger.info(myx)
        # else:
        #     self.logger.info("Got nothing!")

    def test_eclgrid_import1(self):
        """
        Eclipse GRID import
        """
        self.getlogger('test_eclgrid_import1')

        self.logger.info('Name is {}'.format(__name__))
        g = Grid()
        self.logger.info("Import Eclipse GRID...")
        g.from_file('../xtgeo-testdata/3dgrids/gfb/G1.GRID',
                    fformat="grid")

        self.assertEqual(g.nx, 20, 'Grid NX from Eclipse')
        self.assertEqual(g.ny, 20, 'Grid NY from Eclipse')

    def test_eclgrid_import2(self):
        """
        Eclipse EGRID import
        """
        self.getlogger('test_eclgrid_import2')

        self.logger.info('Name is {}'.format(__name__))
        g = Grid()
        self.logger.info("Import Eclipse GRID...")
        g.from_file('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID',
                    fformat="egrid")

        self.assertEqual(g.nx, 99, 'EGrid NX from Eclipse')
        self.assertEqual(g.ny, 120, 'EGrid NY from Eclipse')
        self.assertEqual(g.nactive, 368004, 'EGrid NTOTAL from Eclipse')
        self.assertEqual(g.ntotal, 558360, 'EGrid NACTIVE from Eclipse')

    def test_eclgrid_import3(self):
        """
        Eclipse GRDECL import and translate
        """

        self.getlogger('test_eclgrid_import3')

        self.logger.info('Name is {}'.format(__name__))

        g = Grid()
        self.logger.info("Import Eclipse GRDECL...")
        g.from_file('../xtgeo-testdata/3dgrids/gfb/g1_comments.grdecl',
                    fformat="grdecl")

        mylist = g.get_geometrics()

        xori1 = mylist[0]

        # translate the coordinates
        g.translate_coordinates(translate=(100, 100, 10), flip=(1, 1, 1))

        mylist = g.get_geometrics()

        xori2 = mylist[0]

        # check if origin is translated 100m in X
        self.assertEqual(xori1 + 100, xori2, 'Translate X distance')

        g.to_file('TMP/g1_translate.roff', fformat="roff_binary")

    def test_simple_io(self):
        """Test various import and export formats"""
        self.getlogger('test_simple_io')

        gg = Grid('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID',
                  fformat="egrid")

        gg.to_file("TMP/gullfaks_test.roff")

    def test_ecl_run(self):
        """Test import an eclrun with dates and export to roff after a diff"""
        self.getlogger('test_ecl_run')

        eclroot = '../xtgeo-testdata/3dgrids/gfb/GULLFAKS'
        dates = [19851001, 20150101]
        rprops = ['PRESSURE', 'SWAT']

        gg = Grid(eclroot, fformat='eclipserun', restartdates=dates,
                  restartprops=rprops)

        # get the property object:
        pres1 = gg.get_prop_by_name('PRESSURE_20150101')
        self.assertAlmostEqual(pres1.values.mean(), 239.505447, places=3)

        pres1.to_file("TMP/pres1.roff")

        pres2 = gg.get_prop_by_name('PRESSURE_19851001')

        if isinstance(pres2, GridProperty):
            print("OK==============================================")

        self.logger.debug(pres1.values)
        self.logger.debug(pres2.values)
        self.logger.debug(pres1)

        pres1.values = pres1.values - pres2.values
        self.logger.debug(pres1.values)
        self.logger.debug(pres1)
        avg = pres1.values.mean()
        # ok checked in RMS:
        self.assertAlmostEqual(avg, -93.046011, places=3)

        pres1.to_file("TMP/pressurediff.roff", name="PRESSUREDIFF")


if __name__ == '__main__':

    unittest.main()
