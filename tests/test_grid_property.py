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


class TestGridProperty(unittest.TestCase):
    """Testing suite for 3D grid properties"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_create(self):
        x = GridProperty()
        self.assertEqual(x.nx, 5, 'NX is OK')
        self.assertEqual(x.ny, 12, 'NY')

        # print(repr(x.values))

        m = GridProperty(discrete=True)
        (repr(m.values))
        # print(m.values.dtype)

    def test_roffbin_import1(self):
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        x = GridProperty()
        self.logger.info("Import roff...")
        x.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2_poro.roffbin',
                    fformat="roff", name='PORO')

        self.logger.info(repr(x.values))
        self.logger.info(x.values.dtype)
        self.logger.info("Mean porosity is {}".format(x.values.mean()))
        self.assertAlmostEqual(x.values.mean(), 0.262558836873,
                               places=4, msg='Average porosity')

    def test_roffbin_import2(self):

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        dz = GridProperty()
        self.logger.info("Import roff...")
        dz.from_file('../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff',
                     fformat="roff", name='Z_increment')

        self.logger.info(repr(dz.values))
        self.logger.info(dz.values.dtype)
        self.logger.info("Mean DZ is {}".format(dz.values.mean()))

        hc = GridProperty()
        self.logger.info("Import roff...")
        hc.from_file('../xtgeo-testdata/3dgrids/eme/1/emerald_hetero.roff',
                     fformat="roff", name='Oil_HCPV')

        self.logger.info(repr(hc.values))
        self.logger.info(hc.values.dtype)
        self.logger.info(hc.values3d.shape)
        nx, ny, nz = hc.values3d.shape

        self.assertEqual(ny, 100, 'NY from shape (Emerald)')

        self.logger.info("Mean HCPV is {}".format(hc.values.mean()))

    def test_eclinit_import(self):
        """
        Property import from Eclipse. Needs a grid object first.
        """

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        gg = Grid()
        gg.from_file('../xtgeo-testdata/3dgrids/bri/B.GRID',
                     fformat="grid")
        po = GridProperty()
        self.logger.info("Import INIT...")
        po.from_file('../xtgeo-testdata/3dgrids/bri/B.INIT',
                     fformat="init", name='PORO', grid=gg)

        self.assertEqual(po.nx, 20, 'NX from B.INIT')

        self.logger.debug(po.values[0:400])
        self.assertAlmostEqual(float(po.values3d[1:2, 13:14, 0:1]), 0.17146,
                               places=5, msg='PORO in cell 2 14 1')

        # discrete prop
        eq = GridProperty()
        eq.from_file('../xtgeo-testdata/3dgrids/bri/B.INIT',
                     fformat="init", name='EQLNUM', grid=gg)
        self.logger.info(eq.values[0:400])
        self.assertEqual(eq.values3d[12:13, 13:14, 0:1], 3,
                         'EQLNUM in cell 13 14 1')

    def test_eclinit_import_gull(self):
        """
        Property import from Eclipse. Gullfaks
        """

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        gg = Grid()
        gg.from_file('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.EGRID',
                     fformat="egrid")
        self.assertEqual(gg.nx, 99, "Gullfaks NX")
        po = GridProperty()
        self.logger.info("Import INIT...")
        po.from_file('../xtgeo-testdata/3dgrids/gfb/GULLFAKS.INIT',
                     fformat="init", name='PORO', grid=gg)

    def test_export_roff(self):
        """
        Property import from Eclipse. Then export to roff.
        """

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        gg = Grid()
        gg.from_file('../xtgeo-testdata/3dgrids/bri/B.GRID',
                     fformat="grid")
        po = GridProperty()
        self.logger.info("Import INIT...")
        po.from_file('../xtgeo-testdata/3dgrids/bri/B.INIT',
                     fformat="init", name='PORO', grid=gg)

        po.to_file('TMP/bdata.roff')

    def test_io_roff_discrete(self):
        """
        Import ROFF discrete property; then export to ROFF
        """

        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        po = GridProperty()
        po.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2_zone.roffbin',
                     fformat="roff", name='Zone')

        self.logger.info("\nCodes ({})\n{}".format(po.ncodes, po.codes))

        # tests:
        self.assertEqual(po.ncodes, 19)
        self.logger.debug(po.codes[17])
        self.assertEqual(po.codes[17], "SEQ2")

        # export to ROFF
        # po.to_file("TMP/zone.roff")

if __name__ == '__main__':

    unittest.main()
