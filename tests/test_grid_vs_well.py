# -*- coding: utf-8 -*-
import sys

from xtgeo.grid3d import Grid
from xtgeo.well import Well
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================


def test_report_zlog_mismatch():
    """Report zone log mismatch grid and well."""
    logger.info('Name is {}'.format(__name__))
    g1 = Grid()
    g1.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')

    g2 = Grid()
    g2.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')

    g2.reduce_to_one_layer()

    z = GridProperty()
    z.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2_zone.roff',
                name='Zone')

    # w1 = Well()
    # w1.from_file('../xtgeo-testdata/wells/gfb/1/34_10-A-42.w')

    w2 = Well('../xtgeo-testdata/wells/gfb/1/34_10-1.w')

    w3 = Well('../xtgeo-testdata/wells/gfb/1/34_10-B-21_B.w')

    wells = [w2, w3]

    for w in wells:
        response = g1.report_zone_mismatch(
            well=w, zonelogname='ZONELOG', mode=0, zoneprop=z,
            onelayergrid=g2, zonelogrange=[0, 19], option=0,
            depthrange=[1700, 9999])

        if response is None:
            continue
        else:
            logger.info(response)
        # if w.wellname == w1.wellname:
        #     match = float("{0:.2f}".format(response[0]))
        #     .logger.info(match)
        #     .assertEqual(match, 72.39, 'Match ration A42')

    # perforation log instead

    for w in wells:

        response = g1.report_zone_mismatch(
            well=w, zonelogname='ZONELOG', mode=0, zoneprop=z,
            onelayergrid=g2, zonelogrange=[0, 19], option=0,
            perflogname='PERFLOG')

        if response is None:
            continue
        else:
            logger.info(response)

        # if w.wellname == w1.wellname:
        #     match = float("{0:.1f}".format(response[0]))
        #     logger.info(match)
        #     assertEqual(match, 87.70, 'Match ratio A42')

        if w.wellname == w3.wellname:
            match = float("{0:.2f}".format(response[0]))
            logger.info(match)
            assert match == 0.0, 'Match ratio B21B'
