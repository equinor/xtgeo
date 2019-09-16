# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function
import os
import sys

from xtgeo.grid3d import Grid
from xtgeo.well import Well
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

TDMP = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================

gridfile = "../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff"
zonefile = "../xtgeo-testdata/3dgrids/reek/reek_sim_zone.roff"
well1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
well2 = "../xtgeo-testdata/wells/reek/1/OP_2.w"
well3 = "../xtgeo-testdata/wells/reek/1/OP_3.w"
well4 = "../xtgeo-testdata/wells/reek/1/OP_4.w"
well5 = "../xtgeo-testdata/wells/reek/1/OP_5.w"
well6 = "../xtgeo-testdata/wells/reek/1/WI_1.w"
well7 = "../xtgeo-testdata/wells/reek/1/WI_3.w"

# A problem here is that the OP wells has very few samples, which
# makes a assumed match of 100% (since only one point)
# Also, the match percent seems to be a bit unstable, hence
# the rounding to INT...


def test_report_zlog_mismatch():
    """Report zone log mismatch grid and well."""
    logger.info("Name is {}".format(__name__))
    g1 = Grid()
    g1.from_file(gridfile)

    g2 = Grid()
    g2.from_file(gridfile)

    g2.reduce_to_one_layer()
    g2.to_file(os.path.join(TDMP, "test.roff"), fformat="roff")

    z = GridProperty()
    z.from_file(zonefile, name="Zone")

    w1 = Well(well1)
    w2 = Well(well2)
    w3 = Well(well3)
    w4 = Well(well4)
    w5 = Well(well5)
    w6 = Well(well6)
    w7 = Well(well7)

    wells = [w1, w2, w3, w4, w5, w6, w7]

    resultd = {}
    # matchd = {'WI_1': 69, 'WI_3': 70, 'OP_4': 74, 'OP_5': 75, 'OP_1': 75,
    #           'OP_2': 74, 'OP_3': 70}

    for w in wells:
        response = g1.report_zone_mismatch(
            well=w,
            zonelogname="Zonelog",
            zoneprop=z,
            onelayergrid=g2,
            zonelogrange=(1, 3),
            option=0,
            depthrange=[1300, 9999],
        )

        if response is None:
            continue
        else:
            logger.info(response)
            match = int(float("{0:.4f}".format(response[0])))
            logger.info(match)
            resultd[w.wellname] = match


#    assert resultd == matchd
