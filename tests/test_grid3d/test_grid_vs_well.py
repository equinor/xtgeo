# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function
import os

from xtgeo.grid3d import Grid
from xtgeo.well import Well
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TDMP = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================

GRIDFILE = "../xtgeo-testdata/3dgrids/reek/reek_sim_grid.roff"
ZONEFILE = "../xtgeo-testdata/3dgrids/reek/reek_sim_zone.roff"
WELL1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
WELL2 = "../xtgeo-testdata/wells/reek/1/OP_2.w"
WELL3 = "../xtgeo-testdata/wells/reek/1/OP_3.w"
WELL4 = "../xtgeo-testdata/wells/reek/1/OP_4.w"
WELL5 = "../xtgeo-testdata/wells/reek/1/OP_5.w"
WELL6 = "../xtgeo-testdata/wells/reek/1/WI_1.w"
WELL7 = "../xtgeo-testdata/wells/reek/1/WI_3.w"

# A problem here is that the OP wells has very few samples, which
# makes a assumed match of 100% (since only one point)
# Also, the match percent seems to be a bit unstable, hence
# the rounding to INT...


def test_report_zlog_mismatch():
    """Report zone log mismatch grid and well."""
    g1 = Grid()
    g1.from_file(GRIDFILE)

    zo = GridProperty()
    zo.from_file(ZONEFILE, name="Zone")

    w1 = Well(WELL1)
    w2 = Well(WELL2)
    w3 = Well(WELL3)
    w4 = Well(WELL4)
    w5 = Well(WELL5)
    w6 = Well(WELL6)
    w7 = Well(WELL7)

    wells = [w1, w2, w3, w4, w5, w6, w7]

    resultd = {}
    # matchd = {'WI_1': 69, 'WI_3': 70, 'OP_4': 74, 'OP_5': 75, 'OP_1': 75,
    #           'OP_2': 74, 'OP_3': 70}

    for wll in wells:
        response = g1.report_zone_mismatch(
            well=wll,
            zonelogname="Zonelog",
            zoneprop=zo,
            onelayergrid=g2,
            zonelogrange=(1, 3),
            depthrange=[1300, 9999],
        )

        if response is None:
            continue
        else:
            logger.info(response)
            match = int(float("{0:.4f}".format(response[0])))
            logger.info(match)
            resultd[wll.wellname] = match


#    assert resultd == matchd
