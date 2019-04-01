# coding: utf-8
"""Testing block wells sets from and to ROXAPI"""

from __future__ import division, absolute_import
from __future__ import print_function
import os

import xtgeo
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TESTPATH = '../xtgeo-testdata-equinor/data/rmsprojects'

PROJ = dict()
PROJ['1.1'] = os.path.join(TESTPATH, 'reek.rms10.1.1')
PROJ['1.2.1'] = os.path.join(TESTPATH, 'reek.rms11.0.1')
PROJ['1.3'] = os.path.join(TESTPATH, 'reek.rms11.1.0')


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_blockedwells():
    """Get a blocked well set from a RMS project."""
    from roxar import __version__ as ver

    logger.info('Project is {}'.format(PROJ[ver]))

    bwells = xtgeo.BlockedWells()
    bwells.from_roxar(PROJ[ver], 'Reek_geo', 'BW', ijk=False,
                      lognames=['Poro'])

    print(bwells.wells)

    for wll in bwells.wells:
        print(wll.name)
        # logger.info(bwells.dataframe.head())


# @tsetup.equinor
# @tsetup.skipunlessroxar
# def test_rox_get_blockedwells_oneliner():
#     """Get a blocked well set from a RMS project using a oneliner."""
#     from roxar import __version__ as ver

#     logger.info('Project is {}'.format(PROJ[ver]))

#     bwell = xtgeo.blockedwell_from_roxar(PROJ[ver], 'Reek_geo', 'BW', 'OP_1',
#                                          ijk=False, lognames=['Poro'])
#     logger.info(bwell.dataframe.head())
#     #
