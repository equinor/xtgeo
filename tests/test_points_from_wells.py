import os
import sys
import glob
import logging
from xtgeo.xyz import Points
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

format = xtg.loggingformat

logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

logger = logging.getLogger(__name__)

wfiles = "../xtgeo-testdata/wells/tro/1/31_2-1.w"


def test_get_zone_tops_some_wells():
    """Import somewells and get the zone tops"""

    wlist = []
    for w in glob.glob(wfiles):
        wlist.append(Well(w))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist)

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))

    mypoints.to_file('TMP/points_w1.rmsasc', fformat='rms_attr',
                     attributes=['WellName', 'TopName'])


def test_get_zone_thickness_some_wells():
    """Import somewells and get the zone thicknesses"""

    wlist = []
    for w in glob.glob(wfiles):
        wlist.append(Well(w))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, tops=False, zonelist=[12, 13, 14])

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))
