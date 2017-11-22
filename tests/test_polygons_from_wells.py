import os
import sys
import glob
import logging
from xtgeo.xyz import Polygons
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

wfiles1 = "../xtgeo-testdata/wells/tro/1/31_2-1.w"
wfiles2 = "../xtgeo-testdata/wells/tro/1/31*.w"


def test_get_polygons_one_well():
    """Import a well and get the polygon segments"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoly = Polygons()
    nwell = mypoly.from_wells(wlist, 21)

    print(mypoly.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))

    mypoly.to_file('TMP/poly_w1.irapasc')


def test_get_polygons_many_wells():
    """Import some wells and get the polygon segments"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        print('Imported well {}'.format(w))

    mypoly = Polygons()
    nwell = mypoly.from_wells(wlist, 21, resample=10)

    print(mypoly.dataframe)

    print('Number of well made to tops: {}'.format(nwell))

    mypoly.to_file('TMP/poly_w1_many.irapasc')
