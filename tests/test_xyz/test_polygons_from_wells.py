import os
import sys
import glob
import logging
from xtgeo.xyz import Polygons
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog
import tests.test_common.test_xtg as tsetup

path = "TMP"
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

wfiles1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
wfiles2 = "../xtgeo-testdata/wells/reek/1/OP_[1-5].w"


@tsetup.skipifroxar
def test_get_polygons_one_well():
    """Import a well and get the polygon segments"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoly = Polygons()
    nwell = mypoly.from_wells(wlist, 2)

    print(mypoly.dataframe)

    logger.info("Number of well made to tops: {}".format(nwell))

    mypoly.to_file("TMP/poly_w1.irapasc")


@tsetup.skipifroxar
def test_get_polygons_many_wells():
    """Import some wells and get the polygon segments"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname="Zonelog"))
        print("Imported well {}".format(w))

    mypoly = Polygons()
    nwell = mypoly.from_wells(wlist, 2, resample=10)

    print(mypoly.dataframe)

    print("Number of well made to tops: {}".format(nwell))

    mypoly.to_file("TMP/poly_w1_many.irapasc")
