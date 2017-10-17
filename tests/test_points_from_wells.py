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

wfiles1 = "../xtgeo-testdata/wells/tro/1/31_2-1.w"
wfiles2 = "../xtgeo-testdata/wells/tro/1/31*.w"


def test_get_zone_tops_one_well():
    """Import a well and get the zone tops"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist)

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))

    mypoints.to_file('TMP/points_w1.rmsasc', fformat='rms_attr',
                     attributes=['WellName', 'TopName'])

    mypoints.to_file('TMP/points_w1.rmswpicks', fformat='rms_wellpicks',
                     wcolumn='WellName', hcolumn='TopName')


def test_get_zone_tops_one_well_w_undef():
    """Import a well and get the zone tops, include undef transition"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, use_undef=True)

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))

    mypoints.to_file('TMP/points_w1_w_undef.rmsasc', fformat='rms_attr',
                     attributes=['WellName', 'TopName'])

    mypoints.to_file('TMP/points_w1_w_undef.rmswpicks',
                     fformat='rms_wellpicks',
                     wcolumn='WellName', hcolumn='TopName')


def test_get_zone_tops_some_wells():
    """Import some well and get the zone tops"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist)

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))

    mypoints.to_file('TMP/points_w1_many.rmsasc', fformat='rms_attr',
                     attributes=['WellName', 'TopName'])

    mypoints.to_file('TMP/points_w1_many.rmswpicks', fformat='rms_wellpicks',
                     wcolumn='WellName', hcolumn='TopName')


def test_get_zone_thickness_one_well():
    """Import a wells and get the zone thicknesses"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, tops=False, zonelist=[12, 13, 14])

    print(mypoints.dataframe)

    logger.info('Number of well made to tops: {}'.format(nwell))


def test_get_zone_thickness_some_wells():
    """Import some wells and get the zone thicknesses"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname='ZONELOG'))
        logger.info('Imported well {}'.format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, tops=False, zonelist=(1, 22))

    print(nwell, '\n', mypoints.dataframe)

    mypoints.to_file('TMP/zpoints_w_so622.rmsasc', fformat='rms_attr',
                     attributes=['WellName', 'ZoneName'],
                     filter={'ZoneName': ['SO622']})


    logger.info('Number of well made to tops: {}'.format(nwell))
