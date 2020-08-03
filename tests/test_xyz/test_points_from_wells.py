# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import glob
from xtgeo.xyz import Points
from xtgeo.well import Well
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

wfiles1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
wfiles2 = "../xtgeo-testdata/wells/reek/1/OP_[1-5]*.w"
wfiles3 = "../xtgeo-testdata/wells/reek/1/XP_*.w"


def test_get_zone_tops_one_well():
    """Import a well and get the zone tops"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist)

    print(mypoints.dataframe)

    logger.info("Number of well made to tops: {}".format(nwell))

    mypoints.to_file(
        "TMP/points_w1.rmsasc", fformat="rms_attr", attributes=["WellName", "TopName"]
    )

    mypoints.to_file(
        "TMP/points_w1.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_tops_one_well_w_undef():
    """Import a well and get the zone tops, include undef transition"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, use_undef=True)

    print(mypoints.dataframe)

    logger.info("Number of well made to tops: {}".format(nwell))

    mypoints.to_file(
        "TMP/points_w1_w_undef.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        "TMP/points_w1_w_undef.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_tops_some_wells():
    """Import some well and get the zone tops"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist)

    print(mypoints.dataframe)

    logger.info("Number of well made to tops: {}".format(nwell))

    mypoints.to_file(
        "TMP/points_w1_many.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        "TMP/points_w1_many.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_thickness_one_well():
    """Import a well and get the zone thicknesses"""

    wlist = []
    for w in glob.glob(wfiles1):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, tops=False, zonelist=[12, 13, 14])

    print(mypoints.dataframe)

    logger.info("Number of well made to tops: {}".format(nwell))


def test_get_zone_thickness_some_wells():
    """Import some wells and get the zone thicknesses"""

    wlist = []
    for w in glob.glob(wfiles2):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    nwell = mypoints.from_wells(wlist, tops=False, zonelist=(1, 22))

    print(nwell, "\n", mypoints.dataframe)

    mypoints.to_file(
        "TMP/zpoints_w_so622.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        pfilter={"ZoneName": ["SO622"]},
    )

    # filter, for backwards compatibility
    mypoints.to_file(
        "TMP/zpoints_w_so622_again.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        filter={"ZoneName": ["SO622"]},
    )

    logger.info("Number of well made to tops: {}".format(nwell))


def test_get_faciesfraction_some_wells():
    """Import some wells and get the facies fractions per zone, for
    wells < 70 degrees inclination.
    """

    wlist = []
    for w in sorted(glob.glob(wfiles2)):
        wlist.append(Well(w, zonelogname="Zonelog"))
        logger.info("Imported well {}".format(w))

    mypoints = Points()
    facname = "Facies"
    fcode = [1]

    nwell = mypoints.dfrac_from_wells(
        wlist, facname, fcode, zonelist=None, incl_limit=70
    )

    # rename column
    mypoints.zname = "FACFRAC"

    logger.info("Number of wells is %s, DATAFRAME:\n, %s", nwell, mypoints.dataframe)

    myquery = 'WELLNAME == "OP_1" and ZONE == 1'
    usedf = mypoints.dataframe.query(myquery)

    assert abs(usedf[mypoints.zname].values[0] - 0.86957) < 0.001

    mypoints.to_file(
        "TMP/ffrac_per_zone.rmsasc",
        fformat="rms_attr",
        attributes=["WELLNAME", "ZONE"],
        pfilter={"ZONE": [1]},
    )
