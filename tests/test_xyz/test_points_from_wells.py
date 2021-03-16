import pathlib

import glob
from xtgeo.xyz import Points
from xtgeo.well import Well

WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5]*.w")


def test_get_zone_tops_one_well(testpath, tmp_path):
    """Import a well and get the zone tops"""

    wlist = [Well((testpath / WFILES1), zonelogname="Zonelog")]

    mypoints = Points()
    mypoints.from_wells(wlist)

    mypoints.to_file(
        tmp_path / "points_w1.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        tmp_path / "points_w1.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_tops_one_well_w_undef(testpath, tmp_path):
    """Import a well and get the zone tops, include undef transition"""

    wlist = [Well((testpath / WFILES1), zonelogname="Zonelog")]

    mypoints = Points()
    mypoints.from_wells(wlist, use_undef=True)

    mypoints.to_file(
        tmp_path / "points_w1_w_undef.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        tmp_path / "points_w1_w_undef.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_tops_some_wells(testpath, tmp_path):
    """Import some well and get the zone tops"""

    wlist = [
        Well(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoints = Points()
    mypoints.from_wells(wlist)

    mypoints.to_file(
        tmp_path / "points_w1_many.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        tmp_path / "points_w1_many.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_thickness_one_well(testpath):
    """Import a well and get the zone thicknesses"""

    wlist = [Well((testpath / WFILES1), zonelogname="Zonelog")]

    mypoints = Points()
    mypoints.from_wells(wlist, tops=False, zonelist=[12, 13, 14])


def test_get_zone_thickness_some_wells(testpath, tmp_path):
    """Import some wells and get the zone thicknesses"""

    wlist = [
        Well(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoints = Points()
    mypoints.from_wells(wlist, tops=False, zonelist=(1, 22))

    mypoints.to_file(
        tmp_path / "zpoints_w_so622.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        pfilter={"ZoneName": ["SO622"]},
    )

    # filter, for backwards compatibility
    mypoints.to_file(
        tmp_path / "zpoints_w_so622_again.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        filter={"ZoneName": ["SO622"]},
    )


def test_get_faciesfraction_some_wells(testpath, tmp_path):
    """Import some wells and get the facies fractions per zone, for
    wells < 70 degrees inclination.
    """

    wlist = [
        Well(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoints = Points()
    facname = "Facies"
    fcode = [1]

    mypoints.dfrac_from_wells(wlist, facname, fcode, zonelist=None, incl_limit=70)

    # rename column
    mypoints.zname = "FACFRAC"

    myquery = 'WELLNAME == "OP_1" and ZONE == 1'
    usedf = mypoints.dataframe.query(myquery)

    assert abs(usedf[mypoints.zname].values[0] - 0.86957) < 0.001

    mypoints.to_file(
        tmp_path / "ffrac_per_zone.rmsasc",
        fformat="rms_attr",
        attributes=["WELLNAME", "ZONE"],
        pfilter={"ZONE": [1]},
    )
