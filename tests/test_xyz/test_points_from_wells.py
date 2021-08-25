import glob
import pathlib

import pytest
import xtgeo
from xtgeo.xyz import Points

WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5]*.w")


def test_get_zone_tops_one_well_old(testpath, tmp_path):
    """Import a well and get the zone tops, old method to be depr."""

    wlist = [xtgeo.well_from_file(testpath / WFILES1, zonelogname="Zonelog")]

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


def test_get_zone_tops_one_well_glassmethod(testpath, tmp_path):
    """Import a well and get the zone tops"""

    wlist = [xtgeo.well_from_file(testpath / WFILES1, zonelogname="Zonelog")]

    mypoints = xtgeo.points_from_wells(wlist)

    assert mypoints.dataframe["TopName"][2] == "TopBelow_TopLowerReek"
    assert mypoints.dataframe["X_UTME"][2] == pytest.approx(462698.333)

    mypoints.to_file(
        tmp_path / "points_w1_classmethod.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "TopName"],
    )

    mypoints.to_file(
        tmp_path / "points_w1_classmethod.rmswpicks",
        fformat="rms_wellpicks",
        wcolumn="WellName",
        hcolumn="TopName",
    )


def test_get_zone_tops_one_well_w_undef(testpath):
    """Import a well and get the zone tops, include undef transition"""

    single = xtgeo.well_from_file((testpath / WFILES1), zonelogname="Zonelog")
    wlist = [single]

    # legacy
    p1 = Points()
    p1.from_wells(wlist, use_undef=True)

    # classmethod
    p2 = xtgeo.points_from_wells(single, use_undef=True)
    p3 = xtgeo.points_from_wells(wlist, use_undef=False)  # intentional to use wlist

    assert p1.dataframe.equals(p2.dataframe)
    assert not p2.dataframe.equals(p3.dataframe)

    assert p2.dataframe["Zone"][0] == 0
    assert p3.dataframe["Zone"][0] == 1


def test_get_zone_tops_some_wells(testpath, tmp_path):
    """Import some well and get the zone tops"""

    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    # legacy
    p1 = Points()
    p1.from_wells(wlist)

    # classmethod
    p2 = xtgeo.points_from_wells(wlist)
    assert p1.dataframe.equals(p2.dataframe)


def test_get_zone_thickness_one_well(testpath):
    """Import a well and get the zone thicknesses"""

    wlist = [xtgeo.well_from_file(testpath / WFILES1, zonelogname="Zonelog")]

    mypoints = Points()
    mypoints = xtgeo.points_from_wells(wlist, tops=False, zonelist=[1, 2, 3])
    mypoints.zname = "THICKNESS"
    assert mypoints.dataframe["THICKNESS"][0] == pytest.approx(16.8397)


def test_get_zone_thickness_some_wells(testpath, tmp_path):
    """Import some wells and get the zone thicknesses"""

    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]
    mypoints = xtgeo.points_from_wells(wlist, tops=False, zonelist=(1, 22))

    # alternative import well files and set zonelogname
    wlist2 = glob.glob(str(testpath / WFILES2))
    mypoints2 = xtgeo.points_from_wells(
        wlist2, tops=False, zonelogname="Zonelog", zonelist=(1, 22)
    )

    assert mypoints.dataframe.equals(mypoints2.dataframe)

    mypoints.to_file(
        tmp_path / "zpoints_w_so622.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        pfilter={"ZoneName": ["SO622"]},
    )

    # filter instead of pfilter, for backwards compatibility
    mypoints.to_file(
        tmp_path / "zpoints_w_so622_again.rmsasc",
        fformat="rms_attr",
        attributes=["WellName", "ZoneName"],
        filter={"ZoneName": ["SO622"]},
    )

    assert mypoints.dataframe["X_UTME"][0] == pytest.approx(462698.2375)
    assert mypoints.dataframe["WellName"][0] == "OP_1_RKB"
    assert mypoints.dataframe["X_UTME"][20] == pytest.approx(462749.4820)
    assert mypoints.dataframe["WellName"][20] == "OP_5"


def test_get_faciesfraction_some_wells_old(testpath, tmp_path):
    """Import some wells and get the facies fractions per zone."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoints = Points()
    facname = "Facies"
    fcode = [1]

    # In this example for wells < 70 degrees inclination.
    mypoints.dfrac_from_wells(wlist, facname, fcode, zonelist=None, incl_limit=70)

    # rename column
    mypoints.zname = "FACFRAC"

    myquery = 'WELLNAME == "OP_1" and ZONE == 1'
    usedf = mypoints.dataframe.query(myquery)

    assert abs(usedf[mypoints.zname].values[0] - 0.86957) < 0.001

    mypoints.to_file(
        tmp_path / "ffrac_per_zone_old.rmsasc",
        fformat="rms_attr",
        attributes=["WELLNAME", "ZONE"],
        pfilter={"ZONE": [1]},
    )


def test_get_faciesfraction_some_wells(testpath, tmp_path):
    """Import some wells and get the facies fractions per zone."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    facname = "Facies"
    fcode = [1]

    # In this example for wells < 70 degrees inclination.
    mypoints = xtgeo.points_from_wells_dfrac(
        wlist, facname, fcode, zonelist=None, incl_limit=70
    )

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
