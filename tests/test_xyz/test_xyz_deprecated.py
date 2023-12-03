"""Collect deprecated stuff for Points() and Polygons()."""
import glob
import pathlib

import pandas as pd
import pytest
import xtgeo
from packaging import version
from xtgeo.common.version import __version__ as xtgeo_version

PFILE1A = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.zmap")
PFILE1B = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.xyz")
PFILE1C = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.pol")
SURFACE = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5]*.w")
POLSET2 = pathlib.Path("polygons/reek/1/polset2.pol")
POINTSET2 = pathlib.Path("points/reek/1/pointset2.poi")
CSV1 = pathlib.Path("3dgrids/etc/gridqc1_rms_cellcenter.csv")


def test_load_points_polygons_file_deprecated(testpath):
    """Load from file."""
    with pytest.warns(
        DeprecationWarning, match="Initializing directly from file name is deprecated"
    ):
        poi = xtgeo.Points(testpath / POINTSET2)

    assert poi.nrow == 30

    with pytest.warns(
        DeprecationWarning, match="Initializing directly from file name is deprecated"
    ):
        pol = xtgeo.Polygons(testpath / POLSET2)

    assert pol.nrow == 25


def test_points_from_list_deprecated():
    plist = [(234, 556, 11), (235, 559, 14), (255, 577, 12)]

    with pytest.warns(DeprecationWarning, match="Use direct"):
        mypoints = xtgeo.Points(plist)

        old_points = xtgeo.Points()
        old_points.from_list(plist)
        assert mypoints.dataframe.equals(old_points.dataframe)


def test_import_from_dataframe_deprecated(testpath):
    """Import Points via Pandas dataframe, deprecated behaviour."""

    dfr = pd.read_csv(testpath / CSV1, skiprows=3)

    mypoints = xtgeo.Points()
    attr = {"IX": "I", "JY": "J", "KZ": "K"}
    with pytest.warns(DeprecationWarning):
        mypoints.from_dataframe(dfr, east="X", north="Y", tvdmsl="Z", attributes=attr)

    assert mypoints.dataframe.X_UTME.mean() == dfr.X.mean()


@pytest.mark.parametrize(
    "filename, fformat",
    [
        (PFILE1A, "zmap"),
        (PFILE1B, "xyz"),
        (PFILE1C, "pol"),
    ],
)
def test_polygons_from_file_alternatives_with_deprecated(testpath, filename, fformat):
    if version.parse(xtgeo_version) < version.parse("2.21"):
        # to avoid test failure before tag is actually set
        pytest.skip()
    else:
        with pytest.warns(DeprecationWarning):
            polygons1 = xtgeo.Polygons(testpath / filename)

        polygons2 = xtgeo.Polygons()
        with pytest.warns(DeprecationWarning):
            polygons2.from_file(testpath / filename, fformat=fformat)

        polygons3 = xtgeo.polygons_from_file(testpath / filename, fformat=fformat)
        polygons4 = xtgeo.polygons_from_file(testpath / filename)

        pd.testing.assert_frame_equal(polygons1.dataframe, polygons2.dataframe)
        pd.testing.assert_frame_equal(polygons2.dataframe, polygons3.dataframe)
        pd.testing.assert_frame_equal(polygons3.dataframe, polygons4.dataframe)


def test_get_polygons_one_well_deprecated(testpath, tmp_path):
    """Import a well and get the polygon segments."""
    wlist = [xtgeo.well_from_file((testpath / WFILES1), zonelogname="Zonelog")]

    mypoly = xtgeo.Polygons()
    with pytest.warns(DeprecationWarning):
        mypoly.from_wells(wlist, 2)

    assert mypoly.nrow == 29
    mypoly.to_file(tmp_path / "poly_w1.irapasc")


def test_get_polygons_many_wells_deprecated(testpath, tmp_path):
    """Import some wells and get the polygon segments."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoly = xtgeo.Polygons()
    with pytest.warns(DeprecationWarning):
        mypoly.from_wells(wlist, 2, resample=10)

    mypoly.to_file(tmp_path / "poly_w1_many.irapasc")

    # compare with new class initialization (which also adds a well column)
    mypoly2 = xtgeo.polygons_from_wells(wlist, 2, resample=10)
    mypoly2.to_file(tmp_path / "poly_w1_many_new.irapasc")

    pd.testing.assert_frame_equal(
        mypoly.dataframe.iloc[:, 0:4], mypoly2.dataframe.iloc[:, 0:4]
    )


def test_init_with_surface_deprecated(testpath):
    """Initialise points object with surface instance, to be deprecated."""
    surf = xtgeo.surface_from_file(testpath / SURFACE)
    with pytest.warns(
        DeprecationWarning,
        match="Initializing directly from RegularSurface is deprecated",
    ):
        poi = xtgeo.Points(surf)

    poi.zname = "VALUES"
    pd.testing.assert_frame_equal(poi.dataframe, surf.dataframe())


def test_get_zone_tops_one_well_deprecated(testpath):
    """Import a well and get the zone tops, old method to be depr."""

    wlist = [xtgeo.well_from_file(testpath / WFILES1, zonelogname="Zonelog")]

    mypoints = xtgeo.Points()
    with pytest.warns(DeprecationWarning, match="from_wells is deprecated"):
        mypoints.from_wells(wlist)

    mypoints_new = xtgeo.points_from_wells(wlist)

    pd.testing.assert_frame_equal(mypoints.dataframe, mypoints_new.dataframe)


def test_get_zone_tops_some_wells_deprecated(testpath):
    """Import some well and get the zone tops"""

    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    # legacy
    p1 = xtgeo.Points()
    with pytest.warns(DeprecationWarning, match="from_wells is deprecated as of"):
        p1.from_wells(wlist)
    assert p1.nrow == 28

    # classmethod
    p2 = xtgeo.points_from_wells(wlist)
    assert p1.dataframe.equals(p2.dataframe)


def test_get_faciesfraction_some_wells_deprecated(testpath):
    """Import some wells and get the facies fractions per zone."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoints = xtgeo.Points()
    facname = "Facies"
    fcode = [1]

    with pytest.warns(DeprecationWarning, match="dfrac_from_wells is deprecated"):
        mypoints.dfrac_from_wells(wlist, facname, fcode, zonelist=None, incl_limit=70)

    # rename column
    mypoints.zname = "FACFRAC"

    myquery = 'WELLNAME == "OP_1" and ZONE == 1'
    usedf = mypoints.dataframe.query(myquery)

    assert abs(usedf[mypoints.zname].values[0] - 0.86957) < 0.001
