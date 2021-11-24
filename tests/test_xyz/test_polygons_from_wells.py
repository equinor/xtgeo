import glob
import pathlib

import xtgeo
from xtgeo.xyz import Polygons

WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5].w")


def test_get_polygons_one_well_deprecated(testpath, tmp_path):
    """Import a well and get the polygon segments using deprecated method."""
    wlist = [xtgeo.well_from_file((testpath / WFILES1), zonelogname="Zonelog")]

    mypoly = Polygons()
    mypoly.from_wells(wlist, 2)

    mypoly.to_file(tmp_path / "poly_w1.irapasc")


def test_get_polygons_many_wells_deprecated(testpath, tmp_path):
    """Import some wells and get the polygon segments using deprecated method."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoly = Polygons()
    nwells = mypoly.from_wells(wlist, 2, resample=10)
    assert nwells == 5
    assert mypoly.nrow == 21

    mypoly.to_file(tmp_path / "poly_w1_many.irapasc")


def test_get_polygons_many_wells_classmethod(testpath, tmp_path):
    """Import some wells and get the polygon segments using new class method."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoly = xtgeo.polygons_from_wells(wlist, None, 2, resample=10)
    print(mypoly.dataframe)

    mypoly.to_file(tmp_path / "poly_w1_many_classmethod.irapasc")

    assert mypoly.get_nwells() == 5

    # compare with deprecated legacy result
    mypoly.dataframe = mypoly.dataframe.drop("WellName", axis=1)
    mypoly_legacy = Polygons()
    mypoly_legacy.from_wells(wlist, 2, resample=10)
    assert mypoly_legacy.dataframe.equals(mypoly.dataframe)
