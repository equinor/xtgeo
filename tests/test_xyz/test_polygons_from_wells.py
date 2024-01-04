import glob
import pathlib

import xtgeo

WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5]*.w")


def test_get_polygons_one_well(testpath):
    """Import a well and get the polygon segments."""
    wlist = [xtgeo.well_from_file((testpath / WFILES1), zonelogname="Zonelog")]

    mypoly = xtgeo.polygons_from_wells(wlist, 2)

    assert mypoly.nrow == 29


def test_get_polygons_many_wells(testpath):
    """Import some wells and get the polygon segments."""
    wlist = [
        xtgeo.well_from_file(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]
    mypoly = xtgeo.polygons_from_wells(wlist, 2, resample=10)
    assert mypoly.nrow == 29
