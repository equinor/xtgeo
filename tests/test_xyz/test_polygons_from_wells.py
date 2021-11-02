import glob
import pathlib

from xtgeo.well import Well
from xtgeo.xyz import Polygons

WFILES1 = pathlib.Path("wells/reek/1/OP_1.w")
WFILES2 = pathlib.Path("wells/reek/1/OP_[1-5].w")


def test_get_polygons_one_well(testpath, tmp_path):
    """Import a well and get the polygon segments."""
    wlist = [Well((testpath / WFILES1), zonelogname="Zonelog")]

    mypoly = Polygons()
    mypoly.from_wells(wlist, 2)

    mypoly.to_file(tmp_path / "poly_w1.irapasc")


def test_get_polygons_many_wells(testpath, tmp_path):
    """Import some wells and get the polygon segments."""
    wlist = [
        Well(wpath, zonelogname="Zonelog")
        for wpath in glob.glob(str(testpath / WFILES2))
    ]

    mypoly = Polygons()
    mypoly.from_wells(wlist, 2, resample=10)

    mypoly.to_file(tmp_path / "poly_w1_many.irapasc")
