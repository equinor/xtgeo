from os.path import join

from xtgeo.xyz import Points
from xtgeo.xyz import Polygons

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TSTPATH = xtg.testpathobj

POLSET2 = join(TSTPATH, "polygons/reek/1/polset2.pol")
POINTSET2 = join(TSTPATH, "points/reek/1/pointset2.poi")


def test_points_in_polygon():
    """Import XYZ points and do operations if inside or outside"""

    poi = Points(POINTSET2)
    pol = Polygons(POLSET2)
    assert poi.nrow == 30

    # remove points in polygon
    poi.operation_polygons(pol, 0, opname="eli", where=True)

    assert poi.nrow == 19
    poi.to_file(join(TMPD, "poi_test.poi"))

    poi = Points(POINTSET2)
    # remove points outside polygon
    poi.operation_polygons(pol, 0, opname="eli", inside=False, where=True)
    assert poi.nrow == 1
