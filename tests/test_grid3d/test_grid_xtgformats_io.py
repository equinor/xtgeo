# coding: utf-8
"""Testing new xtg formats."""
import pathlib
import uuid
import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TPATH = xtg.testpath

REEKGRID1 = "../xtgeo-testdata/3dgrids/reek/reek_geo_grid.roff"
REEKPROP1 = "../xtgeo-testdata/3dgrids/reek2/geogrid--poro.roff"
REEKPROP3 = "../xtgeo-testdata/3dgrids/reek2/geogrid--facies.roff"


def test_grid_export_import_many():
    """Test exporting etc to xtgcpgeom format."""
    grid1 = xtgeo.Grid(REEKGRID1)

    nrange = 50

    fformat = "xtgcpgeom"

    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = uuid.uuid4().hex + "." + fformat
        fname = pathlib.Path(TMPD) / fname
        fnames.append(fname)
        grid1.to_file(fname, fformat=fformat)

    logger.info("Timing export %s gridgeom with %s: %s", nrange, fformat, xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    grid2 = None
    for fname in fnames:
        grid2 = xtgeo.Grid()
        grid2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s gridgeom with %s: %s", nrange, fformat, xtg.timer(t1))

    assert grid1._zcornsv.mean() == pytest.approx(grid2._zcornsv.mean())
    assert grid1._coordsv.mean() == pytest.approx(grid2._coordsv.mean())
    assert grid1._actnumsv.mean() == pytest.approx(grid2._actnumsv.mean())


def test_gridprop_export_import_many():
    """Test exporting etc to xtgcpprop format."""
    prop1 = xtgeo.GridProperty(REEKPROP1)

    print(prop1.values1d)

    nrange = 50

    fformat = "xtgcpprop"

    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = uuid.uuid4().hex + "." + fformat
        fname = pathlib.Path(TMPD) / fname
        fnames.append(fname)
        prop1.to_file(fname, fformat=fformat)

    logger.info("Timing export %s gridgeom with %s: %s", nrange, fformat, xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    grid2 = None
    for fname in fnames:
        grid2 = xtgeo.GridProperty()
        grid2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s gridgeom with %s: %s", nrange, fformat, xtg.timer(t1))

    # assert grid1._zcornsv.mean() == pytest.approx(grid2._zcornsv.mean())
    # assert grid1._coordsv.mean() == pytest.approx(grid2._coordsv.mean())
    # assert grid1._actnumsv.mean() == pytest.approx(grid2._actnumsv.mean())
