# coding: utf-8
"""Testing new xtg formats."""
import pathlib
import uuid

import numpy as np
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


def test_gridprop_partial_read_smallcase():
    """Read a partial property based on ijrange from file."""
    vals = np.zeros((5, 7, 3), dtype=np.float32)
    prp = xtgeo.GridProperty(ncol=5, nrow=7, nlay=3, values=vals)
    prp.values[0, 0, 0:3] = 33
    prp.values[1, 0, 0:3] = 66
    prp.values[1, 1, 0:3] = 44
    print(prp.values)
    fname = pathlib.Path(TMPD) / "grdprop.xtgcpprop"
    prp.to_file(fname, fformat="xtgcpprop")

    # import partial
    prp3 = xtgeo.GridProperty()
    prp3.from_file(fname, fformat="xtgcpprop", ijrange=(1, 2, 1, 2))
    assert prp3.values.all() == prp.values[0:2, 0:2, :].all()


def test_gridprop_partial_read_bigcase():
    """Read a partial property based on ijrange from file, big case measure speed."""
    vals = np.zeros((400, 500, 300), dtype=np.float32)
    prp = xtgeo.GridProperty(ncol=400, nrow=500, nlay=300, values=vals)
    prp.values[0, 0, 0:3] = 33
    prp.values[1, 0, 0:3] = 66
    prp.values[1, 1, 0:3] = 44
    fname = pathlib.Path(TMPD) / "grdprop2.xtgcpprop"
    prp.to_file(fname, fformat="xtgcpprop")

    t0 = xtg.timer()
    for _ in range(10):
        prp.from_file(fname, fformat="xtgcpprop")
    t1 = xtg.timer(t0)
    logger.info("Timing from whole grid IJ 400 x 500: %s", t1)
    grdsize1 = prp.values.size
    # read a subpart and measure time
    t0 = xtg.timer()
    for _ in range(10):
        prp.from_file(fname, fformat="xtgcpprop", ijrange=(350, 400, 1, 50))
    t2 = xtg.timer(t0)
    logger.info("Timing from subrange IJ 50 x 50 : %s", t2)
    grdsize2 = prp.values.size
    gridratio = grdsize2 / grdsize1
    readratio = t2 / t1

    logger.info("Timing: speedratio vs gridsizeratio %s %s", readratio, gridratio)
    assert readratio < 0.05
