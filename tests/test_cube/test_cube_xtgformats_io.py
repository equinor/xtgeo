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
TPATH = xtg.testpathobj

TESTSET1 = TPATH / "cubes/reek/syntseis_20030101_seismic_depth_stack.segy"


def test_cube_export_import_many():
    """Test exporting etc to xtgregcube format."""
    cube1 = xtgeo.Cube(TESTSET1)

    nrange = 50

    fformat = "xtgregcube"

    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = uuid.uuid4().hex + "." + fformat

        fname = pathlib.Path(TMPD) / fname
        fnames.append(fname)
        cube1.to_file(fname, fformat=fformat)

    logger.info("Timing export %s cubes with %s: %s", nrange, fformat, xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    for fname in fnames:
        cube2 = xtgeo.Cube()
        cube2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s cubes with %s: %s", nrange, fformat, xtg.timer(t1))

    assert cube1.values.mean() == pytest.approx(cube2.values.mean())
