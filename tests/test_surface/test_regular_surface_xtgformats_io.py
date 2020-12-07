# coding: utf-8
"""Testing new xtg formats."""
import pathlib
import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TPATH = xtg.testpath

TESTSET1 = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"


def test_xtgregsurf_export_import_many():
    """Test exporting to xtgregsurf format."""
    surf1 = xtgeo.RegularSurface(TESTSET1)
    nrange = 500

    fformat = "xtgregsurf"
    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = "$md5sum" + "." + fformat
        fname = pathlib.Path(TMPD) / fname
        surf1.values += num
        newname = surf1.to_file(fname, fformat=fformat)
        fnames.append(newname)

    logger.info("Timing export %s surfs with %s: %s", nrange, fformat, xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    for fname in fnames:
        surf2 = xtgeo.RegularSurface()
        surf2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s surfs with %s: %s", nrange, fformat, xtg.timer(t1))

    assert surf1.values.mean() == pytest.approx(surf2.values.mean())
