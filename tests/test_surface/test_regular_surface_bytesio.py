# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
import os.path
import io

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TPATH = xtg.testpath

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

# =============================================================================
# Do tests
# =============================================================================

TESTSET1 = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"


@tsetup.skipifwindows
def test_irapbin_import_bytesio():
    """Import Irap binary via bytesIO"""
    logger.info("Import and export...")

    with open(TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())
    print(dir(stream))
    print(type(stream.getvalue()))

    xsurf = xtgeo.RegularSurface(stream, fformat="irap_binary")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    xsurf.describe()


@tsetup.skipifwindows
def test_irapbin_load_meta_first_bytesio():
    """Import Irap binary via bytesIO, by just loading metadata first"""
    logger.info("Import and export...")

    with open(TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())

    t0 = xtg.timer()
    for _inum in range(1000):
        # should go very fast
        xsurf = xtgeo.RegularSurface(stream, fformat="irap_binary", values=False)
    t1 = xtg.timer(t0)
    logger.info("Time to import 1000 entries: %4.3f secs", t1)
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    xsurf.describe()

    xsurf.load_values()
    xsurf.describe()
    stream.close()

    # stream is now closed
    with pytest.raises(ValueError) as verr:
        xsurf = xtgeo.RegularSurface(stream, fformat="irap_binary", values=False)
    assert "I/O operation on closed file" in str(verr.value)
