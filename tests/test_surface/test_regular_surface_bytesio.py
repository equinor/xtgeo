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


@tsetup.skipifmac
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
    assert (abs(xsurf.values.mean() - 1698.648) < 0.01)
    xsurf.describe()


@tsetup.skipifmac
def test_get_regsurfi():

    sfile = TESTSET1
    with open(sfile, "rb") as fin:
        stream = io.BytesIO(fin.read())

    logger.info("File is %s", sfile)
    for _itmp in range(20):
        rf = xtgeo.RegularSurface(stream, fformat="irap_binary")
        assert (abs(rf.values.mean() - 1698.648) < 0.01)
        print(_itmp)


@tsetup.skipifmac
def test_get_regsurff():

    sfile = TESTSET1
    logger.info("File is %s", sfile)
    for _itmp in range(20):
        rf = xtgeo.RegularSurface(sfile, fformat="irap_binary")
        assert (abs(rf.values.mean() - 1698.648) < 0.01)
        print(_itmp)


@tsetup.skipifmac
@tsetup.skipifwindows
def test_irapbin_load_meta_first_bytesio():
    """Import Irap binary via bytesIO, by just loading metadata first"""
    logger.info("Import and export...")

    with open(TESTSET1, "rb") as fin:
        stream = io.BytesIO(fin.read())

    xsurf = xtgeo.RegularSurface(stream, fformat="irap_binary", values=False)
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
