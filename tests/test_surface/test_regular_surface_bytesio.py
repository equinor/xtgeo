# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
from os.path import join
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
@tsetup.skipiftravis
@tsetup.skipifpython2
def test_irapbin_import_bytesio():
    """Import Irap binary via bytesIO"""
    logger.info("Import file as BytesIO")

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
@tsetup.skipifwindows
@tsetup.skipiftravis
@tsetup.skipifpython2
def test_irapbin_export_bytesio():
    """Export Irap binary to bytesIO, then read again"""
    logger.info("Import and export to bytesio")

    xsurf = xtgeo.RegularSurface(TESTSET1, fformat="irap_binary")
    assert xsurf.ncol == 554
    assert xsurf.nrow == 451
    assert (abs(xsurf.values.mean() - 1698.648) < 0.01)
    xsurf.describe()
    xsurf.to_file(join(TMPD, "bytesio1.gri"), fformat="irap_binary")

    xsurf.values -= 200

    stream = io.BytesIO()

    xsurf.to_file(stream, fformat="irap_binary")

    xsurfx = xtgeo.RegularSurface(stream, fformat="irap_binary")
    logger.info("XSURFX mean %s", xsurfx.values.mean())

    with open(join(TMPD, "bytesio2.gri"), "wb") as myfile:
        myfile.write(stream.getvalue())

    xsurf1 = xtgeo.RegularSurface(join(TMPD, "bytesio1.gri"), fformat="irap_binary")
    xsurf2 = xtgeo.RegularSurface(join(TMPD, "bytesio2.gri"), fformat="irap_binary")
    assert abs(xsurf1.values.mean() - xsurf2.values.mean() - 200) < 0.001

    stream.close()


@tsetup.skipifmac
@tsetup.skipiftravis
@tsetup.skipifpython2
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
@tsetup.skipiftravis
def test_get_regsurff():

    sfile = TESTSET1
    logger.info("File is %s", sfile)
    for _itmp in range(20):
        rf = xtgeo.RegularSurface(sfile, fformat="irap_binary")
        assert (abs(rf.values.mean() - 1698.648) < 0.01)
        print(_itmp)


@tsetup.skipifmac
@tsetup.skipifwindows
@tsetup.skipiftravis
@tsetup.skipifpython2
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
