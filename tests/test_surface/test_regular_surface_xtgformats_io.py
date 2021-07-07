# coding: utf-8
"""Testing new xtg formats both natiev ans hdf5 based."""

import pytest

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

TESTSET1 = TPATH / "surfaces/reek/1/topreek_rota.gri"


def test_xtgregsurf_export_import_many(tmp_path):
    """Test exporting to xtgregsurf format."""
    surf1 = xtgeo.RegularSurface(TESTSET1)
    nrange = 500

    fformat = "xtgregsurf"
    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = "$md5sum" + "." + fformat
        fname = tmp_path / fname
        surf1.values += num
        newname = surf1.to_file(fname, fformat=fformat)
        fnames.append(newname)

    logger.info("Timing export %s surfs with %s: %s", nrange, fformat, xtg.timer(t1))
    print("Timing export 500 surfs with xtg: ", xtg.timer(t1))

    # timing of reader
    t1 = xtg.timer()
    for fname in fnames:
        surf2 = xtgeo.RegularSurface()
        surf2.from_file(fname, fformat=fformat)

    logger.info("Timing import %s surfs with %s: %s", nrange, fformat, xtg.timer(t1))

    assert surf1.values.mean() == pytest.approx(surf2.values.mean())


def test_hdf5_export_import_many(tmp_path):
    """Test exporting to hdf5 format."""
    surf1 = xtgeo.RegularSurface(TESTSET1)
    nrange = 1

    fnames = []

    # timing of writer
    t1 = xtg.timer()
    for num in range(nrange):
        fname = "$md5sum" + ".h5"
        fname = tmp_path / fname
        surf1.values += num
        newname = surf1.to_hdf(fname, compression="blosc")
        fnames.append(newname)

    print(f"Timing export {nrange} surfs with hdf5: ", xtg.timer(t1))

    # timing of reader
    surf2 = xtgeo.RegularSurface()
    t1 = xtg.timer()
    for fname in fnames:
        surf2.from_hdf(fname)

    print(f"Timing import {nrange} surfs with hdf5: ", xtg.timer(t1))

    assert surf1.values.mean() == pytest.approx(surf2.values.mean())
