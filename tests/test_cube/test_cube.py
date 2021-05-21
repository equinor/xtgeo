# -*- coding: utf-8 -*-
import os
from os.path import join

import pytest
import numpy as np

import xtgeo
from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog


xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit("Cannot find test setup")

TMD = xtg.tmpdir
TPATH = xtg.testpathobj

XTGSHOW = False
if "XTGSHOW" in os.environ:
    XTGSHOW = True


SFILE1 = join(TPATH, "cubes/reek/syntseis_20000101_seismic_depth_stack.segy")
SFILE3 = join(TPATH, "cubes/reek/syntseis_20000101_seismic_depth_stack.storm")
SFILE4 = join(TPATH, "cubes/etc/ib_test_cube2.segy")

# pylint: disable=redefined-outer-name


@pytest.fixture()
def loadsfile1():
    """Fixture for loading a SFILE1"""
    return Cube(SFILE1)


def test_create():
    """Create default cube instance."""
    xcu = Cube()
    assert xcu.ncol == 5, "NCOL"
    assert xcu.nrow == 3, "NROW"
    vec = xcu.values
    xdim, _ydim, _zdim = vec.shape
    assert xdim == 5, "NX from numpy shape "


def test_segy_scanheader():
    """Scan SEGY and report header, using XTGeo internal reader."""
    Cube().scan_segy_header(SFILE1, outfile=join(TMD, "cube_scanheader"))


def test_segy_scantraces():
    """Scan and report SEGY first and last trace (internal reader)."""
    Cube().scan_segy_traces(SFILE1, outfile="TMP/cube_scantraces")


def test_segy_no_file_exception():
    with pytest.raises(xtgeo.XTGeoCLibError, match="Could not open file"):
        Cube().scan_segy_traces("not_a_file", outfile="not_relevant")


def test_storm_import():
    """Import Cube using Storm format (case Reek)."""

    acube = Cube()

    st1 = xtg.timer()
    acube.from_file(SFILE3, fformat="storm")
    elapsed = xtg.timer(st1)
    logger.info("Reading Storm format took %s", elapsed)

    assert acube.ncol == 280, "NCOL"

    vals = acube.values

    assert vals[180, 185, 4] == pytest.approx(0.117074, 0.0001)

    acube.to_file(join(TMD, "cube.rmsreg"), fformat="rms_regular")


# @skipsegyio
# @skiplargetest
def test_segy_import(loadsfile1):
    """Import SEGY using internal reader (case 1 Reek)."""

    st1 = xtg.timer()
    xcu = loadsfile1
    elapsed = xtg.timer(st1)
    logger.info("Reading with XTGEO took %s", elapsed)

    assert xcu.ncol == 408, "NCOL"

    dim = xcu.values.shape

    assert dim == (408, 280, 70), "Dimensions 3D"

    print(xcu.values.max())
    assert xcu.values.max() == pytest.approx(7.42017, 0.001)


def test_segyio_import(loadsfile1):
    """Import SEGY (case 1 Reek) via SegIO library."""

    st1 = xtg.timer()
    xcu = loadsfile1
    elapsed = xtg.timer(st1)
    logger.info("Reading with SEGYIO took %s", elapsed)

    assert xcu.ncol == 408, "NCOL"
    dim = xcu.values.shape

    assert dim == (408, 280, 70), "Dimensions 3D"
    assert xcu.values.max() == pytest.approx(7.42017, 0.001)


def test_segyio_import_export(loadsfile1):
    """Import and export SEGY (case 1 Reek) via SegIO library."""

    logger.info("Import SEGY format via SEGYIO")

    xcu = loadsfile1

    assert xcu.ncol == 408, "NCOL"
    dim = xcu.values.shape

    logger.info("Dimension is %s", dim)
    assert dim == (408, 280, 70), "Dimensions 3D"
    assert xcu.values.max() == pytest.approx(7.42017, 0.001)

    input_mean = xcu.values.mean()

    logger.info(input_mean)

    xcu.values += 200

    xcu.to_file(join(TMD, "reek_cube.segy"))

    # reread that file
    y = Cube(join(TMD, "reek_cube.segy"))

    logger.info(y.values.mean())


def test_segyio_import_export_pristine(loadsfile1):
    """Import and export as pristine SEGY (case 1 Reek) via SegIO library."""

    logger.info("Import SEGY format via SEGYIO")

    xcu = loadsfile1

    assert xcu.ncol == 408, "NCOL"
    dim = xcu.values.shape

    logger.info("Dimension is %s", dim)
    assert dim == (408, 280, 70), "Dimensions 3D"
    assert xcu.values.max() == pytest.approx(7.42017, 0.001)

    input_mean = xcu.values.mean()

    logger.info(input_mean)

    xcu.values += 200

    xcu.to_file(join(TMD, "reek_cube_pristine.segy"), pristine=True)

    # # reread that file
    # y = Cube('TMP/reek_cube_pristine.segy')

    # logger.info(y.values.mean())


def test_segyio_export_xtgeo(loadsfile1):
    """Import via SEGYIO and and export SEGY (case 1 Reek) via XTGeo."""

    logger.info("Import SEGY format via SEGYIO")

    xcu = loadsfile1

    xcu.values += 200

    xcu.to_file(join(TMD, "reek_cube_xtgeo.segy"), engine="xtgeo")

    xxcu = Cube()
    xxcu.scan_segy_header(
        join(TMD, "reek_cube_xtgeo.segy"), outfile=join(TMD, "cube_scanheader2")
    )

    xxcu.scan_segy_traces(
        join(TMD, "reek_cube_xtgeo.segy"), outfile=join(TMD, "cube_scantraces2")
    )

    # # reread that file, scan header
    # y = Cube('TMP/reek_cube_pristine.segy')

    # logger.info(y.values.mean())


def test_cube_resampling(loadsfile1):
    """Import a cube, then make a smaller and resample, then export the new"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1

    newcube = Cube(
        xori=460500,
        yori=5926100,
        zori=1540,
        xinc=40,
        yinc=40,
        zinc=5,
        ncol=200,
        nrow=100,
        nlay=100,
        rotation=incube.rotation,
        yflip=incube.yflip,
    )

    newcube.resample(incube, sampling="trilinear", outside_value=10.0)

    assert newcube.values.mean() == pytest.approx(5.3107, 0.0001)
    assert newcube.values[20, 20, 20] == pytest.approx(10.0, 0.0001)

    # newcube.to_file(join(TMD, "cube_resmaple1.segy"))


def test_cube_thinning(loadsfile1):
    """Import a cube, then make a smaller by thinning every N line"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1
    logger.info(incube)

    # thinning to evey second column and row, but not vertically
    incube.do_thinning(2, 2, 1)
    logger.info(incube)

    incube.to_file(join(TMD, "cube_thinned.segy"))

    incube2 = Cube(join(TMD, "cube_thinned.segy"))
    logger.info(incube2)


def test_cube_cropping(loadsfile1):
    """Import a cube, then make a smaller by cropping"""

    logger.info("Import SEGY format via SEGYIO")

    incube = loadsfile1

    # thinning to evey second column and row, but not vertically
    incube.do_cropping((2, 13), (10, 22), (30, 0))

    incube.to_file(join(TMD, "cube_cropped.segy"))


def test_cube_get_xy_from_ij(loadsfile1):
    """Import a cube, then report XY for a given IJ"""

    logger.info("Checking get xy from IJ")

    incube = loadsfile1

    # thinning to evey second column and row, but not vertically
    xpos, ypos = incube.get_xy_value_from_ij(0, 0, zerobased=True)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(1, 1, zerobased=False)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(0, 0, ixline=True)
    assert xpos == incube.xori
    assert ypos == incube.yori

    xpos, ypos = incube.get_xy_value_from_ij(200, 200, zerobased=True)

    assert xpos == pytest.approx(463327.8811957213, 0.01)
    assert ypos == pytest.approx(5933633.598034564, 0.01)


def test_cube_swapaxes():
    """Import a cube, do axes swapping back and forth"""

    logger.info("Import SEGY format via SEGYIO")

    incube = Cube(SFILE4)
    logger.info(incube)
    val1 = incube.values.copy()

    incube.swapaxes()
    logger.info(incube)

    incube.swapaxes()
    val2 = incube.values.copy()
    logger.info(incube)

    np.testing.assert_array_equal(val1, val2)


def test_cube_randomline():
    """Import a cube, and compute a randomline given a simple Polygon"""

    incube = Cube(SFILE4)

    poly = xtgeo.Polygons()
    poly.from_list([[778133, 6737650, 2000, 1], [776880, 6738820, 2000, 1]])

    logger.info("Generate random line...")
    hmin, hmax, vmin, vmax, random = incube.get_randomline(poly)

    assert hmin == pytest.approx(-15.7, 0.1)
    assert random.mean() == pytest.approx(-12.5, 0.1)

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(
            random,
            cmap="seismic",
            interpolation="sinc",
            extent=(hmin, hmax, vmax, vmin),
        )
        plt.axis("tight")
        plt.colorbar()
        plt.show()
