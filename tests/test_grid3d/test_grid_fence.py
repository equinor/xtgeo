# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os

import xtgeo
import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

REEKROOT = "../xtgeo-testdata/3dgrids/reek/REEK"
WELL1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
FENCE1 = "../xtgeo-testdata/polygons/reek/1/fence.pol"
FENCE2 = "../xtgeo-testdata/polygons/reek/1/minifence.pol"

# =============================================================================
# Do tests
# =============================================================================


def test_randomline_fence_from_well():
    """Import ROFF grid with props and make fences"""

    grd = xtgeo.Grid(REEKROOT, fformat="eclipserun", initprops=["PORO"])
    wll = xtgeo.Well(WELL1, zonelogname="Zonelog")

    print(grd.describe(details=True))

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)
    print(fspec.dataframe)

    tsetup.assert_almostequal(fspec.dataframe[fspec.dhname][4], 12.6335, 0.001)
    logger.info(fspec.dataframe)

    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1200, zmax=1700, zincrement=1.0
    )

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(
            por, cmap="seismic", interpolation="sinc", extent=(hmin, hmax, vmax, vmin)
        )
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_from_polygon():
    """Import ROFF grid with props and make fence from polygons"""

    grd = xtgeo.Grid(REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(FENCE1)

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    tsetup.assert_almostequal(fspec.dataframe[fspec.dhname][4], 10, 1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    logger.info("Getting randomline...")
    timer1 = xtg.timer()
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1680, zmax=1750, zincrement=0.5
    )
    logger.info("Getting randomline... took {0:5.3f} secs".format(xtg.timer(timer1)))

    timer1 = xtg.timer()
    hmin, hmax, vmin, vmax, perm = grd.get_randomline(
        fspec, "PERMX", zmin=1680, zmax=1750, zincrement=0.5
    )
    logger.info(
        "Getting randomline (2 time)... took {0:5.3f} secs".format(xtg.timer(timer1))
    )

    if XTGSHOW:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.figure()
        plt.imshow(perm, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()
