# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os
from os.path import join
import glob

import matplotlib.pyplot as plt

import xtgeo
from xtgeo.plot import XSection
import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath

XTGSHOW = False
if "XTG_SHOW" in os.environ:
    XTGSHOW = True

logger.info("Use env variable XTG_SHOW to show interactive plots to screen")

# =========================================================================
# Do tests
# =========================================================================

USEFILE1 = "../xtgeo-testdata/wells/reek/1/OP_1.w"
USEFILE2 = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"
USEFILE3 = "../xtgeo-testdata/polygons/reek/1/mypoly.pol"
USEFILE4 = "../xtgeo-testdata/wells/reek/1/OP_5.w"
USEFILE5 = "../xtgeo-testdata/surfaces/reek/2/*.gri"
USEFILE6 = (
    "../xtgeo-testdata/cubes/reek/" + "syntseis_20000101_seismic_depth_stack.segy"
)

USEFILE7 = "../xtgeo-testdata/wells/reek/1/OP_2.w"

BIGRGRID1 = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg.roff"
BIGPROP1 = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg_phix.roff"
BIGWELL1 = "../xtgeo-testdata-equinor/data/wells/gfb/1/34_10-A-42.w"
BIGWELL2 = "../xtgeo-testdata-equinor/data/wells/gfb/1/34_10-A-41.w"


@tsetup.skipifroxar
def test_very_basic():
    """Just test that matplotlib works."""
    assert "matplotlib" in str(plt)

    plt.title("Hello world")
    plt.savefig(join(TMPD, "helloworld1.png"))
    plt.savefig(join(TMPD, "helloworld1.svg"))
    if XTGSHOW:
        plt.show()
    logger.info("Very basic plotting")
    plt.close()


@tsetup.skipifroxar
def test_xsection_init():
    """Trigger XSection class, basically."""
    xsect = XSection()
    assert xsect.pagesize == "A4"


@tsetup.skipifroxar
def test_simple_plot():
    """Test as simple XSECT plot."""

    mywell = xtgeo.Well(USEFILE4)

    mysurfaces = []
    mysurf = xtgeo.RegularSurface()
    mysurf.from_file(USEFILE2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = "Surface_{}".format(i)
        mysurfaces.append(xsurf)

    myplot = XSection(zmin=1500, zmax=1800, well=mywell, surfaces=mysurfaces)

    # set the color table, from file
    clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    cfil1 = "xtgeo"
    cfil2 = "../xtgeo-testdata/etc/colortables/colfacies.txt"

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in cfil2

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Manamana", subtitle="My Dear Well")

    myplot.plot_surfaces(fill=False)

    myplot.plot_well(zonelogname="Zonelog")

    # myplot.plot_map()

    myplot.savefig(join(TMPD, "xsect_gbf1.png"))

    if XTGSHOW:
        print("Show plot")
        myplot.show()


@tsetup.skipifroxar
def test_simple_plot_with_seismics():
    """Test as simple XSECT plot with seismic backdrop."""

    mywell = xtgeo.Well(USEFILE7)
    mycube = xtgeo.Cube(USEFILE6)

    mysurfaces = []
    mysurf = xtgeo.RegularSurface()
    mysurf.from_file(USEFILE2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = "Surface_{}".format(i)
        mysurfaces.append(xsurf)

    myplot = XSection(
        zmin=1000,
        zmax=1900,
        well=mywell,
        surfaces=mysurfaces,
        cube=mycube,
        sampling=10,
        nextend=2,
    )

    # set the color table, from file
    clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    cfil1 = "xtgeo"
    cfil2 = "../xtgeo-testdata/etc/colortables/colfacies.txt"

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in cfil2

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Plot with seismics", subtitle="Some well")

    myplot.plot_cube()
    myplot.plot_surfaces(fill=False)

    myplot.plot_well()

    myplot.plot_map()

    myplot.savefig(join(TMPD, "xsect_wcube.png"), last=False)

    if XTGSHOW:
        myplot.show()


@tsetup.skipifroxar
@tsetup.equinor
@tsetup.bigtest
def test_xsect_larger_geogrid():
    """Test a larger xsection"""

    mygrid = xtgeo.Grid(BIGRGRID1)
    poro = xtgeo.GridProperty(BIGPROP1)
    mywell1 = xtgeo.Well(BIGWELL1)
    mywell2 = xtgeo.Well(BIGWELL2)

    fence1 = mywell1.get_fence_polyline(sampling=5, tvdmin=1750, asnumpy=True)

    (hmin1, hmax1, vmin1, vmax1, arr1) = mygrid.get_randomline(
        fence1, poro, zmin=1750, zmax=2100, zincrement=0.2
    )

    fence2 = mywell2.get_fence_polyline(sampling=5, tvdmin=1500, asnumpy=True)

    (hmin2, hmax2, vmin2, vmax2, arr2) = mygrid.get_randomline(
        fence2, poro, zmin=1500, zmax=1850, zincrement=0.2
    )

    if XTGSHOW:
        plt.figure()
        plt.imshow(arr1, cmap="rainbow", extent=(hmin1, hmax1, vmax1, vmin1))
        plt.axis("tight")
        plt.figure()
        plt.imshow(arr2, cmap="rainbow", extent=(hmin2, hmax2, vmax2, vmin2))
        plt.axis("tight")
        plt.show()
    # myplot = XSection(
    #     zmin=1000, zmax=1900, well=mywell, surfaces=mysurfaces, cube=mycube
    # )

    # # set the color table, from file
    # clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    # cfil1 = "xtgeo"
    # cfil2 = "../xtgeo-testdata/etc/colortables/colfacies.txt"

    # assert 222 in clist
    # assert "xtgeo" in cfil1
    # assert "colfacies" in cfil2

    # myplot.set_colortable(cfil1, colorlist=None)

    # myplot.canvas(title="Manamana", subtitle="My Dear Well")

    # myplot.plot_cube()
    # myplot.plot_surfaces(fill=False)

    # myplot.plot_well()

    # myplot.plot_map()

    # myplot.savefig(join(TMPD, "xsect_wcube.png"), last=False)

    # if XTGSHOW:
    #     myplot.show()


@tsetup.skipifroxar
def test_reek1():
    """Test XSect for a Reek well."""

    myfield = xtgeo.Polygons()
    myfield.from_file(USEFILE3, fformat="xyz")

    mywells = []
    wnames = glob.glob(USEFILE4)
    wnames.sort()
    for wname in wnames:
        mywell = xtgeo.Well(wname)
        mywells.append(mywell)

    logger.info("Wells are read...")

    mysurfaces = []
    surfnames = glob.glob(USEFILE5)
    surfnames.sort()
    for fname in surfnames:
        mysurf = xtgeo.RegularSurface()
        mysurf.from_file(fname)
        mysurfaces.append(mysurf)

    # Troll lobes
    mylobes = []
    surfnames = glob.glob(USEFILE5)
    surfnames.sort()
    for fname in surfnames:
        mysurf = xtgeo.RegularSurface()
        mysurf.from_file(fname)
        mylobes.append(mysurf)

    for wo in mywells:
        myplot = XSection(
            zmin=1500,
            zmax=1700,
            well=wo,
            surfaces=mysurfaces,
            zonelogshift=-1,
            outline=myfield,
        )

        myplot.canvas(
            title=wo.truewellname,
            subtitle="Before my corrections",
            infotext="Heisan sveisan",
            figscaling=1.2,
        )

        # myplot.colortable('xtgeo')

        myplot.plot_surfaces(fill=True)

        myplot.plot_surfaces(
            surfaces=mylobes,
            fill=False,
            linewidth=4,
            legendtitle="Lobes",
            fancyline=True,
        )

        myplot.plot_well(zonelogname="Zonelog")

        myplot.plot_wellmap()

        myplot.plot_map()

        myplot.savefig(join(TMPD, "xsect2a.svg"), fformat="svg", last=False)
        myplot.savefig(join(TMPD, "xsect2a.png"), fformat="png")
