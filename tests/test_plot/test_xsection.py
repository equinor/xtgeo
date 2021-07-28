# -*- coding: utf-8 -*-


import glob
from os.path import join

import matplotlib.pyplot as plt
import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
from xtgeo.plot import XSection

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

USEFILE1 = TPATH / "wells/reek/1/OP_1.w"
USEFILE2 = TPATH / "surfaces/reek/1/topreek_rota.gri"
USEFILE3 = TPATH / "polygons/reek/1/mypoly.pol"
USEFILE4 = TPATH / "wells/reek/1/OP_5.w"
USEFILE5 = TPATH / "surfaces/reek/2/*.gri"
USEFILE6 = TPATH / "cubes/reek/syntseis_20000101_seismic_depth_stack.segy"

USEFILE7 = TPATH / "wells/reek/1/OP_2.w"

BIGRGRID1 = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg.roff"
BIGPROP1 = "../xtgeo-testdata-equinor/data/3dgrids/gfb/gullfaks_gg_phix.roff"
BIGWELL1 = "../xtgeo-testdata-equinor/data/wells/gfb/1/34_10-A-42.w"
BIGWELL2 = "../xtgeo-testdata-equinor/data/wells/gfb/1/34_10-A-41.w"


@pytest.mark.skipifroxar
def test_xsection_init():
    """Trigger XSection class, basically."""
    xsect = XSection()
    assert xsect.pagesize == "A4"


@pytest.mark.skipifroxar
def test_simple_plot(tmpdir, show_plot, generate_plot):
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
    cfil2 = TPATH / "etc/colortables/colfacies.txt"

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in str(cfil2)

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Manamana", subtitle="My Dear Well")

    myplot.plot_surfaces(fill=False)

    myplot.plot_well(zonelogname="Zonelog")

    if generate_plot:
        myplot.savefig(join(tmpdir, "xsect_gbf1.png"))

    if show_plot:
        myplot.show()


@pytest.mark.skipifroxar
def test_simple_plot_with_seismics(tmpdir, show_plot, generate_plot):
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
    cfil2 = TPATH / "etc/colortables/colfacies.txt"

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in str(cfil2)

    myplot.set_colortable(cfil1, colorlist=None)

    myplot.canvas(title="Plot with seismics", subtitle="Some well")

    myplot.plot_cube()
    myplot.plot_surfaces(fill=False)

    myplot.plot_well()

    myplot.plot_map()

    if generate_plot:
        myplot.savefig(join(tmpdir, "xsect_wcube.png"), last=False)

    if show_plot:
        myplot.show()


@pytest.mark.skipifroxar
@tsetup.equinor
@tsetup.bigtest
def test_xsect_larger_geogrid(show_plot):
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

    if show_plot:
        plt.figure()
        plt.imshow(arr1, cmap="rainbow", extent=(hmin1, hmax1, vmax1, vmin1))
        plt.axis("tight")
        plt.figure()
        plt.imshow(arr2, cmap="rainbow", extent=(hmin2, hmax2, vmax2, vmin2))
        plt.axis("tight")
        plt.show()


@pytest.mark.skipifroxar
def test_reek1(tmpdir, generate_plot):
    """Test XSect for a Reek well."""

    myfield = xtgeo.Polygons()
    myfield.from_file(USEFILE3, fformat="xyz")

    mywells = []
    wnames = glob.glob(str(USEFILE4))
    wnames.sort()
    for wname in wnames:
        mywell = xtgeo.Well(wname)
        mywells.append(mywell)

    logger.info("Wells are read...")

    mysurfaces = []
    surfnames = glob.glob(str(USEFILE5))
    surfnames.sort()
    for fname in surfnames:
        mysurf = xtgeo.RegularSurface()
        mysurf.from_file(fname)
        mysurfaces.append(mysurf)

    # Troll lobes
    mylobes = []
    surfnames = glob.glob(str(USEFILE5))
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

        if generate_plot:
            myplot.savefig(join(tmpdir, "xsect2a.svg"), fformat="svg", last=False)
            myplot.savefig(join(tmpdir, "xsect2a.png"), fformat="png")
