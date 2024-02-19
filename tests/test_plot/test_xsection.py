import glob
import pathlib

import xtgeo
from xtgeo.plot import XSection

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

USEFILE1 = pathlib.Path("wells/reek/1/OP_1.w")
USEFILE2 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
USEFILE3 = pathlib.Path("polygons/reek/1/mypoly.pol")
USEFILE4 = pathlib.Path("wells/reek/1/OP_5.w")
USEFILE5 = pathlib.Path("surfaces/reek/2/*.gri")
USEFILE6 = pathlib.Path("cubes/reek/syntseis_20000101_seismic_depth_stack.segy")

USEFILE7 = pathlib.Path("wells/reek/1/OP_2.w")


def test_xsection_init():
    """Trigger XSection class, basically."""
    xsect = XSection()
    assert xsect.pagesize == "A4"


def test_simple_plot(tmp_path, show_plot, generate_plot, testdata_path):
    """Test as simple XSECT plot."""

    mywell = xtgeo.well_from_file(testdata_path / USEFILE4)

    mysurfaces = []
    mysurf = xtgeo.surface_from_file(testdata_path / USEFILE2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = f"Surface_{i}"
        mysurfaces.append(xsurf)

    myplot = XSection(zmin=1500, zmax=1800, well=mywell, surfaces=mysurfaces)

    # set the color table, from file
    clist = [0, 1, 222, 3, 5, 7, 3, 12, 11, 10, 9, 8]
    cfil1 = "xtgeo"
    cfil2 = testdata_path / pathlib.Path("etc/colortables/colfacies.txt")

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in str(cfil2)

    myplot.colormap = cfil1

    myplot.canvas(title="Manamana", subtitle="My Dear Well")

    myplot.plot_surfaces(fill=False)

    myplot.plot_well(zonelogname="Zonelog")

    if show_plot:
        myplot.show()

    if generate_plot:
        myplot.savefig(tmp_path / "xsect_gbf1.png", last=True)
    else:
        myplot.close()


def test_simple_plot_with_seismics(tmp_path, show_plot, generate_plot, testdata_path):
    """Test as simple XSECT plot with seismic backdrop."""

    mywell = xtgeo.well_from_file(testdata_path / USEFILE7)
    mycube = xtgeo.cube_from_file(testdata_path / USEFILE6)

    mysurfaces = []
    mysurf = xtgeo.surface_from_file(testdata_path / USEFILE2)

    for i in range(10):
        xsurf = mysurf.copy()
        xsurf.values = xsurf.values + i * 20
        xsurf.name = f"Surface_{i}"
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
    cfil2 = testdata_path / pathlib.Path("etc/colortables/colfacies.txt")

    assert 222 in clist
    assert "xtgeo" in cfil1
    assert "colfacies" in str(cfil2)

    myplot.colormap = cfil1

    myplot.canvas(title="Plot with seismics", subtitle="Some well")

    myplot.plot_cube()
    myplot.plot_surfaces(fill=False)

    myplot.plot_well()

    myplot.plot_map()

    if generate_plot:
        myplot.savefig(tmp_path / "xsect_wcube.png", last=False)

    if show_plot:
        myplot.show()

    myplot.close()


def test_multiple_subplots(tmp_path, show_plot, generate_plot, testdata_path):
    """Test as simple XSECT plot."""

    mywell = xtgeo.well_from_file(testdata_path / USEFILE4)
    mysurf = xtgeo.surface_from_file(testdata_path / USEFILE2)

    myplot = XSection(zmin=1500, zmax=1800, well=mywell, surfaces=[mysurf])

    for i in range(4):
        myplot.canvas(title=str(i), subtitle="My Dear Well")

    myplot.close()


def test_reek1(tmp_path, generate_plot, testdata_path):
    """Test XSect for a Reek well."""

    myfield = xtgeo.polygons_from_file(testdata_path / USEFILE3, fformat="xyz")

    mywells = []
    wnames = glob.glob(str(testdata_path / USEFILE4))
    wnames.sort()
    for wname in wnames:
        mywell = xtgeo.well_from_file(wname)
        mywells.append(mywell)

    logger.info("Wells are read...")

    mysurfaces = []
    surfnames = glob.glob(str(testdata_path / USEFILE5))
    surfnames.sort()
    for fname in surfnames:
        mysurf = xtgeo.surface_from_file(fname)
        mysurfaces.append(mysurf)

    # Troll lobes
    mylobes = []
    surfnames = glob.glob(str(testdata_path / USEFILE5))
    surfnames.sort()
    for fname in surfnames:
        mysurf = xtgeo.surface_from_file(fname)
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
            myplot.savefig(tmp_path / "xsect2a.svg", fformat="svg", last=False)
            myplot.savefig(tmp_path / "xsect2a.png", fformat="png")
        else:
            myplot.close()
