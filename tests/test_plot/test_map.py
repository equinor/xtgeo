import sys
import matplotlib.pyplot as plt

from xtgeo.plot import Map
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Polygons
from xtgeo.xyz import Points
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

TD = xtg.tmpdir
TESTPATH = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================


SFILE1 = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"
PFILE1 = "../xtgeo-testdata/polygons/reek/1/top_upper_reek_faultpoly.pol"
SFILE2 = "../xtgeo-testdata/surfaces/reek/1/reek_perm_lay1.gri"


@tsetup.skipifroxar
def test_very_basic_to_file():
    """Just test that matplotlib works, to a file."""
    plt.title("Hello world")
    plt.savefig("TMP/verysimple.png")


@tsetup.skipifroxar
def test_simple_plot():
    """Test as simple map plot only making an instance++ and plot."""

    mysurf = RegularSurface()
    mysurf.from_file(SFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)

    myplot.savefig("TMP/map_simple.png")


@tsetup.skipifroxar
def test_map_plot_with_points():
    """Test as simple map plot with underlying points."""

    mysurf = RegularSurface()
    mysurf.from_file(SFILE1)

    mypoints = Points()
    mypoints.from_surface(mysurf)

    df = mypoints.dataframe.copy()
    df = df[::20]
    mypoints.dataframe = df

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)
    myplot.plot_points(mypoints)

    myplot.savefig("TMP/map_with_points.png")


@tsetup.skipifroxar
def test_more_features_plot():
    """Map with some more features added, such as label rotation."""

    mysurf = RegularSurface()
    mysurf.from_file(SFILE1)

    myfaults = Polygons(PFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="Label rotation")
    myplot.colormap = "gist_rainbow_r"
    myplot.plot_surface(mysurf, minvalue=1250, maxvalue=2200, xlabelrotation=45)

    myplot.plot_faults(myfaults)

    myplot.savefig(TD + "/map_more1.png")


@tsetup.skipifroxar
def test_perm_logarithmic_map():
    """Map with PERM, log scale."""

    mysurf = RegularSurface(SFILE2)

    myplot = Map()
    myplot.canvas(title="PERMX normal scale")
    myplot.colormap = "gist_rainbow_r"
    myplot.plot_surface(
        mysurf, minvalue=0, maxvalue=6000, xlabelrotation=45, logarithmic=True
    )

    myplot.savefig(TD + "/permx_normal.png")
