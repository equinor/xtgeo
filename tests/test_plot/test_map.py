import sys
from os.path import join

import pytest

from xtgeo.common import XTGeoDialog
from xtgeo.plot import Map
from xtgeo.surface import RegularSurface
from xtgeo.xyz import Points, Polygons

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================


SFILE1 = TPATH / "surfaces/reek/1/topreek_rota.gri"
PFILE1 = TPATH / "polygons/reek/1/top_upper_reek_faultpoly.pol"
SFILE2 = TPATH / "surfaces/reek/1/reek_perm_lay1.gri"


@pytest.mark.skipifroxar
def test_simple_plot(tmpdir):
    """Test as simple map plot only making an instance++ and plot."""

    mysurf = RegularSurface()
    mysurf.from_file(SFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)

    myplot.savefig(join(tmpdir, "map_simple.png"))


@pytest.mark.skipifroxar
def test_map_plot_with_points(tmpdir):
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

    myplot.savefig(join(tmpdir, "map_with_points.png"))


@pytest.mark.skipifroxar
def test_more_features_plot(tmpdir):
    """Map with some more features added, such as label rotation."""

    mysurf = RegularSurface()
    mysurf.from_file(SFILE1)

    myfaults = Polygons(PFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="Label rotation")
    myplot.colormap = "rainbow"
    myplot.plot_surface(mysurf, minvalue=1250, maxvalue=2200, xlabelrotation=45)

    myplot.plot_faults(myfaults)

    myplot.savefig(join(tmpdir, "map_more1.png"))


@pytest.mark.skipifroxar
def test_perm_logarithmic_map(tmpdir):
    """Map with PERM, log scale."""

    mysurf = RegularSurface(SFILE2)

    myplot = Map()
    myplot.canvas(title="PERMX normal scale")
    myplot.colormap = "rainbow"
    myplot.plot_surface(
        mysurf, minvalue=0, maxvalue=6000, xlabelrotation=45, logarithmic=True
    )

    myplot.savefig(join(tmpdir, "permx_normal.png"))
