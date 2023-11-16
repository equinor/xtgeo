from os.path import join

import xtgeo
from xtgeo.common.xtgeo_dialog import testdatafolder
from xtgeo.plot import Map

TPATH = testdatafolder

# =========================================================================
# Do tests
# =========================================================================


SFILE1 = TPATH / "surfaces/reek/1/topreek_rota.gri"
PFILE1 = TPATH / "polygons/reek/1/top_upper_reek_faultpoly.pol"
SFILE2 = TPATH / "surfaces/reek/1/reek_perm_lay1.gri"


def test_simple_plot(tmpdir, generate_plot):
    """Test as simple map plot only making an instance++ and plot."""

    mysurf = xtgeo.surface_from_file(SFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)

    if generate_plot:
        myplot.savefig(join(tmpdir, "map_simple.png"), last=True)
    else:
        myplot.close()


def test_map_plot_with_points(tmpdir, generate_plot):
    """Test as simple map plot with underlying points."""

    mysurf = xtgeo.surface_from_file(SFILE1)

    mypoints = xtgeo.points_from_surface(mysurf)

    df = mypoints.dataframe.copy()
    df = df[::20]
    mypoints.dataframe = df

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)
    myplot.plot_points(mypoints)

    if generate_plot:
        myplot.savefig(join(tmpdir, "map_with_points.png"), last=True)
    else:
        myplot.close()


def test_more_features_plot(tmpdir, generate_plot):
    """Map with some more features added, such as label rotation."""

    mysurf = xtgeo.surface_from_file(SFILE1)

    myfaults = xtgeo.polygons_from_file(PFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="Label rotation")
    myplot.colormap = "rainbow"
    myplot.plot_surface(mysurf, minvalue=1250, maxvalue=2200, xlabelrotation=45)

    myplot.plot_faults(myfaults)

    if generate_plot:
        myplot.savefig(join(tmpdir, "map_more1.png"), last=True)
    else:
        myplot.close()


def test_perm_logarithmic_map(tmpdir, generate_plot):
    """Map with PERM, log scale."""

    mysurf = xtgeo.surface_from_file(SFILE2)

    myplot = Map()
    myplot.canvas(title="PERMX normal scale")
    myplot.colormap = "rainbow"
    myplot.plot_surface(
        mysurf, minvalue=0, maxvalue=6000, xlabelrotation=45, logarithmic=True
    )

    if generate_plot:
        myplot.savefig(join(tmpdir, "permx_normal.png"), last=True)
    else:
        myplot.close()
