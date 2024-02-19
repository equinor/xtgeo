import pathlib

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.plot import Map

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

SFILE1 = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
PFILE1 = pathlib.Path("polygons/reek/1/top_upper_reek_faultpoly.pol")
SFILE2 = pathlib.Path("surfaces/reek/1/reek_perm_lay1.gri")


def test_simple_plot(tmp_path, generate_plot, testdata_path):
    """Test as simple map plot only making an instance++ and plot."""

    mysurf = xtgeo.surface_from_file(testdata_path / SFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)

    if generate_plot:
        myplot.savefig(tmp_path / "map_simple.png", last=True)
    else:
        myplot.close()


def test_map_plot_with_points(tmp_path, generate_plot, testdata_path):
    """Test as simple map plot with underlying points."""

    mysurf = xtgeo.surface_from_file(testdata_path / SFILE1)

    mypoints = xtgeo.points_from_surface(mysurf)

    df = mypoints.get_dataframe()
    df = df[::20]
    mypoints.set_dataframe(df)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="My o my")
    myplot.colormap = "gist_ncar"
    myplot.plot_surface(mysurf)
    myplot.plot_points(mypoints)

    if generate_plot:
        myplot.savefig(tmp_path / "map_with_points.png", last=True)
    else:
        myplot.close()


def test_more_features_plot(tmp_path, generate_plot, testdata_path):
    """Map with some more features added, such as label rotation."""

    mysurf = xtgeo.surface_from_file(testdata_path / SFILE1)

    myfaults = xtgeo.polygons_from_file(testdata_path / PFILE1)

    # just make the instance, with a lot of defaults behind the scene
    myplot = Map()
    myplot.canvas(title="Label rotation")
    myplot.colormap = "rainbow"
    myplot.plot_surface(mysurf, minvalue=1250, maxvalue=2200, xlabelrotation=45)

    myplot.plot_faults(myfaults)

    if generate_plot:
        myplot.savefig(tmp_path / "map_more1.png", last=True)
    else:
        myplot.close()


def test_perm_logarithmic_map(tmp_path, generate_plot, testdata_path):
    """Map with PERM, log scale."""

    mysurf = xtgeo.surface_from_file(testdata_path / SFILE2)

    myplot = Map()
    myplot.canvas(title="PERMX normal scale")
    myplot.colormap = "rainbow"
    myplot.plot_surface(
        mysurf, minvalue=0, maxvalue=6000, xlabelrotation=45, logarithmic=True
    )

    if generate_plot:
        myplot.savefig(tmp_path / "permx_normal.png", last=True)
    else:
        myplot.close()
