import pathlib

import pytest

import xtgeo
from xtgeo.common.log import functimer

REEKROOT = pathlib.Path("3dgrids/reek/REEK")
WELL1 = pathlib.Path("wells/reek/1/OP_1.w")
FENCE1 = pathlib.Path("polygons/reek/1/fence.pol")
FENCE2 = pathlib.Path("polygons/reek/1/minifence.pol")


def test_randomline_fence_from_well(show_plot, testdata_path):
    """Import ROFF grid with props and make fences"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO"]
    )
    wll = xtgeo.well_from_file(testdata_path / WELL1, zonelogname="Zonelog")

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=100)

    # assert fspec.get_dataframe()[fspec.dhname][4] == pytest.approx(12.6335, abs=0.001)

    @functimer(output="print")
    def get_polyline():
        """Get the fence polyline as a numpy array."""
        return wll.get_fence_polyline(sampling=1, nextend=10, asnumpy=True, tvdmin=1200)

    fspec = get_polyline()

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow

    @functimer(output="print")
    def get_randomline():
        hmin, hmax, vmin, vmax, por = grd.get_randomline(
            fspec, "PORO", zmin=1550, zmax=1660, zincrement=0.1
        )
        return hmin, hmax, vmin, vmax, por

    hmin, hmax, vmin, vmax, por = get_randomline()
    hmin, hmax, vmin, vmax, por = get_randomline()

    @functimer(output="print")
    def showplot():
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()

        return plt

    if show_plot:
        plt = showplot()
        plt.show()


def test_randomline_fence_from_polygon(show_plot, testdata_path):
    """Import ROFF grid with props and make fence from polygons"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"]
    )
    fence = xtgeo.polygons_from_file(testdata_path / FENCE1)

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert fspec.get_dataframe()[fspec.dhname][4] == pytest.approx(10, abs=1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1680, zmax=1750, zincrement=0.5
    )

    hmin, hmax, vmin, vmax, perm = grd.get_randomline(
        fspec, "PERMX", zmin=1680, zmax=1750, zincrement=0.5
    )

    if show_plot:
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


def test_randomline_fence_calczminzmax(show_plot, testdata_path):
    """Import ROFF grid with props and make fence from polygons, zmin/zmax auto"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"]
    )
    fence = xtgeo.polygons_from_file(testdata_path / FENCE1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    hmin, hmax, vmin, vmax, xx = grd.get_randomline(fspec, "PORO", zmin=None, zmax=None)
    assert vmin == pytest.approx(1547.98, abs=0.01)
    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(xx, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()
