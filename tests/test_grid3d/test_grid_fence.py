from os.path import join

import pytest

import xtgeo

from .ecl_run_fixtures import *  # noqa: F401, F403


def test_randomline_fence_from_well(show_plot, testpath, reek_run):
    grd = reek_run.grid_with_props(initprops=["PORO"])
    wll = xtgeo.Well(
        join(testpath, "wells", "reek", "1", "OP_1.w"), zonelogname="Zonelog"
    )

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=1200)

    assert fspec.dataframe[fspec.dhname][4] == pytest.approx(12.6335, abs=0.001)

    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=True, tvdmin=1200)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1600, zmax=1700, zincrement=1.0
    )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_from_polygon(show_plot, testpath, reek_run):
    grd = reek_run.grid_with_props(initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(join(testpath, "polygons", "reek", "1", "fence.pol"))

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert fspec.dataframe[fspec.dhname][4] == pytest.approx(10, abs=1)

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


def test_randomline_fence_calczminzmax(testpath, reek_run):
    grd = reek_run.grid_with_props(initprops=["PORO", "PERMX"])
    fence = xtgeo.Polygons(join(testpath, "polygons/reek/1/fence.pol"))

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=None, zmax=None
    )
    assert vmin == pytest.approx(1548.10098, abs=0.0001)
