import pathlib

import numpy as np
import numpy.ma as ma
import pytest
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

RPATH1 = pathlib.Path("surfaces/reek")
RPATH3 = pathlib.Path("surfaces/etc")
RPATH2 = pathlib.Path("cubes/reek")
RPATH4 = pathlib.Path("cubes/etc")
RTOP1 = RPATH1 / "1/topreek_rota.gri"
RBAS1 = RPATH1 / "1/basereek_rota.gri"
RBAS2 = RPATH1 / "1/basereek_rota_v2.gri"
RSGY1 = RPATH2 / "syntseis_20000101_seismic_depth_stack.segy"

XTOP1 = RPATH3 / "ib_test_horizon2.gri"
XCUB1 = RPATH4 / "ib_test_cube2.segy"
XCUB2 = RPATH4 / "cube_w_deadtraces.segy"


@pytest.fixture()
def load_cube_rsgy1(testdata_path):
    """Loading test cube (pytest setup fixture)"""
    return xtgeo.cube_from_file(testdata_path / RSGY1)


def test_get_surface_from_cube(load_cube_rsgy1):
    """Construct a constant surface from cube."""

    cube = load_cube_rsgy1

    surf = xtgeo.surface_from_cube(cube, 1999.0)

    assert surf.xinc == cube.xinc
    assert surf.nrow == cube.nrow
    assert surf.values.mean() == pytest.approx(1999.0, abs=0.00001)


def test_slice_nearest_snapxy(tmp_path, load_cube_rsgy1, generate_plot):
    """Slice a cube with a surface, nearest node, snapxy, algorithm 1 + 2"""

    kube = load_cube_rsgy1

    xs1 = xtgeo.surface_from_cube(kube, 1670)
    xs2 = xtgeo.surface_from_cube(kube, 1670)

    xs1.slice_cube(kube, algorithm=1, snapxy=True)

    xs2.slice_cube(kube, algorithm=2, snapxy=True)

    if generate_plot:
        xs1.quickplot(
            filename=tmp_path / "surf_slice_cube_near_snapxy_v1.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, snapxy, algorithm 1",
        )

        xs2.quickplot(
            filename=tmp_path / "surf_slice_cube_near_snapxy_v2.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, snapxy, algorithm 2",
        )

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)


def test_slice_trilinear_snapxy(tmp_path, load_cube_rsgy1, generate_plot):
    """Slice a cube with a surface, trilinear, snapxy, algorithm 1 + 2"""

    kube = load_cube_rsgy1

    xs1 = xtgeo.surface_from_cube(kube, 1670)
    xs2 = xtgeo.surface_from_cube(kube, 1670)

    xs1.slice_cube(kube, algorithm=1, snapxy=True, sampling="trilinear")

    xs2.slice_cube(kube, algorithm=2, snapxy=True, sampling="trilinear")

    if generate_plot:
        xs1.quickplot(
            filename=tmp_path / "surf_slice_cube_tri_snapxy_v1.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: trilinear, snapxy, algorithm 1",
        )

        xs2.quickplot(
            filename=tmp_path / "surf_slice_cube_tri_snapxy_v2.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: trilinear, snapxy, algorithm 2",
        )

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)


def test_slice_nearest_nosnapxy(tmp_path, load_cube_rsgy1, generate_plot):
    """Slice a cube with a surface, nearest node, algorithm 1 + 2, other map layout"""

    kube = load_cube_rsgy1

    # kube.swapaxes()

    xs1 = xtgeo.RegularSurface(
        yori=5927600, xori=457000, xinc=50, yinc=50, ncol=200, nrow=220, values=1670
    )
    xs2 = xs1.copy()

    xs1.slice_cube(kube, algorithm=1, snapxy=False)

    xs2.slice_cube(kube, algorithm=2, snapxy=False)

    if generate_plot:
        xs1.quickplot(
            filename=tmp_path / "surf_slice_cube_near_nosnapxy_v1.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, nosnapxy, algorithm 1",
        )

        xs2.quickplot(
            filename=tmp_path / "surf_slice_cube_near_nosnapxy_v2.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, nosnapxy, algorithm 2",
        )

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)


def test_slice_trilinear_nosnapxy(tmp_path, load_cube_rsgy1, generate_plot):
    """Slice a cube with a surface, nearest node, algorithm 1 + 2, other map layout"""

    kube = load_cube_rsgy1

    # kube.swapaxes()

    xs1 = xtgeo.RegularSurface(
        yori=5927600, xori=457000, xinc=50, yinc=50, ncol=200, nrow=220, values=1670
    )
    xs2 = xs1.copy()

    xs1.slice_cube(kube, algorithm=1, snapxy=False, sampling="trilinear")

    xs2.slice_cube(kube, algorithm=2, snapxy=False, sampling="trilinear")

    if generate_plot:
        xs1.quickplot(
            filename=tmp_path / "surf_slice_cube_tri_nosnapxy_v1.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: trilinear, nosnapxy, algorithm 1",
        )

        xs2.quickplot(
            filename=tmp_path / "surf_slice_cube_tri_nosnapxy_v2.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: trilinear, nosnapxy, algorithm 2",
        )

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)


def test_slice_nearest(tmp_path, load_cube_rsgy1, generate_plot, testdata_path):
    """Slice a cube with a surface, nearest node, algorithm 1"""

    xs = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs.to_file(tmp_path / "surf_slice_cube_initial.gri")

    kube = load_cube_rsgy1

    xs.slice_cube(kube, algorithm=1)

    xs.to_file(tmp_path / "surf_slice_cube_v1.gri", fformat="irap_binary")

    if generate_plot:
        xs.quickplot(
            filename=tmp_path / "surf_slice_cube_near_v1.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, algorithm 1",
        )


def test_slice_nearest_v2(tmp_path, load_cube_rsgy1, generate_plot, testdata_path):
    """Slice a cube with a surface, nearest node, algorithm 2."""

    xs = xtgeo.surface_from_file(testdata_path / RTOP1)

    kube = load_cube_rsgy1

    xs.slice_cube(kube, algorithm=2)

    xs.to_file(tmp_path / "surf_slice_cube_alg2.gri", fformat="irap_binary")

    if generate_plot:
        xs.quickplot(
            filename=tmp_path / "surf_slice_cube_alg2.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest",
        )


def test_slice_various_reek(tmp_path, load_cube_rsgy1, generate_plot, testdata_path):
    """Slice a cube with a surface, both nearest node and interpol, Reek."""

    xs = xtgeo.surface_from_file(testdata_path / RTOP1)

    kube = load_cube_rsgy1

    xs.slice_cube(kube)

    xs.to_file(tmp_path / "surf_slice_cube_reek_interp.gri")

    if generate_plot:
        xs.quickplot(
            filename=tmp_path / "surf_slice_cube_reek_interp.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest",
        )

    # trilinear interpolation:

    xs = xtgeo.surface_from_file(testdata_path / RTOP1)

    xs.slice_cube(kube, sampling="trilinear")

    xs.to_file(tmp_path / "surf_slice_cube_reek_trilinear.gri")

    if generate_plot:
        xs.quickplot(
            filename=tmp_path / "surf_slice_cube_reek_trilinear.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: trilinear",
        )


def test_slice_attr_window_max(load_cube_rsgy1, testdata_path):
    """Slice a cube within a window, get max, using trilinear interpol."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)

    kube = load_cube_rsgy1

    ret = xs1.slice_cube_window(
        kube, attribute="max", sampling="trilinear", algorithm=2
    )
    assert ret is None
    assert xs1.values.mean() == pytest.approx(0.08619, abs=0.001)

    # one attribute but in a list context shall return a dict
    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    ret = xs1.slice_cube_window(
        kube, attribute=["max"], sampling="trilinear", algorithm=2
    )
    assert isinstance(ret, dict)

    assert ret["max"].values.mean() == pytest.approx(0.08619, abs=0.001)


def test_slice_attr_window_max_w_plotting(
    tmp_path, load_cube_rsgy1, generate_plot, testdata_path
):
    """Slice a cube within a window, get max/min etc, using trilinear
    interpol and plotting."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xs1.copy()
    xs3 = xs1.copy()

    kube = load_cube_rsgy1

    xs1.slice_cube_window(kube, attribute="min", sampling="trilinear")

    if generate_plot:
        xs1.quickplot(
            filename=tmp_path / "surf_slice_cube_window_min.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek Minimum",
            infotext="Method: trilinear, window",
        )

    xs2.slice_cube_window(
        kube, attribute="max", sampling="trilinear", showprogress=True
    )

    if generate_plot:
        xs2.quickplot(
            filename=tmp_path / "surf_slice_cube_window_max.png",
            colormap="seismic",
            minmax=(-1, 1),
            title="Reek Maximum",
            infotext="Method: trilinear, window",
        )

    xs3.slice_cube_window(kube, attribute="rms", sampling="trilinear")

    if generate_plot:
        xs3.quickplot(
            filename=tmp_path / "surf_slice_cube_window_rms.png",
            colormap="jet",
            minmax=(0, 1),
            title="Reek rms (root mean square)",
            infotext="Method: trilinear, window",
        )


def test_cube_attr_mean_two_surfaces(
    tmp_path, load_cube_rsgy1, generate_plot, testdata_path
):
    """Get cube attribute (mean) between two surfaces."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xtgeo.surface_from_file(testdata_path / RBAS1)

    kube = load_cube_rsgy1

    xss = xs1.copy()
    xss.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="mean", sampling="trilinear"
    )

    xss.to_file(tmp_path / "surf_slice_cube_2surf_meantri.gri")

    if generate_plot:
        xss.quickplot(
            filename=tmp_path / "surf_slice_cube_2surf_mean.png",
            colormap="jet",
            title="Reek two surfs mean",
            minmax=(-0.1, 0.1),
            infotext="Method: trilinear, 2 surfs",
        )


def test_cube_attr_rms_two_surfaces_compare_window(load_cube_rsgy1, testdata_path):
    """Get cube attribute (rms) between two surfaces, and compare with
    window."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xs1.copy()
    xs2.values += 30

    kube = load_cube_rsgy1

    xss1 = xs1.copy()
    xss1.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="rms", sampling="trilinear"
    )

    xss2 = xs1.copy()
    xss2.values += 15
    xss2.slice_cube_window(kube, zrange=15, attribute="rms", sampling="trilinear")

    assert xss1.values.mean() == xss2.values.mean()


def test_cube_attr_rms_two_surfaces_compare_window_show(
    tmp_path, load_cube_rsgy1, generate_plot, testdata_path
):
    """Get cube attribute (rms) between two surfaces, and compare with
    window, and show plots."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xs1.copy()
    xs2.values += 30

    kube = load_cube_rsgy1

    xss1 = xs1.copy()
    xss1.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="rms", sampling="trilinear"
    )

    if generate_plot:
        xss1.quickplot(
            filename=tmp_path / "surf_slice_cube_2surf_rms1.png",
            colormap="jet",
            minmax=[0, 0.5],
            # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
            infotext="Method: trilinear, 2 surfs, 30ms apart",
        )

    print(f"\n\n{'=' * 100}\n")

    xss2 = xs1.copy()
    xss2.values += 15
    xss2.slice_cube_window(kube, zrange=15, attribute="rms", sampling="trilinear")

    if generate_plot:
        xss2.quickplot(
            filename=tmp_path / "surf_slice_cube_2surf_rms2.png",
            colormap="jet",
            minmax=[0, 0.5],
            # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
            infotext="Method: trilinear, 2 surfs, +- 15ms window",
        )

    assert xss1.values.mean() == xss2.values.mean()


def test_cube_attr_mean_two_surfaces_with_zeroiso(
    tmp_path, load_cube_rsgy1, generate_plot, testdata_path
):
    """Get cube attribute between two surfaces with partly zero isochore."""

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xtgeo.surface_from_file(testdata_path / RBAS2)

    kube = load_cube_rsgy1

    xss = xs1.copy()
    xss.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="mean", sampling="trilinear"
    )

    xss.to_file(tmp_path / "surf_slice_cube_2surf_meantri.gri")

    if generate_plot:
        xss.quickplot(
            filename=tmp_path / "surf_slice_cube_2surf_mean_v2.png",
            colormap="jet",
            title="Reek two surfs mean",
            minmax=(-0.1, 0.1),
            infotext="Method: trilinear, 2 surfs, partly zero isochore",
        )


def test_cube_slice_auto4d_data(tmp_path, generate_plot, testdata_path):
    """Get cube slice aka Auto4D input, with synthetic/scrambled data"""

    xs1 = xtgeo.surface_from_file(testdata_path / XTOP1, fformat="gri")
    xs1.describe()

    xs1out = tmp_path / "XTOP1.ijxyz"
    xs1.to_file(xs1out, fformat="ijxyz")

    xs2 = xtgeo.surface_from_file(xs1out, fformat="ijxyz")

    np.testing.assert_allclose(xs1.values, xs2.values, atol=0.0001)

    kube1 = xtgeo.cube_from_file(testdata_path / XCUB1)
    kube1.describe()

    assert xs2.nactive == 10830

    xs2.slice_cube_window(kube1, sampling="trilinear", mask=True, attribute="max")

    xs2out1 = tmp_path / "XTOP2_sampled_from_cube.ijxyz"
    xs2out2 = tmp_path / "XTOP2_sampled_from_cube.gri"
    xs2out3 = tmp_path / "XTOP2_sampled_from_cube.png"

    xs2.to_file(xs2out1, fformat="ijxyz")
    xs2.to_file(xs2out2)

    assert xs2.nactive == 3275  # 3320  # shall be fewer cells

    if generate_plot:
        xs2.quickplot(
            filename=xs2out3,
            colormap="seismic",
            title="Auto4D Test",
            minmax=(0, 12000),
            infotext="Method: max",
        )


def test_cube_slice_w_ignore_dead_traces_nearest(
    tmp_path, generate_plot, testdata_path
):
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, ignore dead traces."""

    cube1 = xtgeo.cube_from_file(testdata_path / XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.1)

    cells = ((18, 12), (20, 2), (0, 4))

    surf1.slice_cube(cube1, deadtraces=False)
    plotfile = tmp_path / "slice_nea1.png"
    title = "Cube with dead traces; nearest; use just values as is"
    if generate_plot:
        surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == pytest.approx(
            cube1.values[icell, jcell, 0], abs=0.01
        )
    assert ma.count_masked(surf1.values) == 0  # shall be no masked cells

    # swap surface
    surf2 = surf1.copy()
    surf2.values = 1000.1

    surf2.swapaxes()

    surf2.slice_cube(cube1, deadtraces=False)

    assert surf2.values.mean() == pytest.approx(surf1.values.mean(), rel=0.001)

    # swap surface and cube
    surf2 = surf1.copy()
    surf2.values = 1000.1
    surf2.swapaxes()

    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, deadtraces=False)
    assert surf2.values.mean() == pytest.approx(surf1.values.mean(), rel=0.001)

    # swap cube only
    surf2 = surf1.copy()
    surf2.values = 1000.1

    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.slice_cube(cube2, deadtraces=False)
    assert surf2.values.mean() == pytest.approx(surf1.values.mean(), rel=0.001)


def test_cube_slice_w_dead_traces_nearest(tmp_path, generate_plot, testdata_path):
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, undef at dead traces."""

    cube1 = xtgeo.cube_from_file(testdata_path / XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.1)

    cells = ((18, 12),)

    surf1.slice_cube(cube1, deadtraces=True, algorithm=1)
    if generate_plot:
        plotfile = tmp_path / "slice_nea1_dead1.png"
        title = "Cube with dead traces; nearest; UNDEF at dead traces"
        surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == cube1.values[icell, jcell, 0]

    ndead = (cube1.traceidcodes == 2).sum()

    assert ma.count_masked(surf1.values) == ndead

    surf2 = xtgeo.surface_from_cube(cube1, 1000.1)

    surf2.slice_cube(cube1, deadtraces=True, algorithm=2)
    if generate_plot:
        plotfile = tmp_path / "slice_nea1_dead2.png"
        title = "Cube with dead traces; nearest; UNDEF at dead traces algo 2"
        surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf2.values[icell, jcell] == cube1.values[icell, jcell, 0]

    ndead = (cube1.traceidcodes == 2).sum()

    assert ma.count_masked(surf1.values) == ndead


def test_cube_slice_w_ignore_dead_traces_trilinear(
    tmp_path, generate_plot, testdata_path
):
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = xtgeo.cube_from_file(testdata_path / XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

    cells = [(18, 12), (20, 2), (0, 4)]

    surf1.slice_cube(cube1, sampling="trilinear", snapxy=True, deadtraces=False)
    if generate_plot:
        plotfile = tmp_path / "slice_tri1.png"
        title = "Cube with dead traces; trilinear; keep as is at dead traces"
        surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == pytest.approx(
            cube1.values[icell, jcell, 0], abs=0.1
        )
    assert ma.count_masked(surf1.values) == 0  # shall be no masked cells


def test_cube_slice_w_dead_traces_trilinear(tmp_path, generate_plot, testdata_path):
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = xtgeo.cube_from_file(testdata_path / XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

    cells = [(18, 12)]

    surf1.slice_cube(cube1, sampling="trilinear", snapxy=True, deadtraces=True)
    if generate_plot:
        plotfile = tmp_path / "slice_tri1_dead.png"
        title = "Cube with dead traces; trilinear; UNDEF at dead traces"
        surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    ndead = (cube1.traceidcodes == 2).sum()

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == pytest.approx(
            cube1.values[icell, jcell, 0], 0.1
        )
    assert ma.count_masked(surf1.values) == ndead

    # swap cubes and map
    surf2 = surf1.copy()
    surf2.values = 1000.0
    cube2 = cube1.copy()
    cube2.swapaxes()
    surf2.swapaxes()
    surf2.slice_cube(cube2, sampling="trilinear", deadtraces=True)
    if generate_plot:
        plotfile = tmp_path / "slice_tri1__dead_cubeswap.png"
        surf2.quickplot(filename=plotfile, minmax=(-10000, 10000))
    assert ma.count_masked(surf2.values) == ndead
    assert surf2.values.mean() == pytest.approx(surf1.values.mean(), abs=0.01)


def test_cube_attr_mean_two_surfaces_multiattr(
    tmp_path, load_cube_rsgy1, generate_plot, testdata_path
):
    """Get cube attribute (mean) between two surfaces, many attr at the same
    time.
    """

    xs1 = xtgeo.surface_from_file(testdata_path / RTOP1)
    xs2 = xtgeo.surface_from_file(testdata_path / RBAS1)

    kube = load_cube_rsgy1

    xss = xs1.copy()
    xss.slice_cube_window(
        kube,
        other=xs2,
        other_position="below",
        attribute="rms",
        sampling="trilinear",
        showprogress=True,
    )

    xsx = xs1.copy()
    attrs = xsx.slice_cube_window(
        kube,
        other=xs2,
        other_position="below",
        attribute=["max", "mean", "min", "rms"],
        sampling="trilinear",
        showprogress=True,
    )

    assert xss.values.mean() == attrs["rms"].values.mean()
    assert xss.values.std() == attrs["rms"].values.std()

    for attr in attrs:
        xxx = attrs[attr]
        xxx.to_file(tmp_path / f"surf_slice_cube_2surf_{attr}multi.gri")

        minmax = (None, None)
        if attr == "mean":
            minmax = (-0.1, 0.1)

        if generate_plot:
            xxx.quickplot(
                filename=tmp_path / f"surf_slice_cube_2surf_{attr}multi.png",
                colormap="jet",
                minmax=minmax,
                title="Reek two surfs mean multiattr: " + attr,
                infotext="Method: trilinear, 2 surfs multiattr " + attr,
            )
