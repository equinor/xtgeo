# -*- coding: utf-8 -*-


from os.path import join

import numpy.ma as ma
import pytest

import tests.test_common.test_xtg as tsetup
import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.cube import Cube

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

RPATH1 = TPATH / "surfaces/reek"
RPATH3 = TPATH / "surfaces/etc"
RPATH2 = TPATH / "cubes/reek"
RPATH4 = TPATH / "cubes/etc"
RTOP1 = RPATH1 / "1/topreek_rota.gri"
RBAS1 = RPATH1 / "1/basereek_rota.gri"
RBAS2 = RPATH1 / "1/basereek_rota_v2.gri"
RSGY1 = RPATH2 / "syntseis_20000101_seismic_depth_stack.segy"

XTOP1 = RPATH3 / "ib_test_horizon2.gri"
XCUB1 = RPATH4 / "ib_test_cube2.segy"
XCUB2 = RPATH4 / "cube_w_deadtraces.segy"


@pytest.fixture()
def load_cube_rsgy1():
    """Loading test cube (pytest setup fixture)"""
    logger.info("Load cube RSGY1")
    return Cube(RSGY1)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_get_surface_from_cube(load_cube_rsgy1):
    """Construct a constant surface from cube."""

    cube = load_cube_rsgy1

    surf = xtgeo.surface_from_cube(cube, 1999.0)

    assert surf.xinc == cube.xinc
    assert surf.nrow == cube.nrow
    tsetup.assert_almostequal(surf.values.mean(), 1999.0, 0.00001)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_nearest_snapxy(tmpdir, load_cube_rsgy1, generate_plot):
    """Slice a cube with a surface, nearest node, snapxy, algorithm 1 + 2"""

    kube = load_cube_rsgy1

    xs1 = xtgeo.surface_from_cube(kube, 1670)
    xs2 = xtgeo.surface_from_cube(kube, 1670)

    t1 = xtg.timer()
    xs1.slice_cube(kube, algorithm=1, snapxy=True)
    logger.info("Slicing alg 1...done in {} seconds".format(xtg.timer(t1)))

    t1 = xtg.timer()
    xs2.slice_cube(kube, algorithm=2, snapxy=True)
    logger.info("Slicing alg 2...done in {} seconds".format(xtg.timer(t1)))

    if generate_plot:
        xs1.quickplot(
            filename=join(tmpdir, "surf_slice_cube_near_snapxy_v1.png"),
            colortable="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, snapxy, algorithm 1",
        )

        xs2.quickplot(
            filename=join(tmpdir, "surf_slice_cube_near_snapxy_v2.png"),
            colortable="seismic",
            minmax=(-1, 1),
            title="Reek",
            infotext="Method: nearest, snapxy, algorithm 2",
        )

    logger.info("%s vs %s", xs1.values.mean(), xs2.values.mean())
    tsetup.assert_almostequal(xs1.values.mean(), xs2.values.mean(), 0.0001)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_trilinear_snapxy(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, trilinear, snapxy, algorithm 1 + 2"""

    kube = load_cube_rsgy1

    xs1 = xtgeo.surface_from_cube(kube, 1670)
    xs2 = xtgeo.surface_from_cube(kube, 1670)

    t1 = xtg.timer()
    xs1.slice_cube(kube, algorithm=1, snapxy=True, sampling="trilinear")
    logger.info("Slicing alg 1...done in {} seconds".format(xtg.timer(t1)))

    t1 = xtg.timer()
    xs2.slice_cube(kube, algorithm=2, snapxy=True, sampling="trilinear")
    logger.info("Slicing alg 2...done in {} seconds".format(xtg.timer(t1)))

    xs1.quickplot(
        filename=join(tmpdir, "surf_slice_cube_tri_snapxy_v1.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: trilinear, snapxy, algorithm 1",
    )

    xs2.quickplot(
        filename=join(tmpdir, "surf_slice_cube_tri_snapxy_v2.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: trilinear, snapxy, algorithm 2",
    )

    logger.info("%s vs %s", xs1.values.mean(), xs2.values.mean())
    tsetup.assert_almostequal(xs1.values.mean(), xs2.values.mean(), 0.0001)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_nearest_nosnapxy(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, nearest node, algorithm 1 + 2, other map layout"""

    kube = load_cube_rsgy1

    # kube.swapaxes()

    xs1 = xtgeo.RegularSurface(
        yori=5927600, xori=457000, xinc=50, yinc=50, ncol=200, nrow=220, values=1670
    )
    xs2 = xs1.copy()

    t1 = xtg.timer()
    xs1.slice_cube(kube, algorithm=1, snapxy=False)
    logger.info("Slicing alg 1...done in {} seconds".format(xtg.timer(t1)))

    t1 = xtg.timer()
    xs2.slice_cube(kube, algorithm=2, snapxy=False)
    logger.info("Slicing alg 2...done in {} seconds".format(xtg.timer(t1)))

    xs1.quickplot(
        filename=join(tmpdir, "surf_slice_cube_near_nosnapxy_v1.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: nearest, nosnapxy, algorithm 1",
    )

    xs2.quickplot(
        filename=join(tmpdir, "surf_slice_cube_near_nosnapxy_v2.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: nearest, nosnapxy, algorithm 2",
    )

    logger.info("%s vs %s", xs1.values.mean(), xs2.values.mean())
    tsetup.assert_almostequal(xs1.values.mean(), xs2.values.mean(), 0.0001)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_trilinear_nosnapxy(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, nearest node, algorithm 1 + 2, other map layout"""

    kube = load_cube_rsgy1

    # kube.swapaxes()

    xs1 = xtgeo.RegularSurface(
        yori=5927600, xori=457000, xinc=50, yinc=50, ncol=200, nrow=220, values=1670
    )
    xs2 = xs1.copy()

    t1 = xtg.timer()
    xs1.slice_cube(kube, algorithm=1, snapxy=False, sampling="trilinear")
    logger.info("Slicing alg 1...done in {} seconds".format(xtg.timer(t1)))

    t1 = xtg.timer()
    xs2.slice_cube(kube, algorithm=2, snapxy=False, sampling="trilinear")
    logger.info("Slicing alg 2...done in {} seconds".format(xtg.timer(t1)))

    xs1.quickplot(
        filename=join(tmpdir, "surf_slice_cube_tri_nosnapxy_v1.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: trilinear, nosnapxy, algorithm 1",
    )

    xs2.quickplot(
        filename=join(tmpdir, "surf_slice_cube_tri_nosnapxy_v2.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: trilinear, nosnapxy, algorithm 2",
    )

    logger.info("%s vs %s", xs1.values.mean(), xs2.values.mean())
    tsetup.assert_almostequal(xs1.values.mean(), xs2.values.mean(), 0.0001)


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_nearest(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, nearest node, algorithm 1"""

    xs = xtgeo.surface_from_file(RTOP1)
    xs.to_file(join(tmpdir, "surf_slice_cube_initial.gri"))

    kube = load_cube_rsgy1

    t1 = xtg.timer()
    xs.slice_cube(kube, algorithm=1)
    logger.info("Slicing...done in {} seconds".format(xtg.timer(t1)))

    xs.to_file(join(tmpdir, "surf_slice_cube_v1.gri"), fformat="irap_binary")

    xs.quickplot(
        filename=join(tmpdir, "surf_slice_cube_near_v1.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: nearest, algorithm 1",
    )


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_nearest_v2(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, nearest node, algorithm 2."""

    xs = xtgeo.surface_from_file(RTOP1)

    kube = load_cube_rsgy1

    t1 = xtg.timer()

    xs.slice_cube(kube, algorithm=2)
    logger.info("Slicing...done in {} seconds".format(xtg.timer(t1)))

    xs.to_file(join(tmpdir, "surf_slice_cube_alg2.gri"), fformat="irap_binary")

    xs.quickplot(
        filename=join(tmpdir, "surf_slice_cube_alg2.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: nearest",
    )


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_various_reek(tmpdir, load_cube_rsgy1):
    """Slice a cube with a surface, both nearest node and interpol, Reek."""

    logger.info("Loading surface")
    xs = xtgeo.surface_from_file(RTOP1)

    logger.info("Loading cube")
    kube = load_cube_rsgy1

    t1 = xtg.timer()
    xs.slice_cube(kube)
    t2 = xtg.timer(t1)
    logger.info("Slicing... nearest, done in {} seconds".format(t2))

    xs.to_file(join(tmpdir, "surf_slice_cube_reek_interp.gri"))

    xs.quickplot(
        filename=join(tmpdir, "surf_slice_cube_reek_interp.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: nearest",
    )

    # trilinear interpolation:

    logger.info("Loading surface")
    xs = xtgeo.surface_from_file(RTOP1)

    t1 = xtg.timer()
    xs.slice_cube(kube, sampling="trilinear")
    t2 = xtg.timer(t1)
    logger.info("Slicing... trilinear, done in {} seconds".format(t2))

    xs.to_file(join(tmpdir, "surf_slice_cube_reek_trilinear.gri"))

    xs.quickplot(
        filename=join(tmpdir, "surf_slice_cube_reek_trilinear.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek",
        infotext="Method: trilinear",
    )


@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_attr_window_max(load_cube_rsgy1):
    """Slice a cube within a window, get max, using trilinear interpol."""

    logger.info("Loading surface")
    xs1 = xtgeo.surface_from_file(RTOP1)

    logger.info("Loading cube")
    kube = load_cube_rsgy1

    ret = xs1.slice_cube_window(
        kube, attribute="max", sampling="trilinear", algorithm=2
    )
    logger.info(xs1.values.mean())
    assert ret is None
    tsetup.assert_almostequal(xs1.values.mean(), 0.08619, 0.001)

    # one attribute but in a list context shall return a dict
    xs1 = xtgeo.surface_from_file(RTOP1)
    ret = xs1.slice_cube_window(
        kube, attribute=["max"], sampling="trilinear", algorithm=2
    )
    assert isinstance(ret, dict)

    tsetup.assert_almostequal(ret["max"].values.mean(), 0.08619, 0.001)


@tsetup.bigtest
@tsetup.skipsegyio
@pytest.mark.skipifroxar
def test_slice_attr_window_max_w_plotting(tmpdir, load_cube_rsgy1):
    """Slice a cube within a window, get max/min etc, using trilinear
    interpol and plotting."""

    logger.info("Loading surface")
    xs1 = xtgeo.surface_from_file(RTOP1)
    xs2 = xs1.copy()
    xs3 = xs1.copy()

    logger.info("Loading cube")
    kube = load_cube_rsgy1

    t1 = xtg.timer()
    xs1.slice_cube_window(kube, attribute="min", sampling="trilinear")
    t2 = xtg.timer(t1)
    logger.info("Window slicing... {} secs".format(t2))

    xs1.quickplot(
        filename=join(tmpdir, "surf_slice_cube_window_min.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek Minimum",
        infotext="Method: trilinear, window",
    )

    xs2.slice_cube_window(
        kube, attribute="max", sampling="trilinear", showprogress=True
    )

    xs2.quickplot(
        filename=join(tmpdir, "surf_slice_cube_window_max.png"),
        colortable="seismic",
        minmax=(-1, 1),
        title="Reek Maximum",
        infotext="Method: trilinear, window",
    )

    xs3.slice_cube_window(kube, attribute="rms", sampling="trilinear")

    xs3.quickplot(
        filename=join(tmpdir, "surf_slice_cube_window_rms.png"),
        colortable="jet",
        minmax=(0, 1),
        title="Reek rms (root mean square)",
        infotext="Method: trilinear, window",
    )


@pytest.mark.skipifroxar
def test_cube_attr_mean_two_surfaces(tmpdir, load_cube_rsgy1):
    """Get cube attribute (mean) between two surfaces."""

    logger.info("Loading surfaces {} {}".format(RTOP1, RBAS1))
    xs1 = xtgeo.surface_from_file(RTOP1)
    xs2 = xtgeo.surface_from_file(RBAS1)

    logger.info("Loading cube {}".format(RSGY1))
    kube = load_cube_rsgy1

    xss = xs1.copy()
    xss.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="mean", sampling="trilinear"
    )

    xss.to_file(join(tmpdir, "surf_slice_cube_2surf_meantri.gri"))

    xss.quickplot(
        filename=join(tmpdir, "surf_slice_cube_2surf_mean.png"),
        colortable="jet",
        title="Reek two surfs mean",
        minmax=(-0.1, 0.1),
        infotext="Method: trilinear, 2 surfs",
    )

    logger.info("Mean is {}".format(xss.values.mean()))


@pytest.mark.skipifroxar
def test_cube_attr_rms_two_surfaces_compare_window(load_cube_rsgy1):
    """Get cube attribute (rms) between two surfaces, and compare with
    window."""

    xs1 = xtgeo.surface_from_file(RTOP1)
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


@tsetup.bigtest
@pytest.mark.skipifroxar
def test_cube_attr_rms_two_surfaces_compare_window_show(tmpdir, load_cube_rsgy1):
    """Get cube attribute (rms) between two surfaces, and compare with
    window, and show plots."""

    logger.info("Loading surfaces {} {}".format(RTOP1, RBAS1))
    xs1 = xtgeo.surface_from_file(RTOP1)
    xs2 = xs1.copy()
    xs2.values += 30

    logger.info("Loading cube {}".format(RSGY1))
    kube = load_cube_rsgy1

    xss1 = xs1.copy()
    xss1.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="rms", sampling="trilinear"
    )

    xss1.quickplot(
        filename=join(tmpdir, "surf_slice_cube_2surf_rms1.png"),
        colortable="jet",
        minmax=[0, 0.5],
        # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
        infotext="Method: trilinear, 2 surfs, 30ms apart",
    )

    print("\n\n{}\n".format("=" * 100))

    xss2 = xs1.copy()
    xss2.values += 15
    xss2.slice_cube_window(kube, zrange=15, attribute="rms", sampling="trilinear")

    xss2.quickplot(
        filename=join(tmpdir, "surf_slice_cube_2surf_rms2.png"),
        colortable="jet",
        minmax=[0, 0.5],
        # TODO: itle='Reek two surfs mean', minmax=(-0.1, 0.1),
        infotext="Method: trilinear, 2 surfs, +- 15ms window",
    )

    assert xss1.values.mean() == xss2.values.mean()


@pytest.mark.skipifroxar
def test_cube_attr_mean_two_surfaces_with_zeroiso(tmpdir, load_cube_rsgy1):
    """Get cube attribute between two surfaces with partly zero isochore."""

    logger.info("Loading surfaces {} {}".format(RTOP1, RBAS1))
    xs1 = xtgeo.surface_from_file(RTOP1)
    xs2 = xtgeo.surface_from_file(RBAS2)

    logger.info("Loading cube {}".format(RSGY1))
    kube = load_cube_rsgy1

    xss = xs1.copy()
    xss.slice_cube_window(
        kube, other=xs2, other_position="below", attribute="mean", sampling="trilinear"
    )

    xss.to_file(join(tmpdir, "surf_slice_cube_2surf_meantri.gri"))

    xss.quickplot(
        filename=join(tmpdir, "surf_slice_cube_2surf_mean_v2.png"),
        colortable="jet",
        title="Reek two surfs mean",
        minmax=(-0.1, 0.1),
        infotext="Method: trilinear, 2 surfs, partly zero isochore",
    )

    logger.info("Mean is {}".format(xss.values.mean()))


@pytest.mark.skipifroxar
@tsetup.skipifwindows
def test_cube_slice_auto4d_data(tmpdir):
    """Get cube slice aka Auto4D input, with synthetic/scrambled data"""

    xs1 = xtgeo.surface_from_file(XTOP1, fformat="gri")
    xs1.describe()

    xs1out = join(tmpdir, "XTOP1.ijxyz")
    xs1.to_file(xs1out, fformat="ijxyz")

    xs2 = xtgeo.surface_from_file(xs1out, fformat="ijxyz")

    assert xs1.values.mean() == pytest.approx(xs2.values.mean(), abs=0.0001)

    kube1 = Cube(XCUB1)
    kube1.describe()

    assert xs2.nactive == 10830

    xs2.slice_cube_window(kube1, sampling="trilinear", mask=True, attribute="max")

    xs2out1 = join(tmpdir, "XTOP2_sampled_from_cube.ijxyz")
    xs2out2 = join(tmpdir, "XTOP2_sampled_from_cube.gri")
    xs2out3 = join(tmpdir, "XTOP2_sampled_from_cube.png")

    xs2.to_file(xs2out1, fformat="ijxyz")
    xs2.to_file(xs2out2)

    assert xs2.nactive == 3275  # 3320  # shall be fewer cells

    xs2.quickplot(
        filename=xs2out3,
        colortable="seismic",
        title="Auto4D Test",
        minmax=(0, 12000),
        infotext="Method: max",
    )


@pytest.mark.skipifroxar
def test_cube_slice_w_ignore_dead_traces_nearest(tmpdir):
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, ignore dead traces."""

    cube1 = Cube(XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.1)

    cells = ((18, 12), (20, 2), (0, 4))

    surf1.slice_cube(cube1, deadtraces=False)
    plotfile = join(tmpdir, "slice_nea1.png")
    title = "Cube with dead traces; nearest; use just values as is"
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


@pytest.mark.skipifroxar
def test_cube_slice_w_dead_traces_nearest(tmpdir):
    """Get cube slice nearest aka Auto4D input, with scrambled data with
    dead traces, various YFLIP cases, undef at dead traces."""

    cube1 = Cube(XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.1)

    cells = ((18, 12),)

    surf1.slice_cube(cube1, deadtraces=True, algorithm=1)
    plotfile = join(tmpdir, "slice_nea1_dead1.png")
    title = "Cube with dead traces; nearest; UNDEF at dead traces"
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == cube1.values[icell, jcell, 0]

    ndead = (cube1.traceidcodes == 2).sum()

    assert ma.count_masked(surf1.values) == ndead

    surf2 = xtgeo.surface_from_cube(cube1, 1000.1)

    surf2.slice_cube(cube1, deadtraces=True, algorithm=2)
    plotfile = join(tmpdir, "slice_nea1_dead2.png")
    title = "Cube with dead traces; nearest; UNDEF at dead traces algo 2"
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf2.values[icell, jcell] == cube1.values[icell, jcell, 0]

    ndead = (cube1.traceidcodes == 2).sum()

    assert ma.count_masked(surf1.values) == ndead


@pytest.mark.skipifroxar
def test_cube_slice_w_ignore_dead_traces_trilinear(tmpdir):
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = Cube(XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

    cells = [(18, 12), (20, 2), (0, 4)]

    surf1.slice_cube(cube1, sampling="trilinear", snapxy=True, deadtraces=False)
    plotfile = join(tmpdir, "slice_tri1.png")
    title = "Cube with dead traces; trilinear; keep as is at dead traces"
    surf1.quickplot(filename=plotfile, minmax=(-10000, 10000), title=title)

    for cell in cells:
        icell, jcell = cell
        assert surf1.values[icell, jcell] == pytest.approx(
            cube1.values[icell, jcell, 0], abs=0.1
        )
    assert ma.count_masked(surf1.values) == 0  # shall be no masked cells


@pytest.mark.skipifroxar
def test_cube_slice_w_dead_traces_trilinear(tmpdir):
    """Get cube slice trilinear aka Auto4D input, with scrambled data with
    dead traces to be ignored, various YFLIP cases."""

    cube1 = Cube(XCUB2)

    surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

    cells = [(18, 12)]

    surf1.slice_cube(cube1, sampling="trilinear", snapxy=True, deadtraces=True)
    plotfile = join(tmpdir, "slice_tri1_dead.png")
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
    plotfile = join(tmpdir, "slice_tri1__dead_cubeswap.png")
    surf2.quickplot(filename=plotfile, minmax=(-10000, 10000))
    assert ma.count_masked(surf2.values) == ndead
    tsetup.assert_almostequal(surf2.values.mean(), surf1.values.mean(), 0.01)


@tsetup.bigtest
@pytest.mark.skipifroxar
def test_cube_attr_mean_two_surfaces_multiattr(tmpdir, load_cube_rsgy1):
    """Get cube attribute (mean) between two surfaces, many attr at the same
    time.
    """

    logger.info("Loading surfaces {} {}".format(RTOP1, RBAS1))
    xs1 = xtgeo.surface_from_file(RTOP1)
    xs2 = xtgeo.surface_from_file(RBAS1)

    logger.info("Loading cube {}".format(RSGY1))
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

    logger.debug(xss.values.mean())

    xsx = xs1.copy()
    attrs = xsx.slice_cube_window(
        kube,
        other=xs2,
        other_position="below",
        attribute=["max", "mean", "min", "rms"],
        sampling="trilinear",
        showprogress=True,
    )
    logger.debug(attrs["rms"].values.mean())

    assert xss.values.mean() == attrs["rms"].values.mean()
    assert xss.values.std() == attrs["rms"].values.std()

    for attr in attrs.keys():
        logger.info("Working with %s", attr)

        xxx = attrs[attr]
        xxx.to_file(join(tmpdir, "surf_slice_cube_2surf_" + attr + "multi.gri"))

        minmax = (None, None)
        if attr == "mean":
            minmax = (-0.1, 0.1)

        xxx.quickplot(
            filename=join(tmpdir, "surf_slice_cube_2surf_" + attr + "multi.png"),
            colortable="jet",
            minmax=minmax,
            title="Reek two surfs mean multiattr: " + attr,
            infotext="Method: trilinear, 2 surfs multiattr " + attr,
        )
