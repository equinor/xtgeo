"""Tests for 3D grid."""
import math
import pathlib
from collections import OrderedDict

import numpy as np
import pytest
from hypothesis import given

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid

from .grid_generator import dimensions, increments, xtgeo_grids

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__, info=True)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

BRILGRDECL = TPATH / "3dgrids/bri/b.grdecl"
GRIDQC1 = TPATH / "3dgrids/etc/gridqc1.roff"
GRIDQC1_CELLVOL = TPATH / "3dgrids/etc/gridqc1_totbulk.roff"
GRIDQC2 = TPATH / "3dgrids/etc/gridqc_negthick_twisted.roff"

DUALFIL1 = TPATH / "3dgrids/etc/dual_grid.roff"
DUALFIL3 = TPATH / "3dgrids/etc/TEST_DPDK.EGRID"


def test_import_wrong(tmp_path):
    """Importing wrong fformat, etc."""
    grd = xtgeo.create_box_grid((2, 2, 2))
    grd.to_file(tmp_path / "grd.roff")
    with pytest.raises(ValueError):
        xtgeo.grid_from_file(tmp_path / "grd.roff", fformat="stupid_wrong_name")


@given(xtgeo_grids)
def test_get_set_name(grd):
    grd.name = "xxx"
    assert grd.name == "xxx"


def test_create_shoebox(tmp_path):
    """Make a shoebox grid from scratch."""
    grd = xtgeo.create_box_grid((4, 3, 5))
    grd.to_file(tmp_path / "shoebox_default.roff")

    grd.create_box(flip=-1)
    grd.to_file(tmp_path / "shoebox_default_flipped.roff")

    timer1 = xtg.timer()
    grd.create_box(
        origin=(0, 0, 1000), dimension=(300, 200, 30), increment=(20, 20, 1), flip=-1
    )
    logger.info("Making a a 1,8 mill cell grid took %5.3f secs", xtg.timer(timer1))

    dx, dy = grd.get_dxdy()

    assert dx.values.mean() == pytest.approx(20.0, abs=0.0001)
    assert dy.values.mean() == pytest.approx(20.0, abs=0.0001)

    grd.create_box(
        origin=(0, 0, 1000), dimension=(30, 30, 3), rotation=45, increment=(20, 20, 1)
    )

    x, y, z = grd.get_xyz()

    assert x.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert y.values1d[0] == pytest.approx(20 * math.cos(45 * math.pi / 180), abs=0.001)
    assert z.values1d[0] == pytest.approx(1000.5, abs=0.001)

    grd.create_box(
        origin=(0, 0, 1000),
        dimension=(30, 30, 3),
        rotation=45,
        increment=(20, 20, 1),
        oricenter=True,
    )

    x, y, z = grd.get_xyz()

    assert x.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert y.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert z.values1d[0] == pytest.approx(1000.0, abs=0.001)


@pytest.mark.parametrize(
    "dimensions",
    [
        (100, 1, 1),
        (100, 1, 20),
        (300, 20, 30),
    ],
)
def test_shoebox_egrid(tmp_path, dimensions):
    grd = xtgeo.create_box_grid(dimension=dimensions)
    grd.to_file(tmp_path / "E1.EGRID", fformat="egrid")
    grd1 = xtgeo.grid_from_file(tmp_path / "E1.EGRID")
    assert grd1.dimensions == dimensions


@pytest.mark.parametrize("xtgformat", [1, 2])
@pytest.mark.benchmark()
def test_benchmark_get_xyz_cell_cornerns(benchmark, xtgformat):
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    if xtgformat == 1:
        grd._xtgformat1()
    else:
        grd._xtgformat2()

    def run():
        return grd.get_xyz_cell_corners((5, 6, 7))

    corners = benchmark(run)

    assert corners == pytest.approx(
        [4, 5, 6, 5, 5, 6, 4, 6, 6, 5, 6, 6, 4, 5, 7, 5, 5, 7, 4, 6, 7, 5, 6, 7]
    )


def test_eclgrid_import3(tmp_path):
    """Eclipse GRDECL import and translate."""
    grd = xtgeo.grid_from_file(BRILGRDECL, fformat="grdecl")

    mylist = grd.get_geometrics()

    xori1 = mylist[0]

    # translate the coordinates
    grd.translate_coordinates(translate=(100, 100, 10), flip=(1, 1, 1))

    mylist = grd.get_geometrics()

    xori2 = mylist[0]

    # check if origin is translated 100m in X
    assert xori1 + 100 == xori2, "Translate X distance"

    grd.to_file(tmp_path / "g1_translate.roff", fformat="roff_binary")

    grd.to_file(tmp_path / "g1_translate.bgrdecl", fformat="bgrdecl")


def test_npvalues1d():
    """Different ways of getting np arrays."""
    grd = xtgeo.grid_from_file(DUALFIL3)
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)  # [  1.   1.   1.   1.   1.  nan  ...]
    dz2 = dz.get_npvalues1d(activeonly=True)  # [  1.   1.   1.   1.   1.  1. ...]

    assert dz1[0] == 1.0
    assert np.isnan(dz1[5])
    assert dz1[0] == 1.0
    assert not np.isnan(dz2[5])

    grd = xtgeo.grid_from_file(DUALFIL1)  # all cells active
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)
    dz2 = dz.get_npvalues1d(activeonly=True)

    assert dz1.all() == dz2.all()


def test_pathlib(tmp_path):
    """Import and export via pathlib."""
    grd = xtgeo.grid_from_file(pathlib.Path(DUALFIL1))

    assert grd.dimensions == (5, 3, 1)

    out = tmp_path / "grdpathtest.roff"
    grd.to_file(out, fformat="roff")

    with pytest.raises(OSError):
        out = pathlib.Path() / "nosuchdir" / "grdpathtest.roff"
        grd.to_file(out, fformat="roff")


def test_xyz_cell_corners():
    """Test xyz variations."""
    grd = xtgeo.grid_from_file(DUALFIL1)

    allcorners = grd.get_xyz_corners()
    assert len(allcorners) == 24
    assert allcorners[0].get_npvalues1d()[0] == 0.0
    assert allcorners[23].get_npvalues1d()[-1] == 1001.0


@given(xtgeo_grids)
def test_generate_hash(grd1):
    grd2 = grd1.copy()
    assert id(grd1) != id(grd2)
    assert grd1.generate_hash() == grd2.generate_hash()


def test_gridquality_properties(show_plot):
    """Get grid quality props."""
    grd1 = xtgeo.grid_from_file(GRIDQC1)

    props1 = grd1.get_gridquality_properties()
    minang = props1.get_prop_by_name("minangle_topbase")
    assert minang.values[5, 2, 1] == pytest.approx(71.05561, abs=0.001)
    if show_plot:
        lay = 1
        layslice = xtgeo.plot.Grid3DSlice()
        layslice.canvas(title=f"Layer {lay}")
        layslice.plot_gridslice(
            grd1,
            prop=minang,
            mode="layer",
            index=lay + 1,
            window=None,
            linecolor="black",
        )

        layslice.show()

    grd2 = xtgeo.grid_from_file(GRIDQC2)
    props2 = grd2.get_gridquality_properties()

    neg = props2.get_prop_by_name("negative_thickness")
    assert neg.values[0, 0, 0] == 0
    assert neg.values[2, 1, 0] == 1


def test_bulkvol():
    """Test cell bulk volume calculation."""
    grd = xtgeo.grid_from_file(GRIDQC1)
    cellvol_rms = xtgeo.gridproperty_from_file(GRIDQC1_CELLVOL)

    bulk = grd.get_bulk_volume()
    logger.info("Sum this: %s", bulk.values.sum())
    logger.info("Sum RMS: %s", cellvol_rms.values.sum())

    assert bulk.values.sum() == pytest.approx(cellvol_rms.values.sum(), rel=0.001)


@pytest.mark.benchmark(group="bulkvol")
def test_bulkvol_speed(benchmark):
    dimens = (10, 50, 5)
    grd = xtgeo.create_box_grid(dimension=dimens)

    def run():
        _ = grd.get_bulk_volume()

    benchmark(run)


def test_bad_egrid_ends_before_kw(tmp_path):
    egrid_file = tmp_path / "test.egrid"
    with open(egrid_file, "wb") as fh:
        fh.write(b"\x00\x00\x00\x10")
    with pytest.raises(Exception, match="end-of-file while reading keyword"):
        xtgeo.grid_from_file(egrid_file, fformat="egrid")


@given(dimensions, increments, increments, increments)
def test_grid_get_dx(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(
        dimension=dimension, increment=(dx, dy, dz), rotation=0.0
    )
    np.testing.assert_allclose(grd.get_dx(metric="euclid").values, dx, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dx(metric="north south vertical").values, 0.0, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dx(metric="east west vertical").values, dx, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dx(metric="horizontal").values, dx, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="x projection").values, dx, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="y projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="z projection").values, 0.0, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dx(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dx(asmasked=False).values[0, 0, 0], dx, atol=0.01)


@given(dimensions, increments, increments, increments)
def test_grid_get_dy(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(
        dimension=dimension, increment=(dx, dy, dz), rotation=0.0
    )
    np.testing.assert_allclose(grd.get_dy(metric="euclid").values, dy, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dy(metric="north south vertical").values, dy, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dy(metric="east west vertical").values, 0.0, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dy(metric="horizontal").values, dy, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="x projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="y projection").values, dy, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="z projection").values, 0.0, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dy(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dy(asmasked=False).values[0, 0, 0], dy, atol=0.01)


@given(dimensions, increments, increments, increments)
def test_grid_get_dz(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(dimension=dimension, increment=(dx, dy, dz))
    np.testing.assert_allclose(grd.get_dz(metric="euclid").values, dz, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dz(metric="north south vertical").values, dz, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dz(metric="east west vertical").values, dz, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dz(metric="horizontal").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="x projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="y projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="z projection").values, dz, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(flip=False).values, -dz, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dz(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dz(asmasked=False).values[0, 0, 0], dz, atol=0.01)


@given(xtgeo_grids)
def test_get_dxdy_is_get_dx_and_dy(grid):
    assert np.all(grid.get_dxdy(asmasked=True)[0].values == grid.get_dx().values)
    assert np.all(grid.get_dxdy(asmasked=True)[1].values == grid.get_dy().values)


def test_benchmark_grid_get_dz(benchmark):
    grd = xtgeo.create_box_grid(dimension=(100, 100, 100))

    def run():
        grd.get_dz()

    benchmark(run)


def test_benchmark_grid_get_dxdy(benchmark):
    grd = xtgeo.create_box_grid(dimension=(100, 100, 100))

    def run():
        grd.get_dxdy()

    benchmark(run)


def test_grid_get_dxdydz_zero_size():
    grd = xtgeo.create_box_grid(dimension=(0, 0, 0))

    assert grd.get_dx().values.shape == (0, 0, 0)
    assert grd.get_dy().values.shape == (0, 0, 0)
    assert grd.get_dz().values.shape == (0, 0, 0)


def test_grid_get_dxdydz_bad_coordsv_size():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    grd._coordsv = np.zeros(shape=(0, 0, 0))

    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_zcorn_size():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    grd._zcornsv = np.zeros(shape=(0, 0, 0, 0))

    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_grid_top():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

    grd._coordsv[:, :, 2] = 0.0
    grd._coordsv[:, :, 5] = 0.0
    grd._coordsv[:, :, 0] += 1.0

    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_metric():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dx(metric="foo")
    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dy(metric="foo")
    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dz(metric="foo")


def test_grid_roff_subgrids_import_regression(tmp_path):
    grid = xtgeo.create_box_grid(dimension=(5, 5, 67))
    grid.subgrids = OrderedDict(
        [
            ("subgrid_0", list(range(1, 21))),
            ("subgrid_1", list(range(21, 53))),
            ("subgrid_2", list(range(53, 68))),
        ]
    )
    grid.to_file(tmp_path / "grid.roff")

    grid2 = xtgeo.grid_from_file(tmp_path / "grid.roff")
    assert grid2.subgrids == OrderedDict(
        [
            ("subgrid_0", range(1, 21)),
            ("subgrid_1", range(21, 53)),
            ("subgrid_2", range(53, 68)),
        ]
    )


@pytest.mark.parametrize(
    "coordsv_dtype, zcornsv_dtype, actnumsv_dtype, match",
    [
        (np.float32, np.float32, np.int32, "The dtype of the coordsv"),
        (np.float64, np.float64, np.int32, "The dtype of the zcornsv"),
        (np.float64, np.float32, np.uint8, "The dtype of the actnumsv"),
    ],
)
def test_grid_bad_dtype_construction(
    coordsv_dtype, zcornsv_dtype, actnumsv_dtype, match
):
    with pytest.raises(TypeError, match=match):
        Grid(
            np.zeros((2, 2, 6), dtype=coordsv_dtype),
            np.zeros((2, 2, 2, 4), dtype=zcornsv_dtype),
            np.zeros((1, 1, 1), dtype=actnumsv_dtype),
        )


@pytest.mark.parametrize(
    "coordsv_dimensions, zcornsv_dimensions, actnumsv_dimensions, match",
    [
        ((2, 2, 2), (2, 2, 2, 4), (1, 1, 1), "shape of coordsv"),
        ((2, 2, 6), (2, 2, 2, 3), (1, 1, 1), "shape of zcornsv"),
        ((2, 2, 6), (2, 1, 2, 4), (1, 1, 1), "Mismatch between zcornsv and coordsv"),
        ((2, 2, 6), (2, 2, 2, 4), (1, 2, 1), "Mismatch between zcornsv and actnumsv"),
    ],
)
def test_grid_bad_dimensions_construction(
    coordsv_dimensions, zcornsv_dimensions, actnumsv_dimensions, match
):
    with pytest.raises(ValueError, match=match):
        Grid(
            np.zeros(coordsv_dimensions, dtype=np.float64),
            np.zeros(zcornsv_dimensions, dtype=np.float32),
            np.zeros(actnumsv_dimensions, dtype=np.int32),
        )


@given(xtgeo_grids)
def test_reduce_to_one_layer(grd):
    grd.reduce_to_one_layer()

    assert grd.nlay == 1
