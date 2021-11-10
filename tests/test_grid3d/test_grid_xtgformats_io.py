# coding: utf-8
"""Testing new xtgf and hdf5/h5 formats."""
from collections import OrderedDict
from os.path import join

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj


BIGBOX_DIMENSIONS = (100, 100, 20)


def create_box(testpath):
    grid = xtgeo.Grid()
    grid.create_box(BIGBOX_DIMENSIONS)
    return grid


@pytest.fixture(
    name="benchmark_grid",
    params=[
        lambda tp: xtgeo.Grid(join(tp, "3dgrids/reek/reek_geo_grid.roff")),
        create_box,
    ],
    ids=["reek_grid", "big_box"],
)
def benchmark_grid_fixture(request, testpath):
    return request.param(testpath)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_xtgf_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.xtgf")

    def write():
        benchmark_grid.to_xtgf(fname)

    benchmark(write)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_xtgf_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.xtgf")

    benchmark_grid.to_xtgf(fname)

    grid2 = None

    def read():
        nonlocal grid2
        grid2 = xtgeo.grid_from_file(fname)

    benchmark(read)

    assert_allclose(benchmark_grid._zcornsv, grid2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grid2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grid2._actnumsv)


@pytest.mark.bigtest
@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_grdecl_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.grdecl")

    def write():
        benchmark_grid.to_file(fname, fformat="grdecl")

    benchmark(write)


@pytest.mark.bigtest
@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_grdecl_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.grdecl")

    benchmark_grid.to_file(fname, fformat="grdecl")

    grid2 = None

    def read():
        nonlocal grid2
        grid2 = xtgeo.grid_from_file(fname, fformat="grdecl")

    benchmark(read)

    assert_allclose(benchmark_grid._zcornsv, grid2._zcornsv, atol=0.02)
    assert_allclose(benchmark_grid._coordsv, grid2._coordsv, atol=0.02)
    assert_allclose(benchmark_grid._actnumsv, grid2._actnumsv, atol=0.02)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_bgrdecl_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.bgrdecl")

    def write():
        benchmark_grid.to_file(fname, fformat="bgrdecl")

    benchmark(write)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_bgrdecl_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.bgrdecl")

    benchmark_grid.to_file(fname, fformat="bgrdecl")

    grid2 = None

    def read():
        nonlocal grid2
        grid2 = xtgeo.grid_from_file(fname, fformat="bgrdecl")

    benchmark(read)

    assert_allclose(benchmark_grid._zcornsv, grid2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grid2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grid2._actnumsv)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    def write():
        benchmark_grid._zcornsv += 1.0
        benchmark_grid.to_hdf(fname, compression=None)

    benchmark(write)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_import_partial(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    benchmark_grid._zcornsv += 1.0
    fna = benchmark_grid.to_hdf(fname, compression=None)

    grd2 = None

    def partial_read():
        nonlocal grd2
        grd2 = xtgeo.grid_from_file(fna, ijkrange=(1, 20, 1, 20, "min", "max"))

    benchmark(partial_read)

    assert grd2.ncol == 20
    assert grd2.nlay == benchmark_grid.nlay


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    benchmark_grid._zcornsv += 1.0
    fna = benchmark_grid.to_hdf(fname, compression=None)

    grd2 = None

    def read():
        nonlocal grd2
        grd2 = xtgeo.grid_from_file(fna)

    benchmark(read)

    assert_allclose(benchmark_grid._zcornsv, grd2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grd2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grd2._actnumsv)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_export_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    def write():
        benchmark_grid.to_hdf(fname, compression="blosc")

    benchmark(write)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_import_partial_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = None

    def partial_read():
        nonlocal grd2
        grd2 = xtgeo.grid_from_file(
            fna, fformat="hdf", ijkrange=(1, 20, 1, 20, "min", "max")
        )

    benchmark(partial_read)

    assert grd2.ncol == 20
    assert grd2.nlay == benchmark_grid.nlay


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_hdf5_import_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = None

    def read():
        nonlocal grd2
        grd2 = xtgeo.grid_from_file(fna, fformat="hdf")

    benchmark(read)

    assert_allclose(benchmark_grid._zcornsv, grd2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grd2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grd2._actnumsv)


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_egrid_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.egrid")

    @benchmark
    def write():
        benchmark_grid._zcornsv += 1.0
        benchmark_grid.to_file(fname, fformat="egrid")


@pytest.mark.benchmark(group="import/export grid")
def test_benchmark_grid_egrid_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.egrid")

    benchmark_grid._zcornsv += 1.0
    benchmark_grid.to_file(fname, fformat="egrid")

    grd2 = None

    @benchmark
    def read():
        nonlocal grd2
        grd2 = xtgeo.grid_from_file(fname, fformat="egrid")

    assert_allclose(benchmark_grid._zcornsv, grd2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grd2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grd2._actnumsv)


# ======================================================================================
# Grid properties:


def create_big_prop(testpath):
    vals = np.zeros(BIGBOX_DIMENSIONS, dtype=np.float32)
    ncol, nrow, nlay = BIGBOX_DIMENSIONS
    prp = xtgeo.GridProperty(ncol=ncol, nrow=nrow, nlay=nlay, values=vals)
    prp.values[0, 0, 0:3] = 33
    prp.values[1, 0, 0:3] = 66
    prp.values[1, 1, 0:3] = 44
    return prp


def create_small_prop(testpath):
    vals = np.zeros((5, 7, 3), dtype=np.float32)
    prp = xtgeo.GridProperty(ncol=5, nrow=7, nlay=3, values=vals)
    prp.values[0, 0, 0:3] = 33
    prp.values[1, 0, 0:3] = 66
    prp.values[1, 1, 0:3] = 44
    return prp


@pytest.fixture(
    name="benchmark_gridprop",
    params=[
        lambda tp: xtgeo.GridProperty(join(tp, "3dgrids/reek2/geogrid--poro.roff")),
        create_big_prop,
        create_small_prop,
    ],
    ids=["reek poro", "small prop", "big prop"],
)
def benchmark_gridprop_fixture(request, testpath):
    return request.param(testpath)


@pytest.mark.benchmark(group="import/export grid property")
def test_benchmark_gridprop_export(benchmark, tmp_path, benchmark_gridprop):

    fname = tmp_path / "benchmark.xtgcpprop"

    def write():
        benchmark_gridprop.to_file(fname, fformat="xtgcpprop")

    benchmark(write)


@pytest.mark.benchmark(group="import/export grid property")
def test_benchmark_gridprop_import(benchmark, tmp_path, benchmark_gridprop):
    """Test exporting etc to xtgcpprop format."""

    fname = tmp_path / "benchmark.xtgcpprop"

    benchmark_gridprop.to_file(fname, fformat="xtgcpprop")

    prop2 = None

    def read():
        nonlocal prop2
        prop2 = xtgeo.gridproperty_from_file(fname, fformat="xtgcpprop")

    benchmark(read)

    assert benchmark_gridprop.values.all() == prop2.values.all()


@pytest.mark.benchmark(group="import/export grid property")
def test_benchmark_gridprop_import_partial(benchmark, tmp_path, benchmark_gridprop):
    """Test exporting etc to xtgcpprop format."""

    fname = tmp_path / "benchmark.xtgcpprop"

    benchmark_gridprop.to_file(fname, fformat="xtgcpprop")

    prop2 = None

    def read():
        nonlocal prop2
        prop2 = xtgeo.gridproperty_from_file(
            fname, fformat="xtgcpprop", ijrange=(1, 2, 1, 2)
        )

    benchmark(read)

    assert benchmark_gridprop.values[0:2, 0:2, :].all() == prop2.values.all()


def test_hdf5_partial_import_case(benchmark_grid, tmp_path):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = xtgeo.grid_from_file(fna, fformat="hdf", ijkrange=(1, 3, 1, 5, 1, 4))

    assert grd2.ncol == 3
    assert grd2.nrow == 5
    assert grd2.nlay == 4

    assert grd2._xtgformat == 2
    assert grd2._actnumsv.shape == (3, 5, 4)
    assert grd2._coordsv.shape == (4, 6, 6)
    assert grd2._zcornsv.shape == (4, 6, 5, 4)


@st.composite
def ijk_ranges(
    draw,
    i=st.integers(min_value=1, max_value=3),
    j=st.integers(min_value=1, max_value=2),
    k=st.integers(min_value=1, max_value=4),
):
    ijkrange = list(draw(st.tuples(i, i, j, j, k, k)))
    if ijkrange[1] < ijkrange[0]:
        ijkrange[1], ijkrange[0] = ijkrange[0], ijkrange[1]
    if ijkrange[3] < ijkrange[2]:
        ijkrange[3], ijkrange[2] = ijkrange[2], ijkrange[3]
    if ijkrange[5] < ijkrange[4]:
        ijkrange[5], ijkrange[4] = ijkrange[4], ijkrange[5]
    ijkrange[1] += 1
    ijkrange[3] += 1
    ijkrange[5] += 1
    return ijkrange


@settings(
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(ijk_ranges())
def test_hdf5_partial_import(benchmark_grid, tmp_path, ijkrange):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = xtgeo.grid_from_file(fna, fformat="hdf", ijkrange=ijkrange)

    assert grd2.ncol == 1 + ijkrange[1] - ijkrange[0]
    assert grd2.nrow == 1 + ijkrange[3] - ijkrange[2]
    assert grd2.nlay == 1 + ijkrange[5] - ijkrange[4]

    assert grd2._xtgformat == 2
    assert grd2._actnumsv.shape == (grd2.ncol, grd2.nrow, grd2.nlay)
    assert grd2._coordsv.shape == (grd2.ncol + 1, grd2.nrow + 1, 6)
    assert grd2._zcornsv.shape == (grd2.ncol + 1, grd2.nrow + 1, grd2.nlay + 1, 4)

    assert (
        benchmark_grid._actnumsv[
            ijkrange[0] - 1 : ijkrange[1],
            ijkrange[3] - 1 : ijkrange[2],
            ijkrange[5] - 1 : ijkrange[4],
        ].all()
        == grd2._actnumsv.all()
    )
    assert (
        benchmark_grid._zcornsv[
            ijkrange[0] - 1 : ijkrange[1] + 1,
            ijkrange[3] - 1 : ijkrange[2] + 1,
            ijkrange[5] - 1 : ijkrange[4] + 1,
        ].all()
        == grd2._zcornsv.all()
    )
    assert (
        benchmark_grid._coordsv[
            ijkrange[0] - 1 : ijkrange[1] + 1,
            ijkrange[3] - 1 : ijkrange[2] + 1,
            ijkrange[5] - 1 : ijkrange[4] + 1,
        ].all()
        == grd2._coordsv.all()
    )


def test_hdf5_import(benchmark_grid, tmp_path):
    fname = join(tmp_path, "reek_geo_grid2.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    nlay = benchmark_grid.nlay
    benchmark_grid._subgrids = OrderedDict({"1": range(1, nlay + 1)})
    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = xtgeo.grid_from_file(fna, fformat="hdf")

    assert grd2._subgrids == OrderedDict({"1": range(1, nlay + 1)})
