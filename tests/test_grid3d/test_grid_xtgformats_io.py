# coding: utf-8
"""Testing new xtgf and hdf5/h5 formats."""
import os
from os.path import join

import numpy as np
import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj


BIGBOX_DIMENSIONS = (100, 100, 20)
if "XTG_BIGTEST" in os.environ:
    BIGBOX_DIMENSIONS = (1000, 1000, 20)

# ======================================================================================
# Grid geometries:


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
    ids=["reek_grid", "big box"],
)
def benchmark_grid_fixture(request, testpath):
    return request.param(testpath)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_xtgf_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.xtgf")

    @benchmark
    def write():
        benchmark_grid.to_xtgf(fname)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_xtgf_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.xtgf")

    benchmark_grid.to_xtgf(fname)

    grid2 = xtgeo.Grid()

    @benchmark
    def read():
        grid2.from_xtgf(fname)

    assert_allclose(benchmark_grid._zcornsv, grid2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grid2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grid2._actnumsv)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_export(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    @benchmark
    def write():
        benchmark_grid._zcornsv += 1.0
        benchmark_grid.to_hdf(fname, compression=None)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_import_partial(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    benchmark_grid._zcornsv += 1.0
    fna = benchmark_grid.to_hdf(fname, compression=None)

    grd2 = xtgeo.Grid()

    @benchmark
    def partial_read():
        grd2.from_hdf(fna, ijkrange=(1, 20, 1, 20, "min", "max"))

    assert grd2.ncol == 20
    assert grd2.nlay == benchmark_grid.nlay


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_import(benchmark, tmp_path, benchmark_grid):
    fname = join(tmp_path, "reek_geo_grid.hdf")

    benchmark_grid._zcornsv += 1.0
    fna = benchmark_grid.to_hdf(fname, compression=None)

    grd2 = xtgeo.Grid()

    @benchmark
    def read():
        grd2.from_hdf(fna)

    assert_allclose(benchmark_grid._zcornsv, grd2._zcornsv)
    assert_allclose(benchmark_grid._coordsv, grd2._coordsv)
    assert_allclose(benchmark_grid._actnumsv, grd2._actnumsv)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_export_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    @benchmark
    def write():
        benchmark_grid.to_hdf(fname, compression="blosc")


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_import_partial_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = xtgeo.Grid()

    @benchmark
    def partial_read():
        grd2.from_hdf(fna, ijkrange=(1, 20, 1, 20, "min", "max"))

    assert grd2.ncol == 20
    assert grd2.nlay == benchmark_grid.nlay


@pytest.mark.benchmark(group="import/export")
def test_benchmark_grid_hdf5_import_blosc_compression(
    benchmark, tmp_path, benchmark_grid
):
    fname = join(tmp_path, "reek_geo_grid.compressed_h5")

    benchmark_grid._zcornsv += 1.0

    fna = benchmark_grid.to_hdf(fname, compression="blosc")

    grd2 = xtgeo.Grid()

    @benchmark
    def read():
        grd2.from_hdf(fna)

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


@pytest.mark.benchmark()
def test_benchmark_gridprop_export(benchmark, tmp_path, benchmark_gridprop):

    fname = tmp_path / "benchmark.xtgcpprop"

    @benchmark
    def write():
        benchmark_gridprop.to_file(fname, fformat="xtgcpprop")


@pytest.mark.benchmark()
def test_benchmark_gridprop_import(benchmark, tmp_path, benchmark_gridprop):
    """Test exporting etc to xtgcpprop format."""

    fname = tmp_path / "benchmark.xtgcpprop"

    benchmark_gridprop.to_file(fname, fformat="xtgcpprop")

    prop2 = xtgeo.GridProperty()

    @benchmark
    def read():
        prop2.from_file(fname, fformat="xtgcpprop")

    assert benchmark_gridprop.values.all() == prop2.values.all()


@pytest.mark.benchmark()
def test_benchmark_gridprop_import_partial(benchmark, tmp_path, benchmark_gridprop):
    """Test exporting etc to xtgcpprop format."""

    fname = tmp_path / "benchmark.xtgcpprop"

    benchmark_gridprop.to_file(fname, fformat="xtgcpprop")

    prop2 = xtgeo.GridProperty()

    @benchmark
    def read():
        prop2.from_file(fname, fformat="xtgcpprop", ijrange=(1, 2, 1, 2))

    assert benchmark_gridprop.values[0:2, 0:2, :].all() == prop2.values.all()
