# coding: utf-8
"""Testing new xtg formats both natiev ans hdf5 based."""

from os.path import join

import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

if not xtg.testsetup():
    raise SystemExit


@pytest.fixture(name="benchmark_surface")
def benchmark_surface_fixture(testpath):
    return xtgeo.RegularSurface(join(testpath, "surfaces/reek/1/topreek_rota.gri"))


@pytest.mark.benchmark(group="import/export")
def test_benchmark_xtgregsurf_export(benchmark, tmp_path, benchmark_surface):
    """Test exporting to xtgregsurf format."""

    fname = tmp_path / "benchmark_surface.xtgregsurf"

    @benchmark
    def write():
        benchmark_surface.to_file(fname, fformat="xtgregsurf")


@pytest.mark.benchmark(group="import/export")
def test_benchmark_xtgregsurf_import(benchmark, tmp_path, benchmark_surface):
    """Test exporting to xtgregsurf format."""

    fname = tmp_path / "benchmark_surface.xtgregsurf"

    fn = benchmark_surface.to_file(fname, fformat="xtgregsurf")

    surf2 = xtgeo.RegularSurface()

    @benchmark
    def read():
        surf2.from_file(fn, fformat="xtgregsurf")

    assert_allclose(benchmark_surface.values, surf2.values)


@pytest.mark.benchmark(group="import/export")
def test_surface_hdf5_export_blosc(benchmark, tmp_path, benchmark_surface):
    fname = tmp_path / "benchmark_surface.h5"

    @benchmark
    def write():
        benchmark_surface.to_hdf(fname, compression="blosc")


@pytest.mark.benchmark(group="import/export")
def test_surface_hdf5_import_blosc(benchmark, tmp_path, benchmark_surface):
    fname = tmp_path / "benchmark_surface.h5"

    fn = benchmark_surface.to_hdf(fname, compression="blosc")

    surf2 = xtgeo.RegularSurface()

    @benchmark
    def read():
        surf2.from_hdf(fn)

    assert_allclose(benchmark_surface.values, surf2.values)
