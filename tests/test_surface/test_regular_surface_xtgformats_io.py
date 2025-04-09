# coding: utf-8
"""Testing new xtg formats both natiev ans hdf5 based."""

from os.path import join

import pytest
from numpy.testing import assert_allclose

import xtgeo


@pytest.fixture(name="benchmark_surface")
def benchmark_surface_fixture(testdata_path):
    return xtgeo.surface_from_file(
        join(testdata_path, "surfaces/reek/1/topreek_rota.gri")
    )


@pytest.mark.benchmark(group="import/export")
def test_benchmark_xtgregsurf_export(benchmark, tmp_path, benchmark_surface):
    """Test exporting to xtgregsurf format."""

    fname = tmp_path / "benchmark_surface.xtgregsurf"

    def write():
        benchmark_surface.to_file(fname, fformat="xtgregsurf")

    benchmark(write)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_xtgregsurf_import(benchmark, tmp_path, benchmark_surface):
    """Test exporting to xtgregsurf format."""

    fname = tmp_path / "benchmark_surface.xtgregsurf"

    fn = benchmark_surface.to_file(fname, fformat="xtgregsurf")

    surf2 = None

    def read():
        nonlocal surf2
        surf2 = xtgeo.surface_from_file(fn, fformat="xtgregsurf")

    benchmark(read)

    assert_allclose(benchmark_surface.values, surf2.values)


@pytest.mark.benchmark(group="import/export")
def test_surface_hdf5_export_blosc(benchmark, tmp_path, benchmark_surface):
    fname = tmp_path / "benchmark_surface.h5"

    def write():
        benchmark_surface.to_hdf(fname, compression="blosc")

    benchmark(write)


@pytest.mark.benchmark(group="import/export")
def test_surface_hdf5_import_blosc(benchmark, tmp_path, benchmark_surface):
    fname = tmp_path / "benchmark_surface.h5"

    fn = benchmark_surface.to_hdf(fname, compression="blosc")

    surf2 = None

    def read():
        nonlocal surf2
        surf2 = xtgeo.surface_from_file(fn, fformat="hdf")

    benchmark(read)

    assert_allclose(benchmark_surface.values, surf2.values)
