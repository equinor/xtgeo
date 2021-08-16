# coding: utf-8
from os.path import join

import pytest
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

if not xtg.testsetup():
    raise SystemExit


@pytest.mark.benchmark(group="import/export")
def test_benchmark_cube_export(benchmark, tmp_path, testpath):
    cube1 = xtgeo.Cube(
        join(testpath, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = join(tmp_path, "syntseis_20030101_seismic_depth_stack.xtgrecube")

    def write():
        cube1.to_file(fname, fformat="xtgregcube")

    benchmark(write)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_cube_import(benchmark, testpath, tmp_path):
    cube1 = xtgeo.Cube(
        join(testpath, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = join(tmp_path, "syntseis_20030101_seismic_depth_stack.xtgrecube")
    cube1.to_file(fname, fformat="xtgregcube")

    cube2 = xtgeo.Cube()

    def read():
        cube2.from_file(fname, fformat="xtgregcube")

    benchmark(read)

    assert_allclose(cube1.values, cube2.values)
