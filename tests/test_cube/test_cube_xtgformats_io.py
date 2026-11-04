# coding: utf-8
from __future__ import annotations

from os.path import join
from typing import TYPE_CHECKING

import pytest
from numpy.testing import assert_allclose

from xtgeo.cube import cube_from_file

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture


@pytest.mark.benchmark(group="import/export")
def test_benchmark_cube_export(
    benchmark: BenchmarkFixture, tmp_path: Path, testdata_path: str
) -> None:
    cube1 = cube_from_file(
        join(testdata_path, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = join(tmp_path, "syntseis_20030101_seismic_depth_stack.xtgrecube")

    def write() -> None:
        cube1.to_file(fname, fformat="xtgregcube")

    benchmark(write)


@pytest.mark.benchmark(group="import/export")
def test_benchmark_cube_import(
    benchmark: BenchmarkFixture, testdata_path: str, tmp_path: Path
) -> None:
    cube1 = cube_from_file(
        join(testdata_path, "cubes/reek/syntseis_20030101_seismic_depth_stack.segy")
    )

    fname = join(tmp_path, "syntseis_20030101_seismic_depth_stack.xtgrecube")
    cube1.to_file(fname, fformat="xtgregcube")

    cube2 = None

    def read() -> None:
        nonlocal cube2
        cube2 = cube_from_file(fname, fformat="xtgregcube")

    benchmark(read)

    assert cube2 is not None
    assert_allclose(cube1.values, cube2.values)
