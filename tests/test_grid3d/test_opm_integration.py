import os

import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_allclose

import xtgeo

from .grid_generator import xtgeo_grids


def deck_contents(dimensions, load_statement):
    nx, ny, nz = dimensions
    size = nx * ny * nz
    return f"""
RUNSPEC
DIMENS
 {nx} {ny} {nz} /
OIL
WATER
NOSIM
WELLDIMS
 0 /
EQLDIMS
/
TABDIMS
   1* 1* 30 30 /
GRID
{load_statement}
PORO
 {size}*0.1 /
PERMX
 {size}*150 /
PERMY
 {size}*150 /
PERMZ
 {size}*15 /
PROPS
SWOF
 0.0 0.0  1.0 0.0
 1.0 1.0  0   0.0 /
DENSITY
 3*40 /
PVTW
 1000 1.0 0.0 0.0 0.0 /
PVTO
  0.0 500.0  1.0 0.1
      5000.0 0.0 0.2  /
  3.0 1500.0 2.0 0.1
      5000.0 0.0 0.2  /
/
SOLUTION
EQUIL
       3*1000 0 1000 0 0 0 0 /
SUMMARY
SCHEDULE
NOSIM
END
"""


def convert_to_egrid(tmpdir, dimensions, grdecl):
    deck = tmpdir / "test.DATA"
    load_statement = f"INCLUDE\n '{grdecl}' /"
    with tmpdir.as_cwd():
        with open(deck, "w") as fh:
            fh.write(deck_contents(dimensions, load_statement))
        os.system(f"flow {deck}")
    return tmpdir / "TEST.EGRID"


def read_write_egrid(tmpdir, dimensions, egrid):
    deck = tmpdir / "test.DATA"
    load_statement = f"GDFILE\n '{egrid}' /"
    with tmpdir.as_cwd():
        with open(deck, "w") as fh:
            fh.write(deck_contents(dimensions, load_statement))
        os.system(f"flow {deck}")
    return tmpdir / "TEST.EGRID"


@pytest.mark.requires_opm
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(xtgeo_grids)
def test_grdecl_roundtrip(tmpdir, xtgeo_grid):
    grdecl_file = tmpdir / "xtg_grid.grdecl"
    xtgeo_grid.to_file(str(grdecl_file), fformat="grdecl")
    egrid_file = convert_to_egrid(tmpdir, xtgeo_grid.dimensions, grdecl_file)
    opm_grid = xtgeo.Grid(str(egrid_file), fformat="egrid")

    opm_grid._xtgformat2()
    xtgeo_grid._xtgformat2()
    assert opm_grid.dimensions == xtgeo_grid.dimensions
    assert_allclose(opm_grid._actnumsv, xtgeo_grid._actnumsv)
    assert_allclose(opm_grid._coordsv, xtgeo_grid._coordsv, atol=0.02)
    assert_allclose(opm_grid._zcornsv, xtgeo_grid._zcornsv, atol=0.02)


@pytest.mark.skip(
    reason="Currently fails as xtgeo egrid generation is not compatible with opm"
)
@pytest.mark.requires_opm
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(xtgeo_grids)
def test_egrid_roundtrip(tmpdir, xtgeo_grid):
    egrid_file = tmpdir / "xtg_grid.egrid"
    xtgeo_grid.to_file(str(egrid_file), fformat="egrid")
    opm_egrid_file = read_write_egrid(tmpdir, xtgeo_grid.dimensions, egrid_file)
    opm_grid = xtgeo.Grid(str(opm_egrid_file), fformat="egrid")

    opm_grid._xtgformat2()
    xtgeo_grid._xtgformat2()
    assert opm_grid.dimensions == xtgeo_grid.dimensions
    assert_allclose(opm_grid._actnumsv, xtgeo_grid._actnumsv)
    assert_allclose(opm_grid._coordsv, xtgeo_grid._coordsv, atol=0.02)
    assert_allclose(opm_grid._zcornsv, xtgeo_grid._zcornsv, atol=0.02)
