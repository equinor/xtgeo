import os

import pytest
import xtgeo
from hypothesis import given, settings
from numpy.testing import assert_allclose

from ..test_grid3d.grid_generator import xtgeo_grids


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
  0.001 15.0  1.0 0.1 /
  1.6 5000.0  1.8 0.5
      9000.0  1.7 0.6 /
/
SOLUTION
EQUIL
 3*1000 0 1000 0 0 0 0 /
SUMMARY
SCHEDULE
NOSIM
END
"""


def convert_to_egrid(dimensions, grdecl):
    deck = "test.DATA"
    load_statement = f"INCLUDE\n '{grdecl}' /"
    with open(deck, "w") as fh:
        fh.write(deck_contents(dimensions, load_statement))
    os.system(f"flow {deck}")
    return "TEST.EGRID"


def read_write_egrid(dimensions, egrid):
    deck = "test.DATA"
    load_statement = f"GDFILE\n '{egrid}' /"
    with open(deck, "w") as fh:
        fh.write(deck_contents(dimensions, load_statement))
    os.system(f"flow {deck}")
    return "TEST.EGRID"


@pytest.mark.requires_opm
@pytest.mark.usefixtures("setup_tmpdir")
@settings(max_examples=5)
@given(xtgeo_grids)
def test_grdecl_roundtrip(xtgeo_grid):
    grdecl_file = "xtg_grid.grdecl"
    xtgeo_grid.to_file(str(grdecl_file), fformat="grdecl")
    egrid_file = convert_to_egrid(xtgeo_grid.dimensions, grdecl_file)
    opm_grid = xtgeo.Grid(str(egrid_file), fformat="egrid")

    opm_grid._xtgformat2()
    xtgeo_grid._xtgformat2()
    assert opm_grid.dimensions == xtgeo_grid.dimensions
    assert_allclose(opm_grid._actnumsv, xtgeo_grid._actnumsv)
    assert_allclose(opm_grid._coordsv, xtgeo_grid._coordsv, atol=0.02)
    assert_allclose(opm_grid._zcornsv, xtgeo_grid._zcornsv, atol=0.02)


@pytest.mark.usefixtures("setup_tmpdir")
@pytest.mark.requires_opm
@settings(max_examples=5)
@given(xtgeo_grids)
def test_egrid_roundtrip(xtgeo_grid):
    egrid_file = "xtg_grid.EGRID"
    # OPM 2021-04 will crash when given egrid files without actnum
    # missing actnum means all actnum[i]=1, so we trick xtgeo
    # to always output actnum here
    xtgeo_grid._actnumsv[0] = 0
    xtgeo_grid.to_file(str(egrid_file), fformat="egrid")
    opm_egrid_file = read_write_egrid(xtgeo_grid.dimensions, egrid_file)
    opm_grid = xtgeo.Grid(str(opm_egrid_file), fformat="egrid")

    opm_grid._xtgformat2()
    xtgeo_grid._xtgformat2()
    assert opm_grid.dimensions == xtgeo_grid.dimensions
    assert_allclose(opm_grid._actnumsv, xtgeo_grid._actnumsv)
    assert_allclose(opm_grid._coordsv, xtgeo_grid._coordsv, atol=0.02)
    assert_allclose(opm_grid._zcornsv, xtgeo_grid._zcornsv, atol=0.02)
