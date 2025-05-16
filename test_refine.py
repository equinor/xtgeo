import pathlib

import pytest

import xtgeo

EMEGFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
EMERFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_region.roff")

EMEGFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero_grid.roff")
EMEZFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero.roff")

DUAL = pathlib.Path("3dgrids/etc/dual_distorted2.grdecl")
DUALPROPS = pathlib.Path("3dgrids/etc/DUAL")


testdata_path = "../xtgeo-testdata"

emerald_grid = xtgeo.grid_from_file(testdata_path / EMEGFILE)

print(emerald_grid.dimensions)
assert emerald_grid.get_subgrids() == {"subgrid_0": 16, "subgrid_1": 30}

avg_dz1 = emerald_grid.get_dz().values.mean()

# idea; either a scalar (all cells), or a dictionary for zone wise
emerald_grid.refine_vertically(3)

avg_dz2 = emerald_grid.get_dz().values.mean()

print(avg_dz1, avg_dz2)

assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

assert emerald_grid.get_subgrids() == {"subgrid_0": 48, "subgrid_1": 90}
emerald_grid.inactivate_by_dz(0.001)
