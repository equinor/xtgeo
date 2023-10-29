import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from numpy.testing import assert_allclose

import xtgeo
import xtgeo.grid3d._egrid as xtg_egrid
import xtgeo.grid3d._grdecl_grid as ggrid
from xtgeo.grid3d._ecl_grid import (
    inverse_transform_xtgeo_coord_by_mapaxes,
    transform_xtgeo_coord_by_mapaxes,
)

from .egrid_generator import (
    egrids,
    xtgeo_compatible_egridheads,
    xtgeo_compatible_egrids,
)
from .grdecl_grid_generator import grdecl_grids, units
from .grid_generator import xtgeo_grids


def test_unit_conversion_factors():
    metres = ggrid.Units.METRES
    feet = ggrid.Units.FEET
    cm = ggrid.Units.CM

    assert metres.conversion_factor(metres) == 1.0
    assert metres.conversion_factor(cm) == 100
    assert metres.conversion_factor(feet) == pytest.approx(3.28084)

    assert cm.conversion_factor(metres) == 1e-2
    assert cm.conversion_factor(cm) == 1.0
    assert cm.conversion_factor(feet) == pytest.approx(0.0328084)

    assert feet.conversion_factor(metres) == pytest.approx(0.3048)
    assert feet.conversion_factor(cm) == pytest.approx(30.48)
    assert feet.conversion_factor(feet) == 1.0


def ecl_grids(*args, **kwargs):
    return st.one_of([grdecl_grids(*args, **kwargs), egrids(*args, **kwargs)])


@given(ecl_grids(), units)
def test_grdecl_convert_units(grid, unit):
    old_grid_unit = grid.grid_units
    old_zcorn = grid.zcorn.copy()
    old_coord = grid.coord.copy()
    old_actnum = None
    if grid.actnum is not None:
        old_actnum = grid.actnum.copy()

    grid.convert_grid_units(unit)

    assert grid.grid_units == unit

    factor = old_grid_unit.conversion_factor(unit)
    assert grid.zcorn.tolist() == pytest.approx(old_zcorn * factor)
    assert grid.coord.tolist() == pytest.approx(old_coord * factor)
    if old_actnum is not None:
        assert grid.actnum.tolist() == old_actnum.tolist()
    else:
        assert grid.actnum is None


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    xtgeo_grids, units, units, st.sampled_from(["grdecl", "bgrdecl", "egrid", "fegrid"])
)
def test_grid_to_file_conversion(tmp_path, xtgeo_grid, unit1, unit2, fformat):
    filepath = tmp_path / ("grid." + fformat)
    xtgeo_grid.units = unit1
    xtgeo_grid.to_file(filepath, fformat=fformat)
    eclgrid = None
    if fformat in ["grdecl", "bgrdecl"]:
        eclgrid = ggrid.GrdeclGrid.from_file(filepath, fileformat=fformat)
    elif fformat in ["egrid", "fegrid"]:
        eclgrid = xtg_egrid.EGrid.from_file(filepath, fileformat=fformat)
    else:
        assert False

    assert eclgrid.grid_units == unit1
    eclgrid.convert_grid_units(unit2)
    assert eclgrid.grid_units == unit2

    filepath2 = tmp_path / ("grid2." + fformat)
    eclgrid.to_file(filepath2, fileformat=fformat)
    xtgeo_grid2 = xtgeo.grid_from_file(filepath2, fformat="guess")
    xtgeo_grid2.convert_units(unit1)
    assert xtgeo_grid2.units == unit1
    assert_allclose(xtgeo_grid2._coordsv, xtgeo_grid._coordsv, atol=1e-4)
    assert_allclose(xtgeo_grid2._zcornsv, xtgeo_grid._zcornsv, atol=1e-4)


# axis systems that are nearly linearly dependent leads to bad numerical
# behavior (precision loss) when transforming so to simplify things we use a
# list of reasonable systems to test
good_axes = st.sampled_from(
    [
        xtg_egrid.MapAxes((0.5, 0.5), (0.1, 0.2), (0.25, 0.75)),
        xtg_egrid.MapAxes((1.0, 0.0), (0.0, 0.0), (0.0, 1.0)),
        xtg_egrid.MapAxes((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
        xtg_egrid.MapAxes((0.0, -1.0), (0.0, 0.0), (1.0, 0.0)),
        xtg_egrid.MapAxes((0.0, -1.0), (0.5, -0.5), (-1.0, 0.0)),
    ]
)


@given(egrids(), good_axes)
def test_inverse_transform_property(egrid, axes):
    nx, ny, _ = egrid.dimensions
    assume(egrid.valid_mapaxes(egrid.mapaxes))
    coord = np.swapaxes(egrid.coord.reshape((ny + 1, nx + 1, 6)), 0, 1).astype(
        np.float64
    )
    assert_allclose(
        inverse_transform_xtgeo_coord_by_mapaxes(
            axes, transform_xtgeo_coord_by_mapaxes(axes, coord.copy())
        ),
        coord,
        atol=1e-7,
    )


@given(xtgeo_compatible_egrids(head=xtgeo_compatible_egridheads(mpaxes=good_axes)))
def test_transform_map_relative(egrid):
    with pytest.warns(UserWarning):
        assert_allclose(
            egrid.xtgeo_coord(relative_to=xtgeo.GridRelative.MAP),
            transform_xtgeo_coord_by_mapaxes(
                egrid.mapaxes, egrid.xtgeo_coord(relative_to=xtgeo.GridRelative.ORIGIN)
            ),
            atol=1e-7,
        )


@given(xtgeo_compatible_egrids(head=xtgeo_compatible_egridheads(mpaxes=good_axes)))
def test_transform_map_relative_no_double(egrid):
    with pytest.warns(UserWarning):
        egrid.egrid_head.gridunit.grid_relative = xtgeo.GridRelative.MAP
        coord1 = egrid.xtgeo_coord(relative_to=xtgeo.GridRelative.MAP)

        egrid.egrid_head.gridunit.grid_relative = xtgeo.GridRelative.ORIGIN
        coord2 = egrid.xtgeo_coord(relative_to=xtgeo.GridRelative.ORIGIN)

        assert_allclose(coord1, coord2)


@given(xtgeo_compatible_egrids())
def test_conversion_warning(egrid):
    with pytest.warns(None) as warnlog:
        egrid.xtgeo_coord(relative_to=xtgeo.GridRelative.MAP)

    axis_unit_warnings = [w for w in warnlog if "Axis units" in str(w.message)]

    if egrid.mapaxes is not None and egrid.map_axis_units is None:
        assert len(axis_unit_warnings) == 1
    else:
        assert len(axis_unit_warnings) == 0
