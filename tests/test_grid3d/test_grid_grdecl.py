import logging
import math
from unittest.mock import mock_open, patch

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

import xtgeo
import xtgeo.grid3d._ecl_grid as ecl_grid
import xtgeo.grid3d._grdecl_grid as ggrid
from xtgeo.grid3d import Grid
from xtgeo.grid3d._grdecl_format import open_grdecl
from xtgeo.grid3d._grid_import_ecl import grid_from_ecl_grid

from .grdecl_grid_generator import (
    grdecl_grids,
    specgrids,
    xtgeo_compatible_grdecl_grids,
)
from .grid_generator import xtgeo_grids


def test_grid_relative():
    assert ggrid.GridRelative.from_grdecl("MAP") == ggrid.GridRelative.MAP
    assert ggrid.GridRelative.from_grdecl("MAP     ") == ggrid.GridRelative.MAP
    assert ggrid.GridRelative.from_grdecl("") == ggrid.GridRelative.ORIGIN
    assert ggrid.GridRelative.from_grdecl("        ") == ggrid.GridRelative.ORIGIN


@pytest.mark.parametrize(
    "inp_str, values",
    [
        ("MAPAXES\n 0.0 1.0 0.0 0.0 1.0 0.0 /", [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        ("MAPAXES\n 0.0 1.0\n 0.0 0.0\n 1.0 0.0 /", [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]),
        ("MAPAXES\n 0.0 1.0 2*0.0 1.0 0.0 /", [0.0, 1.0, 0.0, 0.0, 1.0]),
        ("MAPAXES\n 123.0 1.0 2*0.0 1.0 0.0 /", [123.0, 1.0, 0.0, 0.0, 1.0]),
    ],
)
def test_mapaxes(inp_str, values):
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["MAPAXES"]) as kw:
            keyword, values = next(kw)
            assert keyword == "MAPAXES"
            mapaxes = ggrid.MapAxes.from_grdecl(values)
            assert list(mapaxes.origin) == [float(v) for v in values[2:4]]
            assert list(mapaxes.y_line) == [float(v) for v in values[0:2]]
            assert list(mapaxes.x_line) == [float(v) for v in values[4:6]]

            assert mapaxes.to_grdecl() == [float(v) for v in values]


def test_gdorient():
    inp_str = "GDORIENT\n INC INC INC DOWN LEFT /"
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["GDORIENT"]) as kw:
            keyword, values = next(kw)
            assert keyword == "GDORIENT"
            gdorient = ecl_grid.GdOrient.from_grdecl(values)

            assert gdorient.i_order == ecl_grid.Order.INCREASING
            assert gdorient.k_order == ecl_grid.Order.INCREASING
            assert gdorient.k_order == ecl_grid.Order.INCREASING
            assert gdorient.z_direction == ecl_grid.Orientation.DOWN
            assert gdorient.handedness == ecl_grid.Handedness.LEFT

            assert gdorient.to_grdecl() == values


def test_specgrid():
    inp_str = "SPECGRID\n 64 118 263 1 F /"
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["SPECGRID"]) as kw:
            keyword, values = next(kw)
            assert keyword == "SPECGRID"
            specgrid = ggrid.SpecGrid.from_grdecl(values)
            assert specgrid.ndivix == 64
            assert specgrid.ndiviy == 118
            assert specgrid.ndiviz == 263
            assert specgrid.numres == 1
            assert specgrid.coordinate_type == ggrid.CoordinateType.CARTESIAN

            assert [str(v) for v in specgrid.to_grdecl()] == values


@pytest.mark.parametrize(
    "inp_str, expected_unit, expected_relative",
    [
        (
            "GRIDUNIT\n 'METRES  ' '        ' /",
            ecl_grid.Units.METRES,
            ggrid.GridRelative.ORIGIN,
        ),
        ("GRIDUNIT\n METRES /", ecl_grid.Units.METRES, ggrid.GridRelative.ORIGIN),
        ("GRIDUNIT\n FEET MAP /", ecl_grid.Units.FEET, ggrid.GridRelative.MAP),
    ],
)
def test_gridunit(inp_str, expected_unit, expected_relative):
    with patch("builtins.open", mock_open(read_data=inp_str)) as mock_file:
        with open_grdecl(mock_file, keywords=["GRIDUNIT"]) as kw:
            keyword, values = next(kw)
            assert keyword == "GRIDUNIT"
            gridunit = ggrid.GridUnit.from_grdecl(values)

            assert gridunit.unit == expected_unit
            assert gridunit.grid_relative == expected_relative


@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(grdecl_grids(), st.sampled_from(["grdecl", "bgrdecl"]))
def test_grdecl_grid_read_write(tmp_path, grgrid, fileformat):
    assume(grgrid.mapaxes is None or fileformat != "bgrdecl")
    tmp_file = tmp_path / ("grid." + fileformat)
    grgrid.to_file(tmp_file, fileformat)
    assert ggrid.GrdeclGrid.from_file(tmp_file, fileformat) == grgrid


@given(grdecl_grids())
def test_grdecl_grid_cylindrical_raises(grgrid):
    assume(grgrid.specgrid.coordinate_type == ggrid.CoordinateType.CYLINDRICAL)
    with pytest.raises(NotImplementedError, match="cylindrical"):
        grgrid.xtgeo_coord()


@given(
    grdecl_grids(
        spec=specgrids(
            coordinates=st.just(ggrid.CoordinateType.CARTESIAN),
            numres=st.integers(min_value=1, max_value=10),
        )
    )
)
def test_grdecl_grid_multireservoir_raises(grgrid):
    assume(grgrid.specgrid.numres != 1)
    with pytest.raises(NotImplementedError, match="reservoir"):
        grgrid.xtgeo_coord()


@given(
    grdecl_grids(
        spec=st.just(ggrid.SpecGrid()),
        gunit=st.just(ggrid.GridUnit(grid_relative=ggrid.GridRelative.MAP)),
    )
)
def test_grdecl_grid_maprelative_raises(grgrid):
    assume(
        grgrid.gridunit is not None
        and grgrid.gridunit.grid_relative == ggrid.GridRelative.MAP
    )
    with pytest.raises(NotImplementedError, match="relative"):
        grgrid.xtgeo_coord()


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format2(xtggrid):
    xtggrid._xtgformat2()
    grdecl_grid = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)

    assert grdecl_grid.xtgeo_actnum().tolist() == xtggrid._actnumsv.tolist()
    assert grdecl_grid.xtgeo_coord() == pytest.approx(xtggrid._coordsv, abs=0.02)
    assert grdecl_grid.xtgeo_zcorn() == pytest.approx(xtggrid._zcornsv, abs=0.02)


@given(xtgeo_grids)
def test_to_from_xtgeogrid_format1(xtggrid):
    xtggrid._xtgformat1()
    grdecl_grid = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)

    xtggrid._xtgformat2()
    assert grdecl_grid.xtgeo_actnum().tolist() == xtggrid._actnumsv.tolist()
    assert grdecl_grid.xtgeo_coord() == pytest.approx(xtggrid._coordsv, abs=0.02)
    assert grdecl_grid.xtgeo_zcorn() == pytest.approx(xtggrid._zcornsv, abs=0.02)


@given(xtgeo_compatible_grdecl_grids)
def test_to_from_grdeclgrid(grdecl_grid):
    xtggrid = Grid(**grid_from_ecl_grid(grdecl_grid))

    grdecl_grid2 = ggrid.GrdeclGrid.from_xtgeo_grid(xtggrid)
    assert grdecl_grid2.xtgeo_actnum().tolist() == xtggrid._actnumsv.tolist()
    assert grdecl_grid2.xtgeo_coord() == pytest.approx(xtggrid._coordsv, abs=0.02)
    assert grdecl_grid2.xtgeo_zcorn() == pytest.approx(xtggrid._zcornsv, abs=0.02)


@settings(
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(xtgeo_compatible_grdecl_grids, st.sampled_from(["grdecl", "bgrdecl"]))
def test_to_from_xtggrid_write(tmp_path, grdecl_grid, fileformat):
    assume(grdecl_grid.mapaxes is None or fileformat != "bgrdecl")
    filepath = tmp_path / ("xtggrid." + fileformat)
    xtggrid = Grid(
        **grid_from_ecl_grid(grdecl_grid, relative_to=ggrid.GridRelative.ORIGIN)
    )

    xtggrid.to_file(filepath, fformat=fileformat)
    grdecl_grid2 = ggrid.GrdeclGrid.from_file(filepath, fileformat=fileformat)

    grdecl_grid2.convert_grid_units(grdecl_grid.grid_units)
    assert grdecl_grid.zcorn == pytest.approx(grdecl_grid2.zcorn, abs=0.02)
    assert grdecl_grid.coord == pytest.approx(grdecl_grid2.coord, abs=0.02)
    if grdecl_grid.actnum is None or grdecl_grid2.actnum is None:
        assert grdecl_grid.actnum is None or np.all(grdecl_grid.actnum)
        assert grdecl_grid2.actnum is None or np.all(grdecl_grid2.actnum)
    else:
        assert grdecl_grid.actnum.tolist() == grdecl_grid2.actnum.tolist()


@settings(
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(xtgeo_compatible_grdecl_grids, st.sampled_from(["grdecl", "bgrdecl"]))
def test_from_to_grdeclgrid_write(tmp_path, caplog, grdecl_grid, fileformat):
    caplog.set_level(logging.CRITICAL)
    assume(grdecl_grid.mapaxes is None or fileformat != "bgrdecl")

    grdecl_grid.to_file(tmp_path / ("xtggrid." + fileformat), fileformat)
    xtggrid = xtgeo.grid_from_file(
        tmp_path / ("xtggrid." + fileformat), fformat=fileformat
    )

    xtggrid._xtgformat2()
    if grdecl_grid.actnum is None:
        assert np.all(xtggrid._actnumsv)
    else:
        assert grdecl_grid.xtgeo_actnum().tolist() == xtggrid._actnumsv.tolist()
    assert grdecl_grid.xtgeo_coord() == pytest.approx(xtggrid._coordsv, abs=0.02)
    assert grdecl_grid.xtgeo_zcorn() == pytest.approx(xtggrid._zcornsv, abs=0.02)
    assert grdecl_grid.dimensions == xtggrid.dimensions


@given(xtgeo_compatible_grdecl_grids)
def test_xtgeo_values_are_c_contiguous(grdecl_grid):
    assert grdecl_grid.xtgeo_coord().flags["C_CONTIGUOUS"]
    assert grdecl_grid.xtgeo_actnum().flags["C_CONTIGUOUS"]
    assert grdecl_grid.xtgeo_zcorn().flags["C_CONTIGUOUS"]


@given(grdecl_grids())
def test_eq_reflexivity(grdecl_grid):
    assert grdecl_grid == grdecl_grid


@given(grdecl_grids(), grdecl_grids())
def test_eq_symmetry(grdecl_grid1, grdecl_grid2):
    if grdecl_grid1 == grdecl_grid2:
        assert grdecl_grid2 == grdecl_grid1


@given(grdecl_grids(), grdecl_grids(), grdecl_grids())
def test_eq_transitivity(grdecl_grid1, grdecl_grid2, grdecl_grid3):
    if grdecl_grid1 == grdecl_grid2 and grdecl_grid2 == grdecl_grid3:
        assert grdecl_grid1 == grdecl_grid3


def test_xtgeo_zcorn():
    zcorn = np.arange(32).reshape((2, 2, 2, 2, 2, 1), order="F")
    grdecl_grid = ggrid.GrdeclGrid(
        specgrid=ggrid.SpecGrid(2, 2, 1),
        zcorn=zcorn.ravel(order="F"),
        coord=np.arange(54),
    )
    xtgeo_zcorn = grdecl_grid.xtgeo_zcorn()
    assert xtgeo_zcorn[0, 0, 0].tolist() == [0, 0, 0, 0]
    assert xtgeo_zcorn[2, 2, 1].tolist() == [31, 31, 31, 31]

    # 1,1,1 means the upper corner of the middle line,
    # and xtgeo values are in the order sw, se, nw, ne
    # These are all upper values in the original zcorn
    west = 0
    east = 1
    south = 0
    north = 1
    assert xtgeo_zcorn[1, 1, 1].tolist() == [
        # sw of (1,1) corner line ne of cell at (0,0)
        zcorn[east, 0, north, 0, 1, 0],
        # se of (1,1) corner line is nw of cell at (1,0)
        zcorn[west, 1, north, 0, 1, 0],
        # nw of (1,1) corner line is se of cell at (0,1)
        zcorn[east, 0, south, 1, 1, 0],
        # ne of (1,1) corner line is sw of cell at (1,1)
        zcorn[west, 1, south, 1, 1, 0],
    ]


def test_duplicate_insignificant_values():
    zcorn = np.zeros((2, 1, 2, 1, 2, 1))
    zcorn_xtgeo = np.zeros((2, 2, 2, 4))
    grdecl_grid = ggrid.GrdeclGrid(
        specgrid=ggrid.SpecGrid(1, 1, 1),
        zcorn=zcorn.ravel(),
        coord=np.ones((2, 2, 6)).ravel(),
    )

    # set significant values to one
    zcorn_xtgeo[1:, 1:, :, 0] = 1
    zcorn_xtgeo[:2, 1:, :, 1] = 2
    zcorn_xtgeo[1:, :2, :, 2] = 3
    zcorn_xtgeo[:2, :2, :, 3] = 4

    grdecl_grid.duplicate_insignificant_xtgeo_zcorn(zcorn_xtgeo)

    assert zcorn_xtgeo.tolist() == [
        [[[4] * 4] * 2, [[2] * 4] * 2],
        [[[3] * 4] * 2, [[1] * 4] * 2],
    ]


@given(xtgeo_compatible_grdecl_grids)
def test_duplicate_insignificant_values_property(grid):
    nx, ny, nz = grid.dimensions
    xtgeo_zcorn = grid.xtgeo_zcorn()

    assert np.all(xtgeo_zcorn[nx, ny, :, :] == xtgeo_zcorn[nx, ny, :, 0, np.newaxis])
    assert np.all(xtgeo_zcorn[0, ny, :, :] == xtgeo_zcorn[0, ny, :, 1, np.newaxis])
    assert np.all(xtgeo_zcorn[nx, 0, :, :] == xtgeo_zcorn[nx, 0, :, 2, np.newaxis])
    assert np.all(xtgeo_zcorn[0, 0, :, :] == xtgeo_zcorn[0, 0, :, 3, np.newaxis])


def test_grdecl_ascii_layout(tmp_path):
    """Test that grdecl files produce proper formatted layout.

    No line wider than 128 chars (Eclipse limit)
    """

    # use math.pi by purpose to generate 'noisy' numbers for the test
    grd = xtgeo.create_box_grid(
        (3, 5, 2),
        origin=(672221.0 + math.pi, 234422.2 + math.pi, 2500.0 + math.pi),
        increment=(10 * math.pi, 11 * math.pi, math.pi),
        rotation=33.2 + math.pi,
    )
    grd.to_file(tmp_path / "some.grdecl", fformat="grdecl")

    with open(tmp_path / "some.grdecl", "r", encoding="utf8") as stream:
        for line in stream.readlines():
            assert len(line) < 128  # eclipse will not read beyond char 128

    newgrd = xtgeo.grid_from_file(tmp_path / "some.grdecl")

    np.testing.assert_array_almost_equal(newgrd._zcornsv, grd._zcornsv, decimal=7)
    np.testing.assert_array_almost_equal(newgrd._coordsv, grd._coordsv, decimal=1)

    gm1 = grd.get_geometrics(return_dict=True)
    gm2 = newgrd.get_geometrics(return_dict=True)
    assert gm2["avg_dx"] == pytest.approx(gm1["avg_dx"], rel=0.01)
    assert gm2["avg_dz"] == pytest.approx(gm1["avg_dz"], rel=0.01)


def test_grdecl_ascii_mixed_content(tmp_path):
    """Test grdecl files with a combination of grid and 'random' properties."""

    grd = xtgeo.create_box_grid((3, 5, 2))
    gfile = tmp_path / "xsome.grdecl"
    grd.to_file(gfile, fformat="grdecl")

    prop1 = xtgeo.GridProperty(grd, values=1.0, name="xsome")
    prop1.to_file(gfile, append=True, fformat="grdecl")

    prop2 = xtgeo.GridProperty(grd, values=2.0, name="a_quite_long_name")
    prop2.to_file(gfile, append=True, fformat="grdecl")

    grd2 = xtgeo.grid_from_file(gfile)

    np.testing.assert_array_almost_equal(grd2._zcornsv, grd._zcornsv, decimal=5)
    np.testing.assert_array_almost_equal(grd2._coordsv, grd._coordsv, decimal=5)

    prop = xtgeo.gridproperty_from_file(gfile, name="a_quite_long_name", grid=grd2)
    assert prop.values.mean() == pytest.approx(2.0)
