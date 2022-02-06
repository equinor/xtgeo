# pylint: disable=no-member
from unittest.mock import mock_open, patch

import hypothesis.strategies as st
import numpy as np
import pytest
import xtgeo
from hypothesis import HealthCheck, assume, given, settings
from numpy.testing import assert_allclose
from xtgeo.grid3d._gridprop_import_grdecl import read_grdecl_3d_property

from .grid_generator import xtgeo_grids as grids
from .gridprop_generator import grid_properties, keywords


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(grids, st.data())
def test_gridprop_to_from_file_is_identity(tmp_path, grid, data):
    filepath = tmp_path / "gridprop.grdecl"

    prop = data.draw(st.sampled_from(grid.get_xyz_corners()))
    prop.to_file(filepath, fformat="grdecl")
    prop_from_file = xtgeo.gridproperty_from_file(
        filepath, name=prop.name, fformat="grdecl", grid=grid
    )

    assert_allclose(prop.get_npvalues1d(), prop_from_file.get_npvalues1d(), atol=1e-3)


@pytest.mark.parametrize(
    "file_data, shape, expected_value",
    [
        ("PROP\n 1 5 3 7 2 6 4 8 /\n", (2, 2, 2), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ("PROP\n 1 3 2 4 /\n", (1, 2, 2), [[[1, 2], [3, 4]]]),
    ],
)
def test_read_grdecl_3d_property(file_data, shape, expected_value):
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        np.array_equal(
            read_grdecl_3d_property(mock_file, "PROP", shape, int), expected_value
        )


@given(grid_properties())
def test_read_write_grid_property_is_identity(grid_property):
    values = [str(v) for v in grid_property.values.flatten(order="F")]
    file_data = f"{grid_property.name}\n {' '.join(values)} /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        assert_allclose(
            read_grdecl_3d_property(
                mock_file, grid_property.name, grid_property.values.shape
            ),
            grid_property.values,
        )


@given(grid_properties())
def test_read_grid_property_is_c_contiguous(grid_property):
    values = [str(v) for v in grid_property.values.flatten(order="F")]
    file_data = f"{grid_property.name}\n {' '.join(values)} /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        assert read_grdecl_3d_property(
            mock_file, grid_property.name, grid_property.values.shape
        ).flags["C_CONTIGUOUS"]


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("fformat", ["grdecl", "bgrdecl"])
@given(grid_properties())
def test_read_write_roundtrip(tmp_path, fformat, grid_property):
    filename = tmp_path / f"gridprop.{fformat}"
    grid_property.to_file(filename, fformat=fformat, name=grid_property.name)
    prop2 = xtgeo.gridproperty_from_file(
        filename, fformat=fformat, name=grid_property.name, grid=grid_property.geometry
    )
    if fformat == "grdecl":
        prop2.isdiscrete = grid_property.isdiscrete

    assert_allclose(grid_property.values, prop2.values, atol=0.01)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@pytest.mark.parametrize("fformat", ["grdecl", "bgrdecl"])
@given(grid_properties(), keywords)
def test_read_write_bad_name(tmp_path, fformat, grid_property, keyword):
    assume(keyword != grid_property.name)
    filename = tmp_path / f"gridprop.{fformat}"
    grid_property.to_file(filename, fformat=fformat, name=grid_property.name)

    with pytest.raises(xtgeo.KeywordNotFoundError):
        xtgeo.gridproperty_from_file(
            filename, fformat=fformat, name=keyword, grid=grid_property.geometry
        )
