# pylint: disable=no-member
from unittest.mock import mock_open, patch

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

from xtgeo.grid3d import GridProperty
from xtgeo.grid3d._gridprop_import_grdecl import read_grdecl_3d_property

from .grid_generator import indecies
from .grid_generator import xtgeo_grids as grids


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(grids, st.data())
def test_gridprop_to_from_file_is_identity(tmp_path, grid, data):
    filepath = tmp_path / "gridprop.grdecl"

    prop = data.draw(st.sampled_from(grid.get_xyz_corners()))
    prop.to_file(filepath, fformat="grdecl")
    prop_from_file = GridProperty().from_file(
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


keywords = st.text(
    alphabet=st.characters(whitelist_categories=("Nd", "Lu")), min_size=1
)
grid_properties = arrays(
    elements=st.floats(), dtype="float", shape=st.tuples(indecies, indecies, indecies)
)


@given(keywords, grid_properties)
def test_read_write_grid_property_is_identity(keyword, grid_property):
    values = [str(v) for v in grid_property.flatten(order="F")]
    file_data = f"{keyword}\n {' '.join(values)} /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        assert_allclose(
            read_grdecl_3d_property(mock_file, keyword, grid_property.shape),
            grid_property,
        )


@given(keywords, grid_properties)
def test_read_grid_property_is_c_contiguous(keyword, grid_property):
    values = [str(v) for v in grid_property.flatten(order="F")]
    file_data = f"{keyword}\n {' '.join(values)} /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        assert read_grdecl_3d_property(mock_file, keyword, grid_property.shape).flags[
            "C_CONTIGUOUS"
        ]
