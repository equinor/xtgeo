# pylint: disable=no-member
from unittest.mock import mock_open, patch

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays
from numpy.testing import assert_allclose

import xtgeo
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.grid3d._gridprop_import_grdecl import open_grdecl, read_grdecl_3d_property

from .grid_generator import indecies
from .grid_generator import xtgeo_grids as grids


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(grids)
def test_grid_to_from_grdecl_file_is_identity(tmp_path, grid):
    filepath = tmp_path / "grid.grdecl"
    grid.to_file(filepath, fformat="grdecl")
    grid_from_file = Grid().from_file(filepath, fformat="grdecl")

    assert grid.dimensions == grid_from_file.dimensions
    assert np.array_equal(grid.actnum_array, grid_from_file.actnum_array)

    for prop1, prop_from_file in zip(
        grid.get_xyz_corners(), grid_from_file.get_xyz_corners()
    ):
        assert_allclose(
            prop1.get_npvalues1d(), prop_from_file.get_npvalues1d(), atol=1e-3
        )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(grids)
def test_gridprop_to_from_file_is_identity(tmp_path, grid):
    filepath = tmp_path / "gridprop.grdecl"

    for prop in grid.get_xyz_corners():
        prop.to_file(filepath, fformat="grdecl")
        prop_from_file = GridProperty().from_file(
            filepath, name=prop.name, fformat="grdecl", grid=grid
        )

        assert_allclose(
            prop.get_npvalues1d(), prop_from_file.get_npvalues1d(), atol=1e-3
        )


@pytest.mark.parametrize(
    "file_data",
    [
        "PROP\n 1 2 3 4 / \n",
        "OTHERPROP\n 1 2 3 /\n 4 5 /\n PROP\n 1 2 3 4 / \n",
        "OTHERPROP\n 1 2 3 /\n 4 5 /\n/ PROP Eclipse comment\n PROP\n 1 2 3 4 / \n",
        "PROP\n 1 2 3 4 /",
        "PROP\n -- a comment \n 1 2 3 4 /",
        "-- a comment \n PROP\n \n 1 2 3 4 /",
        "PROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "NOECHO\n PROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "ECHO\n PROP\n \n 1 2 \n -- a comment \n 3 4 /",
        "NOECHO\n PROP\n \n 1 2 \n -- a comment \n 3 4 / \n ECHO",
    ],
)
def test_read_simple_property(file_data):
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            assert list(kw) == [("PROP", ["1", "2", "3", "4"])]


def test_read_extra_keyword_characters():
    file_data = (
        "LONGPROP Eclipse comment\n"
        "1 2 3 4 / More Eclipse comment\n OTHERPROP\n 5 6 7 8 /\n"
    )
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=["LONGPROP", "OTHERPROP"]) as kw:
            assert list(kw) == [
                ("LONGPROP", ["1", "2", "3", "4"]),
                ("OTHERPROP", ["5", "6", "7", "8"]),
            ]


def test_read_long_keyword():
    very_long_keyword = "a" * 200
    file_data = f"{very_long_keyword} Eclipse comment\n" "1 2 3 4 /"
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with open_grdecl(mock_file, keywords=[very_long_keyword]) as kw:
            assert list(kw) == [
                (very_long_keyword, ["1", "2", "3", "4"]),
            ]


@pytest.mark.parametrize(
    "undelimited_file_data",
    [
        "PROP\n 1 2 3 4 \n",
        "PROP\n 1 2 3 4 ECHO",
        "ECHO\n PROP\n 1 2 3 4",
        "PROP\n 1 2 3 4 -- a comment",
        "NOECHO\n PROP\n 1 2 3 4 -- a comment",
    ],
)
def test_read_prop_raises_error_when_no_forwardslash(undelimited_file_data):
    with patch(
        "builtins.open", mock_open(read_data=undelimited_file_data)
    ) as mock_file:
        with open_grdecl(mock_file, keywords=["PROP"]) as kw:
            with pytest.raises(ValueError):
                list(kw)


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


@pytest.mark.parametrize(
    "file_data, shape",
    [
        ("NOPROP\n 1 5 3 7 2 6 4 8 /\n", (2, 2, 2)),
    ],
)
def test_read_values_raises_on_missing(file_data, shape):
    with patch("builtins.open", mock_open(read_data=file_data)) as mock_file:
        with pytest.raises(xtgeo.KeywordNotFoundError):
            read_grdecl_3d_property(mock_file, "PROP", shape, int)


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
