"""
Tests for roxar RoxarAPI interface as mocks.

"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import xtgeo
from xtgeo.common._xyz_enum import _XYZType
from xtgeo.xyz import _xyz_roxapi


@pytest.fixture
def point_set():
    """Poinst is just a numpy array of size (nrows, 3)."""
    values = [
        (1.0, 2.0, 44.0),
        (1.1, 2.1, 45.0),
        (1.2, 2.2, 46.0),
        (1.3, 2.3, 47.0),
        (1.4, 2.4, 48.0),
    ]
    return np.array(values)


@pytest.fixture
def polygons_set():
    """Polygons is a list of numpy arrays."""
    values1 = [
        (1.0, 2.0, 44.0),
        (1.1, 2.1, 45.0),
        (1.2, 2.2, 46.0),
        (1.3, 2.3, 47.0),
        (1.4, 2.4, 48.0),
    ]
    values2 = [
        (5.0, 8.0, 64.0),
        (5.1, 8.1, 65.0),
        (5.2, 8.2, 66.0),
        (5.3, 8.3, 67.0),
        (5.4, 8.4, 68.0),
    ]
    return [np.array(values1), np.array(values2)]


@pytest.fixture
def mock_roxutils(mocker):
    mocker.patch("xtgeo.xyz._xyz_roxapi.RoxUtils")
    mocker.patch("xtgeo.xyz._xyz_roxapi._check_presence_in_project", return_value=True)


@pytest.fixture
def point_set_in_roxvalues(point_set, mocker):
    mocker.patch(
        "xtgeo.xyz._xyz_roxapi._get_roxvalues",
        return_value=point_set,
    )


@pytest.fixture
def polygon_set_in_roxvalues(polygons_set, mocker):
    mocker.patch(
        "xtgeo.xyz._xyz_roxapi._get_roxvalues",
        return_value=polygons_set,
    )


@pytest.mark.usefixtures("mock_roxutils", "point_set_in_roxvalues")
def test_load_points_from_roxar():
    poi = xtgeo.points_from_roxar("project", "Name", "Category")
    assert poi.get_dataframe()["X_UTME"][3] == 1.3


@pytest.mark.usefixtures("mock_roxutils", "point_set_in_roxvalues")
def test_points_invalid_stype():
    with pytest.raises(ValueError, match="Invalid stype"):
        xtgeo.points_from_roxar("project", "Name", "Category", stype="")


@pytest.mark.usefixtures("mock_roxutils", "polygon_set_in_roxvalues")
def test_polygons_invalid_stype():
    with pytest.raises(ValueError, match="Invalid stype"):
        xtgeo.polygons_from_roxar("project", "Name", "Category", stype="")


@pytest.mark.usefixtures("mock_roxutils", "polygon_set_in_roxvalues")
def test_load_polygons_from_roxar():
    pol = xtgeo.polygons_from_roxar("project", "Name", "Category")

    assert_frame_equal(
        pol.get_dataframe(),
        pd.DataFrame(
            [
                [1.0, 2.0, 44.0, 0],
                [1.1, 2.1, 45.0, 0],
                [1.2, 2.2, 46.0, 0],
                [1.3, 2.3, 47.0, 0],
                [1.4, 2.4, 48.0, 0],
                [5.0, 8.0, 64.0, 1],
                [5.1, 8.1, 65.0, 1],
                [5.2, 8.2, 66.0, 1],
                [5.3, 8.3, 67.0, 1],
                [5.4, 8.4, 68.0, 1],
            ],
            columns=["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"],
        ),
    )


def test_roxar_polygon_importer():
    """Test the _roxar_importer function with mocked dependencies."""

    # Mock the load_xyz_from_rms function
    mock_load = Mock(
        return_value={"values": "dummy_values", "attributes": {"attr1": "str"}}
    )

    # Create test inputs
    project = "dummy_project"
    name = "test_poly"
    category = "test_category"
    stype = "horizons"
    realisation = 0
    attributes = True

    # Patch the load_xyz_from_rms function
    with patch.object(_xyz_roxapi, "load_xyz_from_rms", mock_load):
        from xtgeo.xyz.polygons import _roxar_importer

        # Call the function
        result = _roxar_importer(
            project, name, category, stype, realisation, attributes
        )

        # Verify the mock was called with correct arguments
        mock_load.assert_called_once_with(
            project,
            name,
            category,
            stype,
            realisation,
            attributes,
            _XYZType.POLYGONS.value,
        )

        # Check the result
        assert result["name"] == "test_poly"
        assert "values" in result
        assert result["values"] == "dummy_values"


def test_roxar_polygon_importer_attrs():
    """Test the _roxar_importer function, include attributes."""

    # Mock the load_xyz_from_rms function with more detailed attributes
    mock_load = Mock(
        return_value={
            "values": "dummy_values",
            "attributes": {
                "attr1": {"values": [1, 2, 3], "dtype": "int"},
                "attr2": {"values": ["a", "b", "c"], "dtype": "str"},
                "attr3": {"values": [1.1, 2.2, 3.3], "dtype": "float"},
            },
        }
    )

    # Create test inputs
    project = "dummy_project"
    name = "test_poly"
    category = "test_category"
    stype = "horizons"
    realisation = 0
    attributes = ["attr1", "attr2", "attr3"]  # Now testing with specific attributes

    # Patch the load_xyz_from_rms function
    with patch.object(_xyz_roxapi, "load_xyz_from_rms", mock_load):
        from xtgeo.xyz.polygons import _roxar_importer

        # Call the function
        result = _roxar_importer(
            project, name, category, stype, realisation, attributes
        )

        # Verify the mock was called with correct arguments
        mock_load.assert_called_once_with(
            project,
            name,
            category,
            stype,
            realisation,
            attributes,
            _XYZType.POLYGONS.value,
        )

        # Check the result
        assert result["name"] == "test_poly"
        assert "values" in result
        assert result["values"] == "dummy_values"

        # Check attributes
        assert "attributes" in result
        attrs = result["attributes"]
        assert len(attrs) == 3
        assert "attr1" in attrs
        assert attrs["attr1"]["dtype"] == "int"
        assert attrs["attr2"]["dtype"] == "str"
        assert attrs["attr3"]["dtype"] == "float"
        assert attrs["attr1"]["values"] == [1, 2, 3]
        assert attrs["attr2"]["values"] == ["a", "b", "c"]
        assert attrs["attr3"]["values"] == [1.1, 2.2, 3.3]
