"""
Tests for roxar RoxarAPI interface as mocks.

"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import xtgeo


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
    mocker.patch("xtgeo.xyz._xyz_roxapi._check_category_etc", return_value=True)


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
    assert poi.dataframe["X_UTME"][3] == 1.3


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
        pol.dataframe,
        pd.DataFrame(
            [
                [0, 1.0, 2.0, 44.0, 0],
                [1, 1.1, 2.1, 45.0, 0],
                [2, 1.2, 2.2, 46.0, 0],
                [3, 1.3, 2.3, 47.0, 0],
                [4, 1.4, 2.4, 48.0, 0],
                [0, 5.0, 8.0, 64.0, 1],
                [1, 5.1, 8.1, 65.0, 1],
                [2, 5.2, 8.2, 66.0, 1],
                [3, 5.3, 8.3, 67.0, 1],
                [4, 5.4, 8.4, 68.0, 1],
            ],
            columns=["index", "X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"],
        ),
    )
