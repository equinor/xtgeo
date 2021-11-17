"""
Tests for roxar RoxarAPI interface as mocks.

"""
from collections import OrderedDict

import numpy as np
import xtgeo


def get_points_set1():
    """Poinst is just a numpy array of size (nrows, 3)."""
    values = [
        (1.0, 2.0, 44.0),
        (1.1, 2.1, 45.0),
        (1.2, 2.2, 46.0),
        (1.3, 2.3, 47.0),
        (1.4, 2.4, 48.0),
    ]
    return np.array(values)


def get_points_set2(tmp_path):
    """Points with attributes via a tmp_file."""
    values = [
        (1.0, 2.0, 44.0, "some"),
        (1.1, 2.1, 45.0, "attr"),
        (1.2, 2.2, 46.0, "here"),
        (1.3, 2.3, 47.0, "my"),
        (1.4, 2.4, 48.0, "friend"),
    ]
    attrs = OrderedDict()
    attrs["TextColumn"] = "str"
    poi = xtgeo.Points(values=values, attributes=attrs)
    tfile = tmp_path / "generic_pset2.rsm_attr"
    poi.to_file(tfile, fformat="rms_attr")
    tfile2 = xtgeo._XTGeoFile(tmp_path / "generic_pset2.rsm_attr")
    args = xtgeo.xyz._xyz_io.import_rms_attr(tfile2)
    return args


def get_polygons_set1():
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


def test_load_points_from_roxar(mocker):
    mocker.patch("xtgeo.xyz._xyz_roxapi.RoxUtils")
    mocker.patch("xtgeo.xyz._xyz_roxapi._check_category_etc", return_value=True)
    mocker.patch(
        "xtgeo.xyz._xyz_roxapi._get_roxvalues",
        return_value=get_points_set1(),
    )
    poi = xtgeo.points_from_roxar("project", "Name", "Category")
    assert poi.dataframe["X_UTME"][3] == 1.3


def test_load_points_with_attrs_from_roxar(mocker, tmp_path):
    mocker.patch("xtgeo.xyz._xyz_roxapi.RoxUtils")
    mocker.patch("xtgeo.xyz._xyz_roxapi._get_roxar")
    mocker.patch("xtgeo.xyz._xyz_roxapi._check_category_etc", return_value=True)
    mocker.patch(
        "xtgeo.xyz._xyz_roxapi._roxapi_import_xyz_viafile",
        return_value=get_points_set2(tmp_path),
    )
    poi = xtgeo.points_from_roxar("project", "Name", "Category", attributes=True)
    print(poi.dataframe)
    assert poi.dataframe["X_UTME"][3] == 1.3


def test_load_polygons_from_roxar(mocker):
    mocker.patch("xtgeo.xyz._xyz_roxapi.RoxUtils")
    mocker.patch("xtgeo.xyz._xyz_roxapi._check_category_etc", return_value=True)
    mocker.patch(
        "xtgeo.xyz._xyz_roxapi._get_roxvalues",
        return_value=get_polygons_set1(),
    )
    pol = xtgeo.polygons_from_roxar("project", "Name", "Category")
    print(pol.dataframe)


#    assert poi.dataframe["X_UTME"][3] == 1.3
