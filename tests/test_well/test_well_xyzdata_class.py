"""Test _XYZData class, in a Well context"""
import pandas as pd
import pytest
from xtgeo.common import _AttrType, _XYZData


@pytest.fixture(name="generate_data")
def fixture_generate_data() -> pd.DataFrame:
    """Make a test dataframe"""

    data = {
        "X_UTME": [1.3, 2.0, 3.0, 4.0, 5.2, 6.0, 9.0],
        "Y_UTMN": [11.0, 21.0, 31.0, 41.1, 51.0, 61.0, 91.0],
        "Z_TVDSS": [21.0, 22.0, 23.0, 24.0, 25.3, 26.0, 29.0],
        "MDEPTH": [13.0, 23.0, 33.0, 43.0, 53.2, 63.0, 93.0],
        "GR": [133.0, 2234.0, -999, 1644.0, 2225.5, 6532.0, 92.0],
        "FACIES": [1, -999, 3, 4, 4, 1, 1],
        "ZONES": [1, 2, 3, 3, 3, 4, -999],
    }

    return pd.DataFrame(data)


def test_well_xyzdata_initialize(generate_data: pd.DataFrame):
    """Initialize data with no attr_records and attr_types given.

    The init shall than then try to infer 'best' guess"""

    instance = _XYZData(generate_data)

    assert instance.dataframe.columns[0] == instance.xname
    assert instance.dataframe.columns[2] == instance.zname


def test_well_xyzdata_ensure_attr(generate_data: pd.DataFrame):
    """Testing private method _ensure_attr_types and _ensure_attr_records"""

    instance = _XYZData(generate_data)
    assert "FACIES" in instance._df.columns
    assert instance.get_attr_record("FACIES") == {1: "1", 3: "3", 4: "4"}
    assert instance.dataframe.FACIES.values.tolist() == [
        1.0,
        2000000000.0,
        3.0,
        4.0,
        4.0,
        1.0,
        1.0,
    ]

    del instance.dataframe["FACIES"]

    instance._ensure_consistency_attr_types()
    assert "FACIES" not in instance.dataframe.columns

    instance.dataframe["NEW"] = 1
    instance._ensure_consistency_attr_types()
    assert "NEW" in instance.dataframe.columns
    assert "NEW" in instance.attr_types
    assert instance.get_attr_type("NEW") == "DISC"

    instance._ensure_consistency_attr_records()
    assert instance.get_attr_record("NEW") == {1: "1"}


def test_infer_attr_dtypes(generate_data: pd.DataFrame):
    """Testing private method _infer_log_dtypes"""

    instance = _XYZData(generate_data)

    instance._attr_types = {}  # for testing, make private _attr_types empty

    instance._infer_attr_dtypes()
    res = instance._attr_types
    assert res["X_UTME"].name == "CONT"
    assert res["FACIES"].name == "DISC"

    # next, FACIES is predefined in attr_types prior to parsing; here as CONT
    # which shall 'win' in this setting
    instance._attr_types = {"FACIES": _AttrType.CONT}
    instance._infer_attr_dtypes()
    res = instance._attr_types
    assert res["X_UTME"].name == "CONT"
    assert res["FACIES"].name == "CONT"


def test_ensure_dataframe_dtypes(generate_data: pd.DataFrame):
    """Testing private method _ensure_cosistency_df_dtypes"""

    instance = _XYZData(generate_data, floatbits="float32")

    assert instance.data["FACIES"].dtype == "float32"
    instance.data["FACIES"] = instance.data["FACIES"].astype("int32")
    assert instance.data["FACIES"].dtype == "int32"

    instance._ensure_consistency_df_dtypes()
    assert instance.data["FACIES"].dtype == "float32"


def test_well_xyzdata_consistency_add_column(generate_data: pd.DataFrame):
    """Add column to the dataframe; check if attr_types and attr_records are updated."""

    instance = _XYZData(generate_data)

    assert instance.attr_types == {
        "X_UTME": _AttrType.CONT,
        "Y_UTMN": _AttrType.CONT,
        "Z_TVDSS": _AttrType.CONT,
        "MDEPTH": _AttrType.CONT,
        "GR": _AttrType.CONT,
        "FACIES": _AttrType.DISC,
        "ZONES": _AttrType.DISC,
    }

    instance.data["NEW"] = 1.992
    assert instance.ensure_consistency() is True

    assert instance.attr_types == {
        "X_UTME": _AttrType.CONT,
        "Y_UTMN": _AttrType.CONT,
        "Z_TVDSS": _AttrType.CONT,
        "MDEPTH": _AttrType.CONT,
        "GR": _AttrType.CONT,
        "FACIES": _AttrType.DISC,
        "ZONES": _AttrType.DISC,
        "NEW": _AttrType.CONT,
    }

    instance.data["DNEW"] = [1, -999, 3, 4, 4, 1, 1]
    assert instance.ensure_consistency() is True

    # rerun on SAME data shall not run ensure_consistency(), hence -> False
    assert instance.ensure_consistency() is False

    assert instance.attr_types == {
        "X_UTME": _AttrType.CONT,
        "Y_UTMN": _AttrType.CONT,
        "Z_TVDSS": _AttrType.CONT,
        "MDEPTH": _AttrType.CONT,
        "GR": _AttrType.CONT,
        "FACIES": _AttrType.DISC,
        "ZONES": _AttrType.DISC,
        "NEW": _AttrType.CONT,
        "DNEW": _AttrType.DISC,
    }

    empty = ("", "")

    assert instance.attr_records == {
        "X_UTME": empty,
        "Y_UTMN": empty,
        "Z_TVDSS": empty,
        "MDEPTH": empty,
        "GR": empty,
        "FACIES": {1: "1", 3: "3", 4: "4"},
        "ZONES": {1: "1", 2: "2", 3: "3", 4: "4"},
        "NEW": empty,
        "DNEW": {1: "1", 3: "3", 4: "4"},
    }


def test_attrtype_class():
    """Test the ENUM type _LogClass"""

    assert _AttrType.DISC.value == "DISC"
    assert _AttrType.CONT.value == "CONT"

    assert "CONT" in _AttrType.__members__
    assert "DISC" in _AttrType.__members__
    assert "FOO" not in _AttrType.__members__

    with pytest.raises(ValueError, match="is not a valid"):
        _AttrType("FOO")


def test_create_attr(generate_data: pd.DataFrame):
    """Try to create attribute"""
    instance = _XYZData(generate_data)
    print(instance.dataframe)

    instance.create_attr("NEWATTR", attr_type="CONT", value=823.0)
    print(instance.dataframe)
    assert instance.attr_records["NEWATTR"] == ("", "")


def test_create_attr_reserved_name(generate_data: pd.DataFrame):
    """Try to create attribute with a reserved name."""
    instance = _XYZData(generate_data)

    with pytest.raises(ValueError, match="The proposed name Q_AZI is a reserved name"):
        instance.create_attr("Q_AZI", attr_type="CONT", value=823.0)

    instance.create_attr("Q_AZI", attr_type="CONT", value=823.0, force_reserved=True)


def test_well_xyzdata_dataframe_copy(generate_data: pd.DataFrame):
    """Test get dataframe method, with option"""

    instance = _XYZData(generate_data, floatbits="float32")

    copy = instance.get_dataframe_copy()
    col = list(copy)

    dtypes = [str(entry) for entry in copy[col].dtypes]
    assert dtypes == [
        "float64",
        "float64",
        "float64",
        "float32",
        "float32",
        "float32",
        "float32",
    ]

    copy = instance.get_dataframe_copy(infer_dtype=True)

    dtypes = [str(entry) for entry in copy[col].dtypes]
    assert dtypes == [
        "float64",
        "float64",
        "float64",
        "float32",
        "float32",
        "int32",
        "int32",
    ]


def test_well_xyzdata_copy_attr(generate_data: pd.DataFrame):
    """Test copying an attribute."""

    instance = _XYZData(generate_data)

    assert instance.copy_attr("GR", "GR_copy") is True
    assert instance.copy_attr("GR", "GR_copy", force=True) is True
    assert instance.copy_attr("GR", "GR_copy", force=False) is False  # already there...

    assert instance.data["GR"].to_list() == instance.data["GR_copy"].to_list()
    assert instance.attr_records["GR"] == instance.attr_records["GR_copy"]

    instance.set_attr_record("GR", ("unit", "linear"))
    assert instance.attr_records["GR"] != instance.attr_records["GR_copy"]

    instance.copy_attr("GR", "GR_new2")
    assert instance.attr_records["GR"] == instance.attr_records["GR_new2"]
