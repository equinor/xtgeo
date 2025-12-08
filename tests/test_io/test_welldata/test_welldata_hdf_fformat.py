"""Tests for HDF5 I/O of well data."""

from __future__ import annotations

import numpy as np
import pytest

from xtgeo.io.welldata._well_io import WellData, WellLog


@pytest.fixture
def simple_welldata():
    """Create a simple WellData object for testing."""
    n = 10
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    gr_log = WellLog(name="GR", values=np.linspace(50.0, 150.0, n), is_discrete=False)
    phit_log = WellLog(name="PHIT", values=np.linspace(0.1, 0.3, n), is_discrete=False)

    return WellData(
        name="TestWell-HDF5",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, phit_log),
    )


@pytest.fixture
def welldata_with_discrete_log():
    """Create WellData with discrete logs for testing."""
    n = 8
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    gr_log = WellLog(name="GR", values=np.random.rand(n) * 100, is_discrete=False)
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0, 3.0, 2.0, 1.0, 3.0, 2.0]),
        is_discrete=True,
        code_names={1: "SHALE", 2: "SAND", 3: "LIMESTONE"},
    )
    zone_log = WellLog(
        name="ZONE",
        values=np.array([10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0]),
        is_discrete=True,
        code_names={10: "ZONE_A", 20: "ZONE_B", 30: "ZONE_C"},
    )

    return WellData(
        name="WellWithZones",
        xpos=150.0,
        ypos=250.0,
        zpos=30.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(gr_log, facies_log, zone_log),
    )


def test_welldata_to_hdf5_simple(simple_welldata, tmp_path):
    """Test writing WellData to HDF5 file."""
    hdf_file = tmp_path / "test_well.h5"

    # Write to HDF5
    simple_welldata.to_hdf5(hdf_file)

    # Verify file was created
    assert hdf_file.exists()


def test_welldata_from_hdf5_simple(simple_welldata, tmp_path):
    """Test reading WellData from HDF5 file."""
    hdf_file = tmp_path / "test_well.h5"

    # Write and read back
    simple_welldata.to_hdf5(hdf_file)
    loaded_well = WellData.from_hdf5(hdf_file)

    # Verify basic properties
    assert loaded_well.name == simple_welldata.name
    assert loaded_well.xpos == pytest.approx(simple_welldata.xpos)
    assert loaded_well.ypos == pytest.approx(simple_welldata.ypos)
    assert loaded_well.zpos == pytest.approx(simple_welldata.zpos)
    assert loaded_well.n_records == simple_welldata.n_records

    # Verify survey arrays
    np.testing.assert_array_almost_equal(loaded_well.survey_x, simple_welldata.survey_x)
    np.testing.assert_array_almost_equal(loaded_well.survey_y, simple_welldata.survey_y)
    np.testing.assert_array_almost_equal(loaded_well.survey_z, simple_welldata.survey_z)

    # Verify logs
    assert len(loaded_well.logs) == len(simple_welldata.logs)
    assert loaded_well.log_names == simple_welldata.log_names


def test_welldata_hdf5_with_discrete_logs(welldata_with_discrete_log, tmp_path):
    """Test HDF5 I/O with discrete logs and code names."""
    hdf_file = tmp_path / "test_well_discrete.h5"

    # Write and read back
    welldata_with_discrete_log.to_hdf5(hdf_file)
    loaded_well = WellData.from_hdf5(hdf_file)

    # Verify discrete logs
    facies_log = loaded_well.get_log("FACIES")
    assert facies_log is not None
    assert facies_log.is_discrete
    assert facies_log.code_names == {1: "SHALE", 2: "SAND", 3: "LIMESTONE"}
    np.testing.assert_array_equal(
        facies_log.values, welldata_with_discrete_log.get_log("FACIES").values
    )

    zone_log = loaded_well.get_log("ZONE")
    assert zone_log is not None
    assert zone_log.is_discrete
    assert zone_log.code_names == {10: "ZONE_A", 20: "ZONE_B", 30: "ZONE_C"}


def test_welldata_hdf5_compression_options(simple_welldata, tmp_path):
    """Test HDF5 writing with different compression options."""
    # Test with lzf compression (default)
    hdf_lzf = tmp_path / "test_well_lzf.h5"
    simple_welldata.to_hdf5(hdf_lzf, compression="lzf")
    assert hdf_lzf.exists()
    loaded = WellData.from_hdf5(hdf_lzf)
    assert loaded.name == simple_welldata.name

    # Test with blosc compression
    hdf_blosc = tmp_path / "test_well_blosc.h5"
    simple_welldata.to_hdf5(hdf_blosc, compression="blosc")
    assert hdf_blosc.exists()
    loaded = WellData.from_hdf5(hdf_blosc)
    assert loaded.name == simple_welldata.name

    # Test with no compression
    hdf_none = tmp_path / "test_well_none.h5"
    simple_welldata.to_hdf5(hdf_none, compression=None)
    assert hdf_none.exists()
    loaded = WellData.from_hdf5(hdf_none)
    assert loaded.name == simple_welldata.name


def test_welldata_from_file_hdf5_format(simple_welldata, tmp_path):
    """Test reading WellData using from_file method with fformat='hdf5'."""
    hdf_file = tmp_path / "test_well.h5"

    # Write using to_hdf5
    simple_welldata.to_hdf5(hdf_file)

    # Read using from_file with fformat='hdf5'
    loaded_well = WellData.from_file(filepath=hdf_file, fformat="hdf5")

    # Verify
    assert loaded_well.name == simple_welldata.name
    assert loaded_well.n_records == simple_welldata.n_records
    assert loaded_well.log_names == simple_welldata.log_names


def test_welldata_to_file_hdf5_format(simple_welldata, tmp_path):
    """Test writing WellData using to_file method with fformat='hdf5'."""
    hdf_file = tmp_path / "test_well.h5"

    # Write using to_file with fformat='hdf5'
    simple_welldata.to_file(filepath=hdf_file, fformat="hdf5")

    # Verify file was created
    assert hdf_file.exists()

    # Read back and verify
    loaded_well = WellData.from_hdf5(hdf_file)
    assert loaded_well.name == simple_welldata.name
    assert loaded_well.n_records == simple_welldata.n_records


def test_welldata_hdf5_roundtrip_with_nan_values(tmp_path):
    """Test HDF5 I/O preserves NaN values in logs."""
    n = 5
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    # Create log with NaN values
    values_with_nan = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
    log_with_nan = WellLog(name="PHIT", values=values_with_nan, is_discrete=False)

    well = WellData(
        name="WellWithNaN",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(log_with_nan,),
    )

    hdf_file = tmp_path / "test_well_nan.h5"
    well.to_hdf5(hdf_file)
    loaded_well = WellData.from_hdf5(hdf_file)

    loaded_log = loaded_well.get_log("PHIT")
    assert loaded_log is not None

    # Verify NaN positions are preserved
    assert np.isnan(loaded_log.values[1])
    assert np.isnan(loaded_log.values[3])
    assert loaded_log.values[0] == pytest.approx(1.0)
    assert loaded_log.values[2] == pytest.approx(3.0)
    assert loaded_log.values[4] == pytest.approx(5.0)


def test_welldata_hdf5_empty_logs(tmp_path):
    """Test HDF5 I/O with well data that has no logs."""
    n = 3
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    well = WellData(
        name="EmptyLogsWell",
        xpos=100.0,
        ypos=200.0,
        zpos=25.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(),  # No logs
    )

    hdf_file = tmp_path / "test_well_empty_logs.h5"
    well.to_hdf5(hdf_file)
    loaded_well = WellData.from_hdf5(hdf_file)

    assert loaded_well.name == "EmptyLogsWell"
    assert loaded_well.n_records == n
    assert len(loaded_well.logs) == 0
    assert loaded_well.log_names == ()


def test_welldata_from_hdf5_file_not_found():
    """Test that from_hdf5 raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        WellData.from_hdf5("nonexistent_file.h5")


def test_welldata_hdf5_log_values_accuracy(tmp_path):
    """Test that log values are preserved with high accuracy."""
    n = 5
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    # Use specific values to test precision
    precise_values = np.array(
        [0.123456789, 1.987654321, 3.141592653, 2.718281828, 1.414213562]
    )
    precise_log = WellLog(name="PRECISE", values=precise_values, is_discrete=False)

    well = WellData(
        name="PrecisionTest",
        xpos=100.123,
        ypos=200.456,
        zpos=25.789,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(precise_log,),
    )

    hdf_file = tmp_path / "test_precision.h5"
    well.to_hdf5(hdf_file)
    loaded_well = WellData.from_hdf5(hdf_file)

    loaded_log = loaded_well.get_log("PRECISE")
    np.testing.assert_array_almost_equal(loaded_log.values, precise_values, decimal=6)


def test_welldata_hdf5_log_without_metadata(tmp_path):
    """Test reading HDF5 file where a log exists but is not in metadata."""
    import json

    import pandas as pd

    n = 5
    survey_x = np.linspace(100.0, 200.0, n)
    survey_y = np.linspace(200.0, 300.0, n)
    survey_z = np.linspace(-1000.0, -1100.0, n)

    # Create a manual HDF5 file with minimal metadata
    hdf_file = tmp_path / "test_minimal_meta.h5"

    # Create DataFrame with survey and one log
    data = {
        "X_UTME": survey_x,
        "Y_UTMN": survey_y,
        "Z_TVDSS": survey_z,
        "MYSTERY_LOG": np.random.rand(n),  # Log not in metadata
    }
    df = pd.DataFrame(data)

    # Minimal metadata without wlogs entry
    metadata = {
        "_required_": {
            "name": "MinimalMetaWell",
            "xpos": 100.0,
            "ypos": 200.0,
            "rkb": 25.0,
            # No wlogs key - this forces the else branch
        }
    }

    jmeta = json.dumps(metadata)

    # Write to HDF5
    with pd.HDFStore(hdf_file, "w") as store:
        store.put("Well", df)
        store.get_storer("Well").attrs["metadata"] = jmeta

    # Now read it back
    loaded_well = WellData.from_hdf5(hdf_file)

    # The MYSTERY_LOG should be loaded as continuous (default)
    mystery_log = loaded_well.get_log("MYSTERY_LOG")
    assert mystery_log is not None
    assert not mystery_log.is_discrete
    assert mystery_log.code_names is None
    assert len(mystery_log.values) == n
