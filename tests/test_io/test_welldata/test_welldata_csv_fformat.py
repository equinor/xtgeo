"""Tests for CSV I/O of blocked well data."""

from __future__ import annotations

import io

import numpy as np
import pytest

from xtgeo.io.welldata._blockedwell_io import BlockedWellData
from xtgeo.io.welldata._well_io import WellData


@pytest.fixture
def sample_csv_well_file(tmp_path):
    """Create a sample CSV file for testing WellData i/o."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT,PERM
25/11-16,464789.0625,6553551.625,1620.5,0.326,125.5
25/11-16,464789.0625,6553551.625,1621.5,0.316,98.2
25/11-16,464789.0625,6553551.625,1622.5,0.318,105.7
25/11-16,464789.0625,6553551.625,1623.5,0.315,92.3
25/11-16,464789.0625,6553551.625,1624.5,0.307,85.1
"""
    csv_file = tmp_path / "test_well.csv"
    csv_file.write_text(csv_content)
    return csv_file


def test_welldata_from_file_csv_format(sample_csv_well_file):
    """Test reading WellData using from_file method with fformat='csv'."""
    well = WellData.from_file(
        filepath=sample_csv_well_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Verify basic properties
    assert well.name == "25/11-16"
    assert well.n_records == 5

    # Verify header position (from first record)
    assert well.xpos == pytest.approx(464789.0625)
    assert well.ypos == pytest.approx(6553551.625)
    assert well.zpos == pytest.approx(1620.5)

    # Verify survey arrays
    assert len(well.survey_x) == 5
    assert len(well.survey_y) == 5
    assert len(well.survey_z) == 5

    # All X and Y values should be the same in this example
    assert np.all(well.survey_x == 464789.0625)
    assert np.all(well.survey_y == 6553551.625)

    # Z values should increase
    expected_z = np.array([1620.5, 1621.5, 1622.5, 1623.5, 1624.5])
    np.testing.assert_array_almost_equal(well.survey_z, expected_z)

    # Verify logs
    assert len(well.logs) == 2
    assert "PHIT" in well.log_names
    assert "PERM" in well.log_names

    # Check PHIT log
    phit = well.get_log("PHIT")
    assert phit is not None
    assert phit.name == "PHIT"
    assert not phit.is_discrete
    assert len(phit.values) == 5
    expected_phit = np.array([0.326, 0.316, 0.318, 0.315, 0.307])
    np.testing.assert_array_almost_equal(phit.values, expected_phit)

    # Check PERM log
    perm = well.get_log("PERM")
    assert perm is not None
    assert perm.name == "PERM"
    assert not perm.is_discrete
    assert len(perm.values) == 5
    expected_perm = np.array([125.5, 98.2, 105.7, 92.3, 85.1])
    np.testing.assert_array_almost_equal(perm.values, expected_perm)


@pytest.fixture
def sample_csv_blockedwell_file(tmp_path):
    """Create a sample CSV file for testing BlockedWellData i/o."""
    csv_content = """X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,WELLNAME,PHIT,PERM
464789.0625,6553551.625,1620.5,109,115,0,25/11-16,0.326,125.5
464789.0625,6553551.625,1621.5,109,115,1,25/11-16,0.316,98.2
464789.0625,6553551.625,1622.5,109,115,2,25/11-16,0.318,105.7
464789.0625,6553551.625,1623.5,109,115,3,25/11-16,0.315,92.3
464789.0625,6553551.625,1624.5,109,115,4,25/11-16,0.307,85.1
"""
    csv_file = tmp_path / "test_blockedwell_well.csv"
    csv_file.write_text(csv_content)
    return csv_file


def test_blockedwell_from_file_csv_format(sample_csv_blockedwell_file):
    """Test reading BlockedWellData using from_file method with fformat='csv'."""
    well = BlockedWellData.from_file(
        filepath=sample_csv_blockedwell_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Verify basic properties
    assert well.name == "25/11-16"
    assert well.n_records == 5
    assert well.n_blocked_cells == 5

    # Verify header position (from first record)
    assert well.xpos == pytest.approx(464789.0625)
    assert well.ypos == pytest.approx(6553551.625)
    assert well.zpos == pytest.approx(1620.5)

    # Verify survey arrays
    assert len(well.survey_x) == 5
    assert len(well.survey_y) == 5
    assert len(well.survey_z) == 5

    # Verify grid indices
    assert len(well.i_index) == 5
    assert len(well.j_index) == 5
    assert len(well.k_index) == 5

    # All I and J values should be the same
    assert np.all(well.i_index == 109)
    assert np.all(well.j_index == 115)

    # K values should increase from 0 to 4
    expected_k = np.array([0, 1, 2, 3, 4], dtype=np.float64)
    np.testing.assert_array_equal(well.k_index, expected_k)

    # Verify logs
    assert len(well.logs) == 2
    assert "PHIT" in well.log_names
    assert "PERM" in well.log_names

    # Check PHIT log
    phit = well.get_log("PHIT")
    assert phit is not None
    expected_phit = np.array([0.326, 0.316, 0.318, 0.315, 0.307])
    np.testing.assert_array_almost_equal(phit.values, expected_phit)


def test_welldata_from_file_uppercase_format(sample_csv_well_file):
    """Test that fformat is case-insensitive."""
    well = WellData.from_file(
        filepath=sample_csv_well_file,
        fformat="CSV",  # Uppercase
        wellname="25/11-16",
    )
    assert well.name == "25/11-16"
    assert well.n_records == 5


def test_welldata_from_file_invalid_format(sample_csv_well_file):
    """Test that invalid format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        WellData.from_file(
            filepath=sample_csv_well_file,
            fformat="invalid_format",
            wellname="25/11-16",
        )


def test_welldata_roundtrip_csv(sample_csv_well_file, tmp_path):
    """Test that reading and writing WellData preserves data."""
    # Read original
    well1 = WellData.from_file(
        filepath=sample_csv_well_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Write to new file using to_file
    output_file = tmp_path / "roundtrip.csv"
    well1.to_file(filepath=output_file, fformat="csv")

    # Read back
    well2 = WellData.from_file(
        filepath=output_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Verify data matches
    assert well1.name == well2.name
    assert well1.n_records == well2.n_records
    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z)

    # Check logs
    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        np.testing.assert_array_almost_equal(log1.values, log2.values)


def test_blockedwell_roundtrip_csv(sample_csv_blockedwell_file, tmp_path):
    """Test that reading and writing BlockedWellData preserves data."""
    # Read original
    well1 = BlockedWellData.from_file(
        filepath=sample_csv_blockedwell_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Write to new file using to_file
    output_file = tmp_path / "roundtrip_blocked.csv"
    well1.to_file(filepath=output_file, fformat="csv")

    # Read back
    well2 = BlockedWellData.from_file(
        filepath=output_file,
        fformat="csv",
        wellname="25/11-16",
    )

    # Verify data matches
    assert well1.name == well2.name
    assert well1.n_records == well2.n_records
    assert well1.n_blocked_cells == well2.n_blocked_cells

    np.testing.assert_array_almost_equal(well1.survey_x, well2.survey_x)
    np.testing.assert_array_almost_equal(well1.survey_y, well2.survey_y)
    np.testing.assert_array_almost_equal(well1.survey_z, well2.survey_z)

    np.testing.assert_array_almost_equal(well1.i_index, well2.i_index)
    np.testing.assert_array_almost_equal(well1.j_index, well2.j_index)
    np.testing.assert_array_almost_equal(well1.k_index, well2.k_index)

    # Check logs
    assert len(well1.logs) == len(well2.logs)
    for log1, log2 in zip(well1.logs, well2.logs):
        assert log1.name == log2.name
        np.testing.assert_array_almost_equal(log1.values, log2.values)


def test_welldata_from_csv_custom_columns(tmp_path):
    """Test reading WellData with custom column names."""
    csv_content = """WELL,EASTING,NORTHING,DEPTH,POR,K
TestWell,100.0,200.0,1000.0,0.25,150.0
TestWell,100.0,200.0,1001.0,0.30,200.0
TestWell,100.0,200.0,1002.0,0.28,175.0
"""
    csv_file = tmp_path / "custom_columns.csv"
    csv_file.write_text(csv_content)

    well = WellData.from_file(
        filepath=csv_file,
        fformat="csv",
        wellname="TestWell",
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
        wellname_col="WELL",
    )

    assert well.name == "TestWell"
    assert well.n_records == 3
    assert well.xpos == 100.0
    assert well.ypos == 200.0
    assert well.zpos == 1000.0

    # Verify logs were read (POR and K)
    assert len(well.logs) == 2
    assert "POR" in well.log_names
    assert "K" in well.log_names


def test_blockedwell_from_csv_custom_columns(tmp_path):
    """Test reading BlockedWellData with custom column names."""
    csv_content = """WELL,EASTING,NORTHING,DEPTH,I,J,K,POR
TestWell,100.0,200.0,1000.0,10,20,1,0.25
TestWell,100.0,200.0,1001.0,10,20,2,0.30
TestWell,100.0,200.0,1002.0,10,20,3,0.28
"""
    csv_file = tmp_path / "custom_blocked.csv"
    csv_file.write_text(csv_content)

    well = BlockedWellData.from_file(
        filepath=csv_file,
        fformat="csv",
        wellname="TestWell",
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
        i_indexname="I",
        j_indexname="J",
        k_indexname="K",
        wellname_col="WELL",
    )

    assert well.name == "TestWell"
    assert well.n_records == 3
    assert well.n_blocked_cells == 3

    # Verify grid indices
    assert np.all(well.i_index == 10)
    assert np.all(well.j_index == 20)
    expected_k = np.array([1, 2, 3], dtype=np.float64)
    np.testing.assert_array_equal(well.k_index, expected_k)


def test_welldata_to_file_custom_format(tmp_path):
    """Test writing WellData with to_file method."""
    # Create a simple WellData object
    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(),
    )

    # Write to CSV
    output_file = tmp_path / "output.csv"
    well.to_file(filepath=output_file, fformat="csv")

    # Verify file was created
    assert output_file.exists()

    # Read it back
    well2 = WellData.from_file(
        filepath=output_file,
        fformat="csv",
        wellname="TestWell",
    )

    assert well2.name == "TestWell"
    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.survey_z, well.survey_z)


def test_blockedwell_to_file_custom_format(tmp_path):
    """Test writing BlockedWellData with to_file method."""
    # Create a simple BlockedWellData object
    well = BlockedWellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        i_index=np.array([10.0, 10.0, 11.0]),
        j_index=np.array([20.0, 20.0, 21.0]),
        k_index=np.array([1.0, 2.0, 3.0]),
        logs=(),
    )

    # Write to CSV using to_file
    output_file = tmp_path / "output_blocked.csv"
    well.to_file(filepath=output_file, fformat="csv")

    # Verify file was created
    assert output_file.exists()

    # Read it back
    well2 = BlockedWellData.from_file(
        filepath=output_file,
        fformat="csv",
        wellname="TestWell",
    )

    assert well2.name == "TestWell"
    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.i_index, well.i_index)
    np.testing.assert_array_almost_equal(well2.k_index, well.k_index)


def test_welldata_to_file_invalid_format(tmp_path):
    """Test that invalid format in to_file raises ValueError."""
    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=np.array([100.0]),
        survey_y=np.array([200.0]),
        survey_z=np.array([1000.0]),
        logs=(),
    )

    output_file = tmp_path / "output.xyz"
    with pytest.raises(ValueError, match="Unsupported file format"):
        well.to_file(filepath=output_file, fformat="invalid_format")


def test_blockedwell_from_file_invalid_format(sample_csv_blockedwell_file):
    """Test that invalid format raises ValueError for BlockedWellData."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        BlockedWellData.from_file(
            filepath=sample_csv_blockedwell_file,
            fformat="invalid_format",
            wellname="25/11-16",
        )


def test_welldata_csv_from_stringio():
    """Test reading CSV WellData from StringIO stream."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT,PERM
TEST_WELL,100.0,200.0,1000.0,0.25,150.0
TEST_WELL,101.0,201.0,1001.0,0.30,200.0
TEST_WELL,102.0,202.0,1002.0,0.28,175.0
"""
    stream = io.StringIO(csv_content)
    well = WellData.from_file(filepath=stream, fformat="csv", wellname="TEST_WELL")

    assert well.name == "TEST_WELL"
    assert well.n_records == 3
    assert well.xpos == pytest.approx(100.0)

    phit = well.get_log("PHIT")
    np.testing.assert_array_almost_equal(phit.values, [0.25, 0.30, 0.28])


def test_welldata_csv_to_stringio():
    """Test writing CSV WellData to StringIO stream."""
    from xtgeo.io.welldata._well_io import WellLog

    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])

    poro_log = WellLog(
        name="Poro", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = WellData(
        name="STREAM_WELL",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=(poro_log,),
    )

    # Write to StringIO
    stream = io.StringIO()
    well.to_file(filepath=stream, fformat="csv")

    # Read back
    stream.seek(0)
    well2 = WellData.from_file(filepath=stream, fformat="csv", wellname="STREAM_WELL")

    assert well2.name == "STREAM_WELL"
    assert well2.n_records == 3
    np.testing.assert_array_almost_equal(well2.survey_x, survey_x)


def test_blockedwell_csv_from_stringio():
    """Test reading CSV BlockedWellData from StringIO stream."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,PHIT
BW_TEST,100.0,200.0,1000.0,10,20,1,0.25
BW_TEST,101.0,201.0,1001.0,10,20,2,0.30
BW_TEST,102.0,202.0,1002.0,10,20,3,0.28
"""
    stream = io.StringIO(csv_content)
    well = BlockedWellData.from_file(filepath=stream, fformat="csv", wellname="BW_TEST")

    assert well.name == "BW_TEST"
    assert well.n_records == 3
    np.testing.assert_array_equal(well.i_index, [10.0, 10.0, 10.0])
    np.testing.assert_array_equal(well.k_index, [1.0, 2.0, 3.0])


def test_blockedwell_csv_to_stringio():
    """Test writing CSV BlockedWellData to StringIO stream."""
    from xtgeo.io.welldata._well_io import WellLog

    survey_x = np.array([100.0, 101.0, 102.0])
    survey_y = np.array([200.0, 201.0, 202.0])
    survey_z = np.array([1000.0, 1001.0, 1002.0])
    i_index = np.array([10.0, 10.0, 10.0])
    j_index = np.array([20.0, 20.0, 20.0])
    k_index = np.array([1.0, 2.0, 3.0])

    poro_log = WellLog(
        name="Poro", values=np.array([0.25, 0.30, 0.28]), is_discrete=False
    )

    well = BlockedWellData(
        name="BW_STREAM",
        xpos=100.0,
        ypos=200.0,
        zpos=1000.0,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
        logs=(poro_log,),
    )

    # Write to StringIO
    stream = io.StringIO()
    well.to_file(filepath=stream, fformat="csv")

    # Read back
    stream.seek(0)
    well2 = BlockedWellData.from_file(
        filepath=stream, fformat="csv", wellname="BW_STREAM"
    )

    assert well2.name == "BW_STREAM"
    assert well2.n_records == 3
    np.testing.assert_array_equal(well2.i_index, i_index)
    np.testing.assert_array_equal(well2.k_index, k_index)


def test_welldata_csv_missing_required_columns(tmp_path):
    """Test that reading CSV with missing required columns raises ValueError."""
    csv_content = """WELLNAME,X_UTME,PHIT
WELL-1,100.0,0.25
WELL-1,101.0,0.26
"""
    csv_file = tmp_path / "missing_cols.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="Missing required columns"):
        WellData.from_csv(csv_file, wellname="WELL-1")


def test_welldata_csv_empty_wellname_column(tmp_path):
    """Test that reading CSV with no wells in wellname column raises ValueError."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
"""
    csv_file = tmp_path / "empty_wells.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="No wells found in CSV file"):
        WellData.from_csv(csv_file)


def test_welldata_csv_well_not_found(tmp_path):
    """Test that reading CSV for non-existent well raises ValueError."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
WELL-1,100.0,200.0,1000.0,0.25
WELL-2,101.0,201.0,1001.0,0.26
"""
    csv_file = tmp_path / "no_match.csv"
    csv_file.write_text(csv_content)

    with pytest.raises(ValueError, match="Well 'WELL-3' not found in CSV file"):
        WellData.from_csv(csv_file, wellname="WELL-3")


def test_welldata_csv_auto_select_first_well(tmp_path):
    """Test that when wellname=None, the first well is automatically selected."""
    csv_content = """WELLNAME,X_UTME,Y_UTMN,Z_TVDSS,PHIT
WELL-1,100.0,200.0,1000.0,0.25
WELL-1,101.0,201.0,1001.0,0.26
WELL-2,102.0,202.0,1002.0,0.27
"""
    csv_file = tmp_path / "multi_well.csv"
    csv_file.write_text(csv_content)

    # Should automatically select WELL-1
    well = WellData.from_csv(csv_file, wellname=None)

    assert well.name == "WELL-1"
    assert well.n_records == 2
