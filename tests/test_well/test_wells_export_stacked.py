"""Test exporting multiple wells to a single stacked file."""

import logging

import pandas as pd
import pytest

import xtgeo

logger = logging.getLogger(__name__)


@pytest.fixture(name="sample_wells")
def fixture_sample_wells():
    """Create a few simple Well objects for testing."""
    wells = []

    # Well A - basic trajectory with one log
    df_a = pd.DataFrame(
        {
            "X_UTME": [100.0, 100.0, 100.0],
            "Y_UTMN": [200.0, 200.0, 200.0],
            "Z_TVDSS": [1000.0, 1001.0, 1002.0],
            "PHIT": [0.25, 0.26, 0.27],
        }
    )
    well_a = xtgeo.Well(xpos=100.0, ypos=200.0, rkb=0.0, wname="WELL-A", df=df_a)
    wells.append(well_a)

    # Well B - different location with two logs
    df_b = pd.DataFrame(
        {
            "X_UTME": [150.0, 150.0],
            "Y_UTMN": [250.0, 250.0],
            "Z_TVDSS": [2000.0, 2001.0],
            "PHIT": [0.30, 0.31],
            "PERM": [100.0, 110.0],
        }
    )
    well_b = xtgeo.Well(xpos=150.0, ypos=250.0, rkb=5.0, wname="WELL-B", df=df_b)
    wells.append(well_b)

    # Well C - different logs
    df_c = pd.DataFrame(
        {
            "X_UTME": [300.0, 300.0, 300.0, 300.0],
            "Y_UTMN": [400.0, 400.0, 400.0, 400.0],
            "Z_TVDSS": [3000.0, 3001.0, 3002.0, 3003.0],
            "GR": [50.0, 60.0, 55.0, 65.0],
        }
    )
    well_c = xtgeo.Well(xpos=300.0, ypos=400.0, rkb=10.0, wname="WELL-C", df=df_c)
    wells.append(well_c)

    return wells


@pytest.fixture(name="sample_blocked_wells")
def fixture_sample_blocked_wells():
    """Create a few simple BlockedWell objects for testing."""
    wells = []

    # Blocked Well A
    df_a = pd.DataFrame(
        {
            "X_UTME": [100.0, 100.0, 100.0],
            "Y_UTMN": [200.0, 200.0, 200.0],
            "Z_TVDSS": [1000.0, 1001.0, 1002.0],
            "I_INDEX": [10.0, 10.0, 10.0],
            "J_INDEX": [20.0, 20.0, 20.0],
            "K_INDEX": [1.0, 2.0, 3.0],
            "PHIT": [0.25, 0.26, 0.27],
        }
    )
    bw_a = xtgeo.BlockedWell(xpos=100.0, ypos=200.0, rkb=0.0, wname="BW-A", df=df_a)
    wells.append(bw_a)

    # Blocked Well B
    df_b = pd.DataFrame(
        {
            "X_UTME": [150.0, 150.0],
            "Y_UTMN": [250.0, 250.0],
            "Z_TVDSS": [2000.0, 2001.0],
            "I_INDEX": [15.0, 15.0],
            "J_INDEX": [25.0, 25.0],
            "K_INDEX": [1.0, 2.0],
            "PHIT": [0.30, 0.31],
        }
    )
    bw_b = xtgeo.BlockedWell(xpos=150.0, ypos=250.0, rkb=5.0, wname="BW-B", df=df_b)
    wells.append(bw_b)

    return wells


def test_wells_to_stacked_rms_file(tmp_path, sample_wells):
    """Test exporting multiple wells to a stacked RMS ASCII file."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "stacked_output.rmswell"

    # Export to stacked file
    result = wells_collection.to_stacked_file(output_file, fformat="rms_ascii_stacked")

    # Verify file was created
    assert output_file.exists()
    assert result == output_file

    # Read back and verify
    reimported = xtgeo.wells_from_stacked_file(output_file, fformat="rms_ascii_stacked")

    assert len(reimported.wells) == 3
    assert set(reimported.names) == {"WELL-A", "WELL-B", "WELL-C"}

    # Verify well details
    well_a = reimported.get_well("WELL-A")
    assert well_a.xpos == pytest.approx(100.0)
    assert well_a.ypos == pytest.approx(200.0)
    assert len(well_a.get_dataframe()) == 3

    well_b = reimported.get_well("WELL-B")
    assert well_b.xpos == pytest.approx(150.0)
    assert well_b.ypos == pytest.approx(250.0)
    assert len(well_b.get_dataframe()) == 2

    well_c = reimported.get_well("WELL-C")
    assert well_c.xpos == pytest.approx(300.0)
    assert well_c.ypos == pytest.approx(400.0)
    assert len(well_c.get_dataframe()) == 4


def test_wells_to_stacked_csv_file(tmp_path, sample_wells):
    """Test exporting multiple wells to a CSV file."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "stacked_output.csv"

    # Export to CSV
    result = wells_collection.to_stacked_file(output_file, fformat="csv")
    print(output_file)

    # Verify file was created
    assert output_file.exists()
    assert result == output_file

    # Read back and verify
    reimported = xtgeo.wells_from_stacked_file(output_file, fformat="csv")

    assert len(reimported.wells) == 3
    assert set(reimported.names) == {"WELL-A", "WELL-B", "WELL-C"}

    # Verify well details
    well_a = reimported.get_well("WELL-A")
    assert well_a.xpos == pytest.approx(100.0)
    assert well_a.ypos == pytest.approx(200.0)
    assert len(well_a.get_dataframe()) == 3

    well_b = reimported.get_well("WELL-B")
    assert well_b.xpos == pytest.approx(150.0)
    assert well_b.ypos == pytest.approx(250.0)
    assert len(well_b.get_dataframe()) == 2


def test_wells_to_csv_custom_columns(tmp_path, sample_wells):
    """Test CSV export with custom column names."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "custom_columns.csv"

    # Export with custom column names
    wells_collection.to_stacked_file(
        output_file,
        fformat="csv",
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
        wellname_col="WELL_NAME",
    )

    # Verify file was created
    assert output_file.exists()

    # Read the CSV to check column names
    df = pd.read_csv(output_file)
    assert "WELL_NAME" in df.columns
    assert "EASTING" in df.columns
    assert "NORTHING" in df.columns
    assert "DEPTH" in df.columns

    # Verify data
    well_a_data = df[df["WELL_NAME"] == "WELL-A"]
    assert len(well_a_data) == 3
    assert well_a_data["EASTING"].iloc[0] == pytest.approx(100.0)

    # Read back with custom column names
    reimported = xtgeo.wells_from_stacked_file(
        output_file,
        fformat="csv",
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
        wellname_col="WELL_NAME",
    )

    assert len(reimported.wells) == 3
    assert set(reimported.names) == {"WELL-A", "WELL-B", "WELL-C"}


def test_empty_wells_raises_error(tmp_path):
    """Test that exporting empty wells list raises ValueError."""
    empty_wells = xtgeo.Wells([])
    output_file = tmp_path / "empty.rmswell"

    with pytest.raises(ValueError, match="Cannot export empty wells list"):
        empty_wells.to_stacked_file(output_file, fformat="rms_ascii_stacked")

    output_file_csv = tmp_path / "empty.csv"
    with pytest.raises(ValueError, match="Cannot export empty wells list"):
        empty_wells.to_stacked_file(output_file_csv, fformat="csv")


def test_unsupported_format_raises_error(tmp_path, sample_wells):
    """Test that unsupported format raises ValueError."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "output.xyz"

    with pytest.raises(Exception, match="(Unsupported format|unknown or unsupported)"):
        wells_collection.to_stacked_file(output_file, fformat="unsupported_format")


def test_blockedwells_to_stacked_rms_file(tmp_path, sample_blocked_wells):
    """Test exporting multiple blocked wells to a stacked RMS ASCII file."""
    bwells_collection = xtgeo.BlockedWells(sample_blocked_wells)
    output_file = tmp_path / "stacked_blocked.rmswell"

    # Export to stacked file
    result = bwells_collection.to_stacked_file(output_file, fformat="rms_ascii_stacked")

    # Verify file was created
    assert output_file.exists()
    assert result == output_file

    # Read back and verify
    reimported = xtgeo.blockedwells_from_stacked_file(
        output_file, fformat="rms_ascii_stacked"
    )

    assert len(reimported.wells) == 2
    assert set(reimported.names) == {"BW-A", "BW-B"}

    # Verify well details
    bw_a = reimported.get_blocked_well("BW-A")
    assert bw_a.xpos == pytest.approx(100.0)
    assert bw_a.ypos == pytest.approx(200.0)
    assert len(bw_a.get_dataframe()) == 3

    # Verify grid indices are preserved
    assert "I_INDEX" in bw_a.get_dataframe().columns
    assert bw_a.get_dataframe()["I_INDEX"].iloc[0] == pytest.approx(10.0)


def test_blockedwells_to_stacked_csv_file(tmp_path, sample_blocked_wells):
    """Test exporting multiple blocked wells to a CSV file."""
    bwells_collection = xtgeo.BlockedWells(sample_blocked_wells)
    output_file = tmp_path / "stacked_blocked.csv"

    # Export to CSV
    result = bwells_collection.to_stacked_file(output_file, fformat="csv")

    # Verify file was created
    assert output_file.exists()
    assert result == output_file

    # Read back and verify
    reimported = xtgeo.blockedwells_from_stacked_file(output_file, fformat="csv")

    assert len(reimported.wells) == 2
    assert set(reimported.names) == {"BW-A", "BW-B"}

    # Verify grid indices are preserved
    bw_a = reimported.get_blocked_well("BW-A")
    assert "I_INDEX" in bw_a.get_dataframe().columns
    assert bw_a.get_dataframe()["I_INDEX"].iloc[0] == pytest.approx(10.0)
    assert bw_a.get_dataframe()["J_INDEX"].iloc[0] == pytest.approx(20.0)
    assert bw_a.get_dataframe()["K_INDEX"].iloc[0] == pytest.approx(1.0)


def test_roundtrip_preserves_log_values(tmp_path, sample_wells):
    """Test that log values are preserved in roundtrip export/import."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "roundtrip.rmswell"

    # Export and reimport
    wells_collection.to_stacked_file(output_file, fformat="rms_ascii_stacked")
    reimported = xtgeo.wells_from_stacked_file(output_file, fformat="rms_ascii_stacked")

    # Compare log values
    original_well_a = wells_collection.get_well("WELL-A")
    reimported_well_a = reimported.get_well("WELL-A")

    original_df = original_well_a.get_dataframe()
    reimported_df = reimported_well_a.get_dataframe()

    # Check PHIT values
    assert len(original_df) == len(reimported_df)
    for i in range(len(original_df)):
        assert reimported_df["PHIT"].iloc[i] == pytest.approx(
            original_df["PHIT"].iloc[i], abs=0.01
        )


def test_csv_roundtrip_preserves_log_values(tmp_path, sample_wells):
    """Test that log values are preserved in CSV roundtrip export/import."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "roundtrip.csv"

    # Export and reimport
    wells_collection.to_stacked_file(output_file, fformat="csv")
    reimported = xtgeo.wells_from_stacked_file(output_file, fformat="csv")

    # Compare log values for WELL-B (has PERM log)
    original_well_b = wells_collection.get_well("WELL-B")
    reimported_well_b = reimported.get_well("WELL-B")

    original_df = original_well_b.get_dataframe()
    reimported_df = reimported_well_b.get_dataframe()

    # Check both PHIT and PERM values
    assert len(original_df) == len(reimported_df)
    for i in range(len(original_df)):
        assert reimported_df["PHIT"].iloc[i] == pytest.approx(
            original_df["PHIT"].iloc[i]
        )
        assert reimported_df["PERM"].iloc[i] == pytest.approx(
            original_df["PERM"].iloc[i]
        )


def test_csv_export_includes_wellname_column(tmp_path, sample_wells):
    """Test that CSV export includes a wellname column for each row."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "with_wellname.csv"

    # Export to CSV
    wells_collection.to_stacked_file(output_file, fformat="csv")

    # Read the CSV and verify wellname column
    df = pd.read_csv(output_file)

    # Check that wellname column exists and has correct values
    assert "WELLNAME" in df.columns

    # Count rows per well
    well_a_rows = df[df["WELLNAME"] == "WELL-A"]
    well_b_rows = df[df["WELLNAME"] == "WELL-B"]
    well_c_rows = df[df["WELLNAME"] == "WELL-C"]

    assert len(well_a_rows) == 3
    assert len(well_b_rows) == 2
    assert len(well_c_rows) == 4

    # Verify total rows
    assert len(df) == 9  # 3 + 2 + 4


def test_csv_column_order(tmp_path, sample_wells):
    """Test that CSV columns are in the correct order."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_file = tmp_path / "column_order.csv"

    # Export to CSV
    wells_collection.to_stacked_file(output_file, fformat="csv")

    # Read the CSV
    df = pd.read_csv(output_file)

    # Check that wellname and coordinates are first
    columns = list(df.columns)
    assert columns[0] == "WELLNAME"
    assert columns[1] == "X_UTME"
    assert columns[2] == "Y_UTMN"
    assert columns[3] == "Z_TVDSS"
