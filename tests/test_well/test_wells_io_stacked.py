"""Test I/O (import and export) of multiple wells from/to stacked files."""

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

import xtgeo

logger = logging.getLogger(__name__)


# ============================================================================
# FIXTURES FOR IMPORT TESTS - Sample file contents
# ============================================================================


@pytest.fixture(name="sample_stacked_rms_well_content")
def fixture_sample_stacked_rms_well_content():
    """Sample RMS ASCII file with multiple wells stacked."""
    return """1.0
Undefined
WELL-A 100.0 200.0
0
100.0 200.0 1000.0
100.0 200.0 1001.0
100.0 200.0 1002.0

1.0
Undefined
WELL-B 150.0 250.0
0
150.0 250.0 2000.0
150.0 250.0 2001.0

1.0
Undefined
WELL-C 300.0 400.0
1
PHIT UNK lin
300.0 400.0 3000.0 0.25
300.0 400.0 3001.0 0.30
300.0 400.0 3002.0 0.28
300.0 400.0 3003.0 0.32
"""


@pytest.fixture(name="sample_stacked_blocked_well_content")
def fixture_sample_stacked_blocked_well_content():
    """Sample RMS ASCII file with multiple blocked wells stacked."""
    return """1.0
Undefined
BW-A 100.0 200.0
3
I_INDEX UNK lin
J_INDEX UNK lin
K_INDEX UNK lin
100.0 200.0 1000.0 10 20 1
100.0 200.0 1001.0 10 20 2
100.0 200.0 1002.0 10 20 3

1.0
Undefined
BW-B 150.0 250.0
4
I_INDEX UNK lin
J_INDEX UNK lin
K_INDEX UNK lin
PHIT UNK lin
150.0 250.0 2000.0 15 25 1 0.25
150.0 250.0 2001.0 15 25 2 0.30
"""


# ============================================================================
# FIXTURES FOR EXPORT TESTS - Sample Well and BlockedWell objects
# ============================================================================


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


# ============================================================================
# IMPORT TESTS - Reading from stacked files
# ============================================================================


def test_wells_from_stacked_rms_file(tmp_path, sample_stacked_rms_well_content):
    """Test reading multiple wells from a stacked RMS ASCII file."""
    # Create temporary file
    stacked_file = tmp_path / "stacked_wells.rmswell"
    stacked_file.write_text(sample_stacked_rms_well_content)

    # Read wells
    wells = xtgeo.wells_from_stacked_file(stacked_file, fformat="rms_ascii_stacked")

    # Verify we got 3 wells
    assert len(wells.wells) == 3

    # Verify well names
    well_names = [w.name for w in wells.wells]
    assert "WELL-A" in well_names
    assert "WELL-B" in well_names
    assert "WELL-C" in well_names

    # Verify number of records
    well_a = wells.get_well("WELL-A")
    well_b = wells.get_well("WELL-B")
    well_c = wells.get_well("WELL-C")

    assert len(well_a.get_dataframe()) == 3
    assert len(well_b.get_dataframe()) == 2
    assert len(well_c.get_dataframe()) == 4

    # Verify coordinates
    assert well_a.xpos == 100.0
    assert well_a.ypos == 200.0
    assert well_b.xpos == 150.0
    assert well_b.ypos == 250.0
    assert well_c.xpos == 300.0
    assert well_c.ypos == 400.0

    # Verify logs
    assert "X_UTME" in well_a.get_dataframe().columns
    assert "Y_UTMN" in well_a.get_dataframe().columns
    assert "Z_TVDSS" in well_a.get_dataframe().columns
    assert "PHIT" in well_c.get_dataframe().columns
    assert "PHIT" not in well_a.get_dataframe().columns  # WELL-A doesn't have PHIT


def test_blockedwells_from_stacked_rms_file(
    tmp_path, sample_stacked_blocked_well_content
):
    """Test reading multiple blocked wells from a stacked RMS ASCII file."""
    # Create temporary file
    stacked_file = tmp_path / "stacked_blocked_wells.rmswell"
    stacked_file.write_text(sample_stacked_blocked_well_content)

    # Read blocked wells
    bwells = xtgeo.blockedwells_from_stacked_file(
        stacked_file, fformat="rms_ascii_stacked"
    )

    # Verify we got 2 blocked wells
    assert len(bwells.wells) == 2

    # Verify well names
    well_names = [w.name for w in bwells.wells]
    assert "BW-A" in well_names
    assert "BW-B" in well_names

    # Verify number of records
    bw_a = bwells.get_blocked_well("BW-A")
    bw_b = bwells.get_blocked_well("BW-B")

    assert len(bw_a.get_dataframe()) == 3
    assert len(bw_b.get_dataframe()) == 2

    # Verify grid indices are present
    assert "I_INDEX" in bw_a.get_dataframe().columns
    assert "J_INDEX" in bw_a.get_dataframe().columns
    assert "K_INDEX" in bw_a.get_dataframe().columns

    # Verify grid index values
    assert bw_a.get_dataframe()["I_INDEX"].iloc[0] == 10.0
    assert bw_a.get_dataframe()["J_INDEX"].iloc[0] == 20.0
    assert bw_a.get_dataframe()["K_INDEX"].iloc[0] == 1.0

    assert bw_b.get_dataframe()["I_INDEX"].iloc[0] == 15.0
    assert bw_b.get_dataframe()["J_INDEX"].iloc[0] == 25.0


def test_wells_from_stacked_rms_with_alternative_alias(
    tmp_path, sample_stacked_rms_well_content
):
    """Test that alternative format aliases work."""
    stacked_file = tmp_path / "stacked_wells.rmswell"
    stacked_file.write_text(sample_stacked_rms_well_content)

    # Try with alternative alias
    wells = xtgeo.wells_from_stacked_file(stacked_file, fformat="rmswell_stacked")

    assert len(wells.wells) == 3


def test_wells_from_stacked_rms_default_format(
    tmp_path, sample_stacked_rms_well_content
):
    """Test that default format works for stacked files."""
    stacked_file = tmp_path / "stacked_wells.rmswell"
    stacked_file.write_text(sample_stacked_rms_well_content)

    # Use default format (should be rms_ascii_stacked)
    wells = xtgeo.wells_from_stacked_file(stacked_file)

    assert len(wells.wells) == 3


def test_wells_from_stacked_empty_file(tmp_path):
    """Test handling of empty stacked file."""
    stacked_file = tmp_path / "empty.rmswell"
    stacked_file.write_text("")

    wells = xtgeo.wells_from_stacked_file(stacked_file, fformat="rms_ascii_stacked")

    # Should return Wells object with empty wells (None or empty list)
    assert wells.wells is None or wells.wells == []


def test_wells_from_stacked_single_well(tmp_path):
    """Test that stacked format works with just one well."""
    content = """1.0
Undefined
SINGLE-WELL 100.0 200.0
0
100.0 200.0 1000.0
"""
    stacked_file = tmp_path / "single_well.rmswell"
    stacked_file.write_text(content)

    wells = xtgeo.wells_from_stacked_file(stacked_file, fformat="rms_ascii_stacked")

    assert len(wells.wells) == 1
    assert wells.wells[0].name == "SINGLE-WELL"


def test_wells_from_stacked_unsupported_format_error(
    tmp_path, sample_stacked_rms_well_content
):
    """Test that unsupported formats raise appropriate errors."""
    stacked_file = tmp_path / "stacked_wells.rmswell"
    stacked_file.write_text(sample_stacked_rms_well_content)

    # Try with unsupported format
    with pytest.raises(ValueError, match="Unsupported format"):
        xtgeo.wells_from_stacked_file(stacked_file, fformat="hdf5")


def test_wells_from_stacked_preserves_well_order(
    tmp_path, sample_stacked_rms_well_content
):
    """Test that wells are returned in the order they appear in the file."""
    stacked_file = tmp_path / "stacked_wells.rmswell"
    stacked_file.write_text(sample_stacked_rms_well_content)

    wells = xtgeo.wells_from_stacked_file(stacked_file, fformat="rms_ascii_stacked")

    well_names = [w.name for w in wells.wells]
    assert well_names == ["WELL-A", "WELL-B", "WELL-C"]


def test_blockedwells_from_stacked_with_phit_log(
    tmp_path, sample_stacked_blocked_well_content
):
    """Test that additional logs are preserved in blocked wells."""
    stacked_file = tmp_path / "stacked_blocked_wells.rmswell"
    stacked_file.write_text(sample_stacked_blocked_well_content)

    bwells = xtgeo.blockedwells_from_stacked_file(
        stacked_file, fformat="rms_ascii_stacked"
    )

    bw_b = bwells.get_blocked_well("BW-B")

    # BW-B should have PHIT log
    assert "PHIT" in bw_b.get_dataframe().columns
    assert bw_b.get_dataframe()["PHIT"].iloc[0] == pytest.approx(0.25)
    assert bw_b.get_dataframe()["PHIT"].iloc[1] == pytest.approx(0.30)


# ============================================================================
# STRINGIO TESTS - Reading from and writing to StringIO objects
# ============================================================================


def test_wells_from_stacked_rms_stringio(sample_stacked_rms_well_content):
    """Test reading wells from StringIO with RMS ASCII format."""
    sio = StringIO(sample_stacked_rms_well_content)
    wells = xtgeo.wells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(wells.wells) == 3
    assert set(wells.names) == {"WELL-A", "WELL-B", "WELL-C"}


def test_blockedwells_from_stacked_rms_stringio(sample_stacked_blocked_well_content):
    """Test reading blocked wells from StringIO with RMS ASCII format."""
    sio = StringIO(sample_stacked_blocked_well_content)
    bwells = xtgeo.blockedwells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(bwells.wells) == 2
    assert set(bwells.names) == {"BW-A", "BW-B"}


def test_wells_to_stacked_rms_stringio(sample_wells):
    """Test exporting wells to StringIO with RMS ASCII format."""
    wells_collection = xtgeo.Wells(sample_wells)
    sio = StringIO()

    # Export to StringIO
    result = wells_collection.to_stacked_file(sio, fformat="rms_ascii_stacked")
    assert result == sio

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.wells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(reimported.wells) == 3
    assert set(reimported.names) == {"WELL-A", "WELL-B", "WELL-C"}


def test_blockedwells_to_stacked_rms_stringio(sample_blocked_wells):
    """Test exporting blocked wells to StringIO with RMS ASCII format."""
    bwells_collection = xtgeo.BlockedWells(sample_blocked_wells)
    sio = StringIO()

    # Export to StringIO
    result = bwells_collection.to_stacked_file(sio, fformat="rms_ascii_stacked")
    assert result == sio

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.blockedwells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(reimported.wells) == 2
    assert set(reimported.names) == {"BW-A", "BW-B"}

    # Verify grid indices preserved
    bw_a = reimported.get_blocked_well("BW-A")
    assert "I_INDEX" in bw_a.get_dataframe().columns


def test_wells_to_stacked_csv_stringio(sample_wells):
    """Test roundtrip export/import with CSV format via StringIO."""
    wells_collection = xtgeo.Wells(sample_wells)
    sio = StringIO()

    # Export to CSV StringIO
    wells_collection.to_stacked_file(sio, fformat="csv")

    # Read back
    sio.seek(0)
    reimported = xtgeo.wells_from_stacked_file(sio, fformat="csv")

    assert len(reimported.wells) == 3
    assert set(reimported.names) == {"WELL-A", "WELL-B", "WELL-C"}

    # Verify data preserved
    well_a = reimported.get_well("WELL-A")
    assert well_a.xpos == pytest.approx(100.0)
    assert len(well_a.get_dataframe()) == 3


def test_blockedwells_to_stacked_csv_stringio(sample_blocked_wells):
    """Test roundtrip export/import of blocked wells with CSV format via StringIO."""
    bwells_collection = xtgeo.BlockedWells(sample_blocked_wells)
    sio = StringIO()

    # Export to CSV StringIO
    bwells_collection.to_stacked_file(sio, fformat="csv")

    # Read back
    sio.seek(0)
    reimported = xtgeo.blockedwells_from_stacked_file(sio, fformat="csv")

    assert len(reimported.wells) == 2

    # Verify grid indices preserved
    bw_a = reimported.get_blocked_well("BW-A")
    assert bw_a.get_dataframe()["I_INDEX"].iloc[0] == pytest.approx(10.0)


def test_wells_stringio_roundtrip_preserves_logs(sample_wells):
    """Test that StringIO roundtrip preserves log values exactly."""
    wells_collection = xtgeo.Wells(sample_wells)
    sio = StringIO()

    # Export and reimport via StringIO
    wells_collection.to_stacked_file(sio, fformat="rms_ascii_stacked")
    sio.seek(0)
    reimported = xtgeo.wells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    # Compare WELL-B which has both PHIT and PERM
    original_well_b = wells_collection.get_well("WELL-B")
    reimported_well_b = reimported.get_well("WELL-B")

    original_df = original_well_b.get_dataframe()
    reimported_df = reimported_well_b.get_dataframe()

    assert len(original_df) == len(reimported_df)
    assert reimported_df["PHIT"].iloc[0] == pytest.approx(original_df["PHIT"].iloc[0])
    assert reimported_df["PERM"].iloc[0] == pytest.approx(original_df["PERM"].iloc[0])


# ============================================================================
# EXPORT TESTS - Writing to stacked files
# ============================================================================


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


# ============================================================================
# EXPORT TESTS - Writing to separate files (to_files method)
# ============================================================================


def test_wells_to_files_rms_ascii(tmp_path, sample_wells):
    """Test exporting each well to separate RMS ASCII files."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "separate_wells"

    # Export to separate files
    created_files = wells_collection.to_files(output_dir, fformat="rms_ascii")

    # Verify correct number of files created
    assert len(created_files) == 3

    # Verify all files exist
    for filepath in created_files:
        assert (tmp_path / filepath).exists() or (
            output_dir / Path(filepath).name
        ).exists()

    # Verify default naming (wellname.w)
    expected_names = {"WELL-A.w", "WELL-B.w", "WELL-C.w"}
    actual_names = {Path(filepath).name for filepath in created_files}
    assert actual_names == expected_names

    # Read back one well and verify
    well_a_path = next(f for f in created_files if "WELL-A.w" in f)
    reimported = xtgeo.well_from_file(well_a_path, fformat="rms_ascii")

    assert reimported.name == "WELL-A"
    assert reimported.xpos == pytest.approx(100.0)
    assert reimported.ypos == pytest.approx(200.0)
    assert len(reimported.get_dataframe()) == 3


def test_wells_to_files_custom_template(tmp_path, sample_wells):
    """Test exporting with custom filename template."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "custom_names"

    # Export with custom template
    created_files = wells_collection.to_files(
        output_dir, fformat="rms_ascii", template="{wellname}_export.rmswell"
    )

    # Verify filenames use custom template
    expected_names = {
        "WELL-A_export.rmswell",
        "WELL-B_export.rmswell",
        "WELL-C_export.rmswell",
    }
    actual_names = {Path(filepath).name for filepath in created_files}
    assert actual_names == expected_names

    # Verify files exist and can be read
    for filepath in created_files:
        assert (tmp_path / filepath).exists() or (
            output_dir / Path(filepath).name
        ).exists()


def test_wells_to_files_csv_format(tmp_path, sample_wells):
    """Test exporting each well to separate CSV files."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "csv_wells"

    # Export to CSV format
    created_files = wells_collection.to_files(
        output_dir, fformat="csv", template="{wellname}.csv"
    )

    # Verify files created
    assert len(created_files) == 3
    expected_names = {"WELL-A.csv", "WELL-B.csv", "WELL-C.csv"}
    actual_names = {Path(filepath).name for filepath in created_files}
    assert actual_names == expected_names

    # Read back and verify
    well_b_path = next(f for f in created_files if "WELL-B.csv" in f)
    df = pd.read_csv(well_b_path)

    # Verify data
    assert len(df) == 2
    assert "X_UTME" in df.columns
    assert "PHIT" in df.columns
    assert "PERM" in df.columns
    assert df["X_UTME"].iloc[0] == pytest.approx(150.0)


def test_wells_to_files_csv_custom_columns(tmp_path, sample_wells):
    """Test CSV export with custom column names."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "custom_csv"

    # Export with custom column names
    created_files = wells_collection.to_files(
        output_dir,
        fformat="csv",
        template="{wellname}.csv",
        xname="EASTING",
        yname="NORTHING",
        zname="DEPTH",
    )

    # Read one file and check column names
    well_a_path = next(f for f in created_files if "WELL-A.csv" in f)
    df = pd.read_csv(well_a_path)

    assert "EASTING" in df.columns
    assert "NORTHING" in df.columns
    assert "DEPTH" in df.columns
    assert df["EASTING"].iloc[0] == pytest.approx(100.0)


def test_wells_to_files_hdf_format(tmp_path, sample_wells):
    """Test exporting each well to separate HDF5 files."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "hdf_wells"

    # Export to HDF format
    created_files = wells_collection.to_files(
        output_dir, fformat="hdf", template="{wellname}.hdf"
    )

    # Verify files created
    assert len(created_files) == 3
    expected_names = {"WELL-A.hdf", "WELL-B.hdf", "WELL-C.hdf"}
    actual_names = {Path(filepath).name for filepath in created_files}
    assert actual_names == expected_names

    # Verify files can be read back
    well_c_path = next(f for f in created_files if "WELL-C.hdf" in f)
    reimported = xtgeo.well_from_file(well_c_path, fformat="hdf")

    assert reimported.name == "WELL-C"
    assert len(reimported.get_dataframe()) == 4


def test_wells_to_files_creates_directory(tmp_path, sample_wells):
    """Test that to_files creates the output directory if it doesn't exist."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "new_directory" / "nested" / "path"

    # Directory doesn't exist yet
    assert not output_dir.exists()

    # Export - should create directory
    created_files = wells_collection.to_files(output_dir, fformat="rms_ascii")

    # Verify directory was created and files exist
    assert output_dir.exists()
    assert len(created_files) == 3
    for filepath in created_files:
        assert (tmp_path / filepath).exists() or (
            output_dir / Path(filepath).name
        ).exists()


def test_wells_to_files_empty_raises_error(tmp_path):
    """Test that exporting empty wells list raises ValueError."""
    empty_wells = xtgeo.Wells([])
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="Cannot export empty wells list"):
        empty_wells.to_files(output_dir, fformat="rms_ascii")


def test_blockedwells_to_files(tmp_path, sample_blocked_wells):
    """Test exporting blocked wells to separate files."""
    bwells_collection = xtgeo.BlockedWells(sample_blocked_wells)
    output_dir = tmp_path / "blocked_separate"

    # Export to separate files
    created_files = bwells_collection.to_files(output_dir, fformat="rms_ascii")

    # Verify files created
    assert len(created_files) == 2
    expected_names = {"BW-A.w", "BW-B.w"}
    actual_names = {Path(filepath).name for filepath in created_files}
    assert actual_names == expected_names

    # Read back and verify grid indices are preserved
    bw_a_path = next(f for f in created_files if "BW-A.w" in f)
    reimported = xtgeo.blockedwell_from_file(bw_a_path, fformat="rms_ascii")

    assert reimported.name == "BW-A"
    df = reimported.get_dataframe()
    assert "I_INDEX" in df.columns
    assert df["I_INDEX"].iloc[0] == pytest.approx(10.0)


def test_wells_to_files_roundtrip_preserves_data(tmp_path, sample_wells):
    """Test that roundtrip export/import preserves all well data."""
    wells_collection = xtgeo.Wells(sample_wells)
    output_dir = tmp_path / "roundtrip"

    # Export to separate files
    created_files = wells_collection.to_files(output_dir, fformat="rms_ascii")

    # Reimport all wells
    reimported_wells = []
    for filepath in created_files:
        well = xtgeo.well_from_file(filepath, fformat="rms_ascii")
        reimported_wells.append(well)

    reimported_collection = xtgeo.Wells(reimported_wells)

    # Verify same number of wells
    assert len(reimported_collection.wells) == len(wells_collection.wells)

    # Verify each well's data
    for original_name in wells_collection.names:
        original_well = wells_collection.get_well(original_name)
        reimported_well = reimported_collection.get_well(original_name)

        assert reimported_well is not None
        assert reimported_well.name == original_well.name
        assert reimported_well.xpos == pytest.approx(original_well.xpos)
        assert reimported_well.ypos == pytest.approx(original_well.ypos)

        original_df = original_well.get_dataframe()
        reimported_df = reimported_well.get_dataframe()
        assert len(reimported_df) == len(original_df)
