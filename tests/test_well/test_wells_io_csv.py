"""Test I/O of multiple wells from a single CSV file."""

import logging

import pandas as pd
import pytest

import xtgeo

logger = logging.getLogger(__name__)


@pytest.fixture(name="sample_csv_wells_content")
def fixture_sample_csv_wells_content():
    """Sample CSV file with multiple wells."""
    return """X_UTME,Y_UTMN,Z_TVDSS,WELLNAME,PHIT,PERM
464789.0625,6553551.625,1620.5,WELL-A,0.326,100.5
464789.0625,6553551.625,1621.5,WELL-A,0.316,95.3
464789.0625,6553551.625,1622.5,WELL-A,0.318,98.2
464790.0625,6553552.625,1620.5,WELL-B,0.300,80.1
464790.0625,6553552.625,1621.5,WELL-B,0.310,85.4
464791.0625,6553553.625,1620.5,WELL-C,0.280,70.5
464791.0625,6553553.625,1621.5,WELL-C,0.290,75.2
464791.0625,6553553.625,1622.5,WELL-C,0.285,72.8
464791.0625,6553553.625,1623.5,WELL-C,0.295,76.9
"""


@pytest.fixture(name="sample_csv_blocked_wells_content")
def fixture_sample_csv_blocked_wells_content():
    """Sample CSV file with multiple blocked wells."""
    return """X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,WELLNAME,PHIT
464789.0625,6553551.625,1620.5,109,115,0,BW-A,0.326
464789.0625,6553551.625,1621.5,109,115,1,BW-A,0.316
464789.0625,6553551.625,1622.5,109,115,2,BW-A,0.318
464790.0625,6553552.625,1620.5,110,116,0,BW-B,0.300
464790.0625,6553552.625,1621.5,110,116,1,BW-B,0.310
464790.0625,6553552.625,1622.5,110,116,2,BW-B,0.305
464791.0625,6553553.625,1620.5,111,117,0,BW-C,0.280
464791.0625,6553553.625,1621.5,111,117,1,BW-C,0.290
"""


def test_wells_from_csv_file(tmp_path, sample_csv_wells_content):
    """Test reading multiple wells from a CSV file."""
    # Create temporary CSV file
    csv_file = tmp_path / "multi_wells.csv"
    csv_file.write_text(sample_csv_wells_content)

    # Read wells
    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

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


def test_wells_from_csv_verifies_coordinates(tmp_path, sample_csv_wells_content):
    """Test that coordinates are correctly set from CSV data."""
    csv_file = tmp_path / "multi_wells.csv"
    csv_file.write_text(sample_csv_wells_content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

    well_a = wells.get_well("WELL-A")
    well_b = wells.get_well("WELL-B")
    well_c = wells.get_well("WELL-C")

    # Verify well positions (from first record)
    assert well_a.xpos == pytest.approx(464789.0625)
    assert well_a.ypos == pytest.approx(6553551.625)

    assert well_b.xpos == pytest.approx(464790.0625)
    assert well_b.ypos == pytest.approx(6553552.625)

    assert well_c.xpos == pytest.approx(464791.0625)
    assert well_c.ypos == pytest.approx(6553553.625)


def test_wells_from_csv_preserves_logs(tmp_path, sample_csv_wells_content):
    """Test that log data is preserved from CSV."""
    csv_file = tmp_path / "multi_wells.csv"
    csv_file.write_text(sample_csv_wells_content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")
    well_a = wells.get_well("WELL-A")

    # Verify logs are present
    df = well_a.get_dataframe()
    assert "X_UTME" in df.columns
    assert "Y_UTMN" in df.columns
    assert "Z_TVDSS" in df.columns
    assert "PHIT" in df.columns
    assert "PERM" in df.columns

    # Verify log values
    assert df["PHIT"].iloc[0] == pytest.approx(0.326)
    assert df["PERM"].iloc[0] == pytest.approx(100.5)


def test_blockedwells_from_csv_file(tmp_path, sample_csv_blocked_wells_content):
    """Test reading multiple blocked wells from a CSV file."""
    csv_file = tmp_path / "multi_blocked_wells.csv"
    csv_file.write_text(sample_csv_blocked_wells_content)

    # Read blocked wells
    bwells = xtgeo.blockedwells_from_stacked_file(csv_file, fformat="csv")

    # Verify we got 3 blocked wells
    assert len(bwells.wells) == 3

    # Verify well names
    well_names = [w.name for w in bwells.wells]
    assert "BW-A" in well_names
    assert "BW-B" in well_names
    assert "BW-C" in well_names

    # Verify number of records
    bw_a = bwells.get_blocked_well("BW-A")
    bw_b = bwells.get_blocked_well("BW-B")
    bw_c = bwells.get_blocked_well("BW-C")

    assert len(bw_a.get_dataframe()) == 3
    assert len(bw_b.get_dataframe()) == 3
    assert len(bw_c.get_dataframe()) == 2


def test_blockedwells_from_csv_preserves_grid_indices(
    tmp_path, sample_csv_blocked_wells_content
):
    """Test that grid indices are preserved for blocked wells."""
    csv_file = tmp_path / "multi_blocked_wells.csv"
    csv_file.write_text(sample_csv_blocked_wells_content)

    bwells = xtgeo.blockedwells_from_stacked_file(csv_file, fformat="csv")
    bw_a = bwells.get_blocked_well("BW-A")

    # Verify grid indices are present
    df = bw_a.get_dataframe()
    assert "I_INDEX" in df.columns
    assert "J_INDEX" in df.columns
    assert "K_INDEX" in df.columns

    # Verify grid index values
    assert df["I_INDEX"].iloc[0] == 109.0
    assert df["J_INDEX"].iloc[0] == 115.0
    assert df["K_INDEX"].iloc[0] == 0.0

    assert df["K_INDEX"].iloc[1] == 1.0
    assert df["K_INDEX"].iloc[2] == 2.0


def test_wells_from_csv_preserves_well_order(tmp_path):
    """Test that wells are returned in the order they first appear in CSV."""
    content = """X_UTME,Y_UTMN,Z_TVDSS,WELLNAME
100.0,200.0,1000.0,THIRD
100.0,200.0,1001.0,FIRST
100.0,200.0,1002.0,SECOND
100.0,200.0,1003.0,FIRST
100.0,200.0,1004.0,THIRD
100.0,200.0,1005.0,SECOND
"""
    csv_file = tmp_path / "ordered_wells.csv"
    csv_file.write_text(content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

    # Wells should appear in order of first occurrence
    well_names = [w.name for w in wells.wells]
    assert well_names == ["THIRD", "FIRST", "SECOND"]


def test_wells_from_csv_single_well(tmp_path):
    """Test CSV with only one well."""
    content = """X_UTME,Y_UTMN,Z_TVDSS,WELLNAME,PHIT
100.0,200.0,1000.0,SINGLE,0.25
100.0,200.0,1001.0,SINGLE,0.30
"""
    csv_file = tmp_path / "single_well.csv"
    csv_file.write_text(content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

    assert len(wells.wells) == 1
    assert wells.wells[0].name == "SINGLE"
    assert len(wells.wells[0].get_dataframe()) == 2


def test_wells_from_csv_missing_wellname_column_error(tmp_path):
    """Test that missing WELLNAME column raises an error."""
    content = """X_UTME,Y_UTMN,Z_TVDSS,PHIT
100.0,200.0,1000.0,0.25
100.0,200.0,1001.0,0.30
"""
    csv_file = tmp_path / "no_wellname.csv"
    csv_file.write_text(content)

    with pytest.raises(ValueError, match="Missing required columns"):
        xtgeo.wells_from_stacked_file(csv_file, fformat="csv")


def test_wells_from_csv_missing_coordinate_columns_error(tmp_path):
    """Test that missing coordinate columns raise an error."""
    content = """X_UTME,Y_UTMN,WELLNAME,PHIT
100.0,200.0,WELL-A,0.25
"""
    csv_file = tmp_path / "no_z_coord.csv"
    csv_file.write_text(content)

    with pytest.raises(ValueError, match="Missing required columns"):
        xtgeo.wells_from_stacked_file(csv_file, fformat="csv")


def test_wells_from_csv_roundtrip(tmp_path, sample_csv_wells_content):
    """Test that wells can be read from CSV and written back."""
    csv_file = tmp_path / "multi_wells.csv"
    csv_file.write_text(sample_csv_wells_content)

    # Read wells
    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

    # Write each well to CSV
    for well in wells.wells:
        output_file = tmp_path / f"{well.name}.csv"
        well.to_file(output_file, fformat="csv")

        # Verify file was created
        assert output_file.exists()

        # Read it back
        well_reloaded = xtgeo.well_from_file(
            output_file, fformat="csv", wellname=well.name
        )
        assert well_reloaded.name == well.name
        assert len(well_reloaded.get_dataframe()) == len(well.get_dataframe())


def test_wells_from_csv_with_nan_values(tmp_path):
    """Test handling of NaN values in CSV."""
    content = """X_UTME,Y_UTMN,Z_TVDSS,WELLNAME,PHIT
100.0,200.0,1000.0,WELL-A,0.25
100.0,200.0,1001.0,WELL-A,
100.0,200.0,1002.0,WELL-A,0.30
"""
    csv_file = tmp_path / "wells_with_nan.csv"
    csv_file.write_text(content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")
    well_a = wells.get_well("WELL-A")

    # Verify NaN is preserved
    df = well_a.get_dataframe()
    assert pd.isna(df["PHIT"].iloc[1])
    assert df["PHIT"].iloc[0] == pytest.approx(0.25)
    assert df["PHIT"].iloc[2] == pytest.approx(0.30)


def test_wells_from_csv_many_wells(tmp_path):
    """Test CSV with many wells to verify performance and correctness."""
    # Generate CSV with 10 wells, each with 5 records
    rows = []
    for well_num in range(10):
        well_name = f"WELL-{well_num}"
        for record in range(5):
            z = 1000.0 + record
            rows.append(f"100.0,200.0,{z},{well_name},0.25")

    content = "X_UTME,Y_UTMN,Z_TVDSS,WELLNAME,PHIT\n" + "\n".join(rows)

    csv_file = tmp_path / "many_wells.csv"
    csv_file.write_text(content)

    wells = xtgeo.wells_from_stacked_file(csv_file, fformat="csv")

    # Verify we got all 10 wells
    assert len(wells.wells) == 10

    # Verify each well has 5 records
    for well in wells.wells:
        assert len(well.get_dataframe()) == 5


def test_blockedwells_from_csv_different_lengths(tmp_path):
    """Test blocked wells with different number of records per well."""
    content = """X_UTME,Y_UTMN,Z_TVDSS,I_INDEX,J_INDEX,K_INDEX,WELLNAME
100.0,200.0,1000.0,10,20,0,SHORT
100.0,200.0,1001.0,10,20,1,SHORT
200.0,300.0,2000.0,15,25,0,LONG
200.0,300.0,2001.0,15,25,1,LONG
200.0,300.0,2002.0,15,25,2,LONG
200.0,300.0,2003.0,15,25,3,LONG
200.0,300.0,2004.0,15,25,4,LONG
"""
    csv_file = tmp_path / "different_lengths.csv"
    csv_file.write_text(content)

    bwells = xtgeo.blockedwells_from_stacked_file(csv_file, fformat="csv")

    short = bwells.get_blocked_well("SHORT")
    long = bwells.get_blocked_well("LONG")

    assert len(short.get_dataframe()) == 2
    assert len(long.get_dataframe()) == 5
