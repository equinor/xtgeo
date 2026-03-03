"""Tests for Wells I/O operations (multi-well file import/export)."""

import logging
from os.path import join

import pytest

import xtgeo
from xtgeo.well import Wells

logger = logging.getLogger(__name__)


@pytest.fixture(name="testwells")
def fixture_testwells(testdata_path):
    w_names = [
        "WELL29",
        "WELL14",
        "WELL30",
        "WELL27",
        "WELL23",
        "WELL32",
        "WELL22",
        "WELL35",
        "WELLX",
    ]
    well_files = [
        join(testdata_path, "wells", "battle", "1", wn + ".rmswell") for wn in w_names
    ]
    return xtgeo.wells_from_files(well_files, fformat="rms_ascii")


def test_wells_to_stacked_file_csv(tmp_path, testwells):
    """Export multiple wells to CSV format."""
    outfile = tmp_path / "all_wells.csv"
    result = testwells.to_stacked_file(outfile, fformat="csv")
    assert result.exists()

    # Verify by reading back
    import pandas as pd

    df = pd.read_csv(outfile)
    assert "WELLNAME" in df.columns
    assert set(df["WELLNAME"].unique()) == set(testwells.names)


def test_wells_to_stacked_file_rms_ascii(tmp_path, testwells):
    """Export multiple wells to RMS ASCII format."""
    outfile = tmp_path / "all_wells.rmswell"
    result = testwells.to_stacked_file(outfile, fformat="rms_ascii")
    assert result.exists()

    # Read back - should be readable but may need custom parsing
    # for multiple wells in one file
    assert outfile.stat().st_size > 0


def test_wells_to_stacked_file_hdf_raises(tmp_path, testwells):
    """Test that HDF5 format raises appropriate error."""
    outfile = tmp_path / "all_wells.hdf5"
    with pytest.raises(NotImplementedError, match="HDF5 format is not supported"):
        testwells.to_stacked_file(outfile, fformat="hdf5")


def test_wells_to_stacked_file_empty_raises():
    """Test that empty Wells raises error."""
    with pytest.raises(ValueError, match="No wells to export"):
        Wells().to_stacked_file("dummy.csv", fformat="csv")


def test_wells_from_stacked_file_csv(tmp_path, testwells):
    """Test importing multiple wells from CSV format."""
    outfile = tmp_path / "all_wells.csv"
    testwells.to_stacked_file(outfile, fformat="csv")

    # Read back
    wells_read = xtgeo.wells_from_stacked_file(outfile, fformat="csv")
    assert len(wells_read.wells) == len(testwells.wells)
    assert set(wells_read.names) == set(testwells.names)


def test_wells_from_stacked_file_rms_ascii(tmp_path, testwells):
    """Test importing multiple wells from RMS ASCII format."""
    outfile = tmp_path / "all_wells.rmswell"
    testwells.to_stacked_file(outfile, fformat="rms_ascii")

    # Read back
    wells_read = xtgeo.wells_from_stacked_file(outfile, fformat="rms_ascii")
    assert len(wells_read.wells) == len(testwells.wells)
    assert set(wells_read.names) == set(testwells.names)


def test_wells_from_stacked_file_csv_no_wellname_raises(tmp_path):
    """Test that CSV without WELLNAME column raises error."""
    # Create a CSV without WELLNAME column
    import pandas as pd

    df = pd.DataFrame({"X_UTME": [0, 1], "Y_UTMN": [0, 1], "Z_TVDSS": [0, 1]})
    csvfile = tmp_path / "no_wellname.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="WELLNAME"):
        xtgeo.wells_from_stacked_file(csvfile, fformat="csv")


def test_wells_from_stacked_file_csv_no_depth_raises(tmp_path):
    """Ensure missing depth column gives a helpful error."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "WELLNAME": ["W1", "W1"],
            "X_UTME": [0, 1],
            "Y_UTMN": [0, 1],
            "PORO": [0.2, 0.3],
        }
    )
    csvfile = tmp_path / "no_depth.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="depth column"):
        xtgeo.wells_from_stacked_file(csvfile, fformat="csv")


def test_wells_from_stacked_file_csv_no_coordinates_raises(tmp_path):
    """CSV import should fail fast when horizontal coordinates are absent."""
    import pandas as pd

    df = pd.DataFrame({"WELLNAME": ["W1", "W1"], "Z_TVDSS": [0, 1], "GR": [50, 60]})
    csvfile = tmp_path / "no_xy.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="coordinate"):
        xtgeo.wells_from_stacked_file(csvfile, fformat="csv")


def test_wells_from_stacked_file_auto_detect_csv(tmp_path, testwells):
    """Test that CSV format is auto-detected from .csv extension."""
    outfile = tmp_path / "wells.csv"
    testwells.to_stacked_file(outfile, fformat="csv")

    wells_read = xtgeo.wells_from_stacked_file(outfile)  # fformat=None
    assert len(wells_read.wells) == len(testwells.wells)
    assert set(wells_read.names) == set(testwells.names)


def test_wells_from_stacked_file_auto_detect_rms_ascii(tmp_path, testwells):
    """Test that RMS ASCII format is auto-detected from .rmswell extension."""
    outfile = tmp_path / "wells.rmswell"
    testwells.to_stacked_file(outfile, fformat="rms_ascii")

    wells_read = xtgeo.wells_from_stacked_file(outfile)  # fformat=None
    assert len(wells_read.wells) == len(testwells.wells)
    assert set(wells_read.names) == set(testwells.names)


def test_wells_to_stacked_file_invalid_format_raises(tmp_path, testwells):
    """Test that invalid format string raises clear error."""
    from xtgeo.common.exceptions import InvalidFileFormatError

    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        testwells.to_stacked_file(tmp_path / "output.xyz", fformat="invalid_format")


def test_wells_from_stacked_file_invalid_format_raises(tmp_path):
    """Test that invalid format string raises clear error on import."""
    import pandas as pd

    from xtgeo.common.exceptions import InvalidFileFormatError

    df = pd.DataFrame(
        {
            "WELLNAME": ["W1", "W1"],
            "X_UTME": [0, 1],
            "Y_UTMN": [0, 1],
            "Z_TVDSS": [0, 1],
        }
    )
    csvfile = tmp_path / "wells.dat"
    df.to_csv(csvfile, index=False)

    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        xtgeo.wells_from_stacked_file(csvfile, fformat="invalid_format")
