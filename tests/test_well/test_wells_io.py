"""Tests for Wells I/O operations (multi-well file import/export)."""

import logging
import pathlib

import pytest

import xtgeo
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.well import Wells

logger = logging.getLogger(__name__)

# Stacked well file (WELL31 + WELL22 + WELL12) lives in xtgeo-testdata.
# Reading it exercises wells_from_stacked_file against a real,
# independently-created file rather than one produced by the stacked writer.
_STACKED_W_FILE = pathlib.Path("wells/etc/stacked_wells.rmswell")
_STACKED_W_WELLS = {"WELL31", "WELL22", "WELL12"}


@pytest.fixture(name="testwells")
def fixture_testwells(testdata_path):
    """Three battle wells loaded directly from a pre-existing stacked file."""
    return xtgeo.wells_from_stacked_file(
        testdata_path / _STACKED_W_FILE, fformat="rms_ascii"
    )


def test_wells_from_stacked_file_reads_actual_file(testdata_path):
    """Read the pre-existing stacked file directly — no writer involvement."""
    wells = xtgeo.wells_from_stacked_file(
        testdata_path / _STACKED_W_FILE, fformat="rms_ascii"
    )
    assert len(wells.wells) == 3
    assert set(wells.names) == _STACKED_W_WELLS

    expected_logs = ["GR", "ZONELOG"]
    expected_nrows = {"WELL31": 708, "WELL22": 805, "WELL12": 1029}
    for w in wells.wells:
        assert w.lognames == expected_logs
        assert w.nrow == expected_nrows[w.name]


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
    testwells.to_stacked_file(outfile, fformat="rms_ascii")

    wells_read = xtgeo.wells_from_stacked_file(outfile, fformat="rms_ascii")
    assert set(wells_read.names) == set(testwells.names)
    for w in wells_read.wells:
        orig = next(o for o in testwells.wells if o.name == w.name)
        assert w.nrow == orig.nrow
        assert w.lognames == orig.lognames


def test_wells_to_stacked_file_hdf_raises(tmp_path, testwells):
    """Test that HDF5 format raises appropriate error."""
    outfile = tmp_path / "all_wells.hdf5"
    with pytest.raises(
        InvalidFileFormatError, match="hdf5 is not supported for Wells.to_stacked_file"
    ):
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
