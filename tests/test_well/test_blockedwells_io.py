"""Tests for BlockedWells I/O operations (multi-well file import/export)."""

import logging
import pathlib

import pytest

import xtgeo
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.well import BlockedWells

logger = logging.getLogger(__name__)

# Stacked blocked-well file (55_33-A-4/5/6) lives in xtgeo-testdata.
# Reading it exercises blockedwells_from_stacked_file against a real,
# independently-created file rather than one produced by the stacked writer.
_STACKED_BW_FILE = pathlib.Path("wells/etc/stacked_blockedwells.rmswell")
_STACKED_BW_WELLS = {"55_33-A-4", "55_33-A-5", "55_33-A-6"}


@pytest.fixture(name="testblockedwells")
def fixture_testblockedwells(testdata_path):
    """Three drogon blocked wells loaded directly from a pre-existing stacked file."""
    return xtgeo.blockedwells_from_stacked_file(
        testdata_path / _STACKED_BW_FILE, fformat="rms_ascii"
    )


def test_blockedwells_from_stacked_file_reads_actual_file(testdata_path):
    """Read the pre-existing stacked file directly — no writer involvement."""
    bwells = xtgeo.blockedwells_from_stacked_file(
        testdata_path / _STACKED_BW_FILE, fformat="rms_ascii"
    )
    assert len(bwells.wells) == 3
    assert set(bwells.names) == _STACKED_BW_WELLS

    expected_logs = ["Facies", "PHIT", "KLOGH", "VSH", "I_INDEX", "J_INDEX", "K_INDEX"]
    expected_nrows = {"55_33-A-4": 31, "55_33-A-5": 20, "55_33-A-6": 20}
    for bw in bwells.wells:
        assert bw.lognames == expected_logs
        assert bw.nrow == expected_nrows[bw.name]


def test_blockedwells_to_stacked_file_csv(tmp_path, testblockedwells):
    """Export multiple blocked wells to CSV format."""
    outfile = tmp_path / "all_bwells.csv"
    result = testblockedwells.to_stacked_file(outfile, fformat="csv")
    assert result.exists()

    # Verify by reading back
    import pandas as pd

    df = pd.read_csv(outfile)
    assert "WELLNAME" in df.columns
    assert set(df["WELLNAME"].unique()) == set(testblockedwells.names)


def test_blockedwells_to_stacked_file_rms_ascii(tmp_path, testblockedwells):
    """Export multiple blocked wells to RMS ASCII format."""
    outfile = tmp_path / "all_bwells.rmswell"
    testblockedwells.to_stacked_file(outfile, fformat="rms_ascii")

    bwells_read = xtgeo.blockedwells_from_stacked_file(outfile, fformat="rms_ascii")
    assert set(bwells_read.names) == set(testblockedwells.names)
    for bw in bwells_read.wells:
        orig = next(o for o in testblockedwells.wells if o.name == bw.name)
        assert bw.nrow == orig.nrow
        assert bw.lognames == orig.lognames


def test_blockedwells_to_stacked_file_hdf5_raises(tmp_path, testblockedwells):
    """Test that HDF5 format raises appropriate error."""
    outfile = tmp_path / "all_bwells.h5"
    with pytest.raises(
        InvalidFileFormatError,
        match="hdf5 is not supported for stacked BlockedWells export",
    ):
        testblockedwells.to_stacked_file(outfile, fformat="hdf5")


def test_blockedwells_to_stacked_file_empty_raises():
    """Test that empty BlockedWells raises error."""
    with pytest.raises(ValueError, match="No blocked wells to export"):
        BlockedWells().to_stacked_file("dummy.csv", fformat="csv")


def test_blockedwells_from_stacked_file_csv(tmp_path, testblockedwells):
    """Test importing multiple blocked wells from CSV format."""
    outfile = tmp_path / "all_bwells.csv"
    testblockedwells.to_stacked_file(outfile, fformat="csv")

    # Read back
    bwells_read = xtgeo.blockedwells_from_stacked_file(outfile, fformat="csv")
    assert len(bwells_read.wells) == len(testblockedwells.wells)
    assert set(bwells_read.names) == set(testblockedwells.names)


def test_blockedwells_from_stacked_file_rms_ascii(tmp_path, testblockedwells):
    """Test importing multiple blocked wells from RMS ASCII format."""
    outfile = tmp_path / "all_bwells.rmswell"
    testblockedwells.to_stacked_file(outfile, fformat="rms_ascii")

    # Read back
    bwells_read = xtgeo.blockedwells_from_stacked_file(outfile, fformat="rms_ascii")
    assert len(bwells_read.wells) == len(testblockedwells.wells)
    assert set(bwells_read.names) == set(testblockedwells.names)


def test_blockedwells_from_stacked_file_hdf5_raises(tmp_path):
    """Test that HDF5 format raises appropriate error."""
    outfile = tmp_path / "dummy.h5"
    outfile.touch()  # Create empty file
    with pytest.raises(
        InvalidFileFormatError,
        match="hdf5 is not supported for blockedwells_from_stacked_file",
    ):
        xtgeo.blockedwells_from_stacked_file(outfile, fformat="hdf5")


def test_blockedwells_from_stacked_file_csv_no_wellname_raises(tmp_path):
    """Test that CSV without WELLNAME column raises error."""
    # Create a CSV without WELLNAME column
    import pandas as pd

    df = pd.DataFrame({"X_UTME": [0, 1], "Y_UTMN": [0, 1], "Z_TVDSS": [0, 1]})
    csvfile = tmp_path / "no_wellname.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="WELLNAME"):
        xtgeo.blockedwells_from_stacked_file(csvfile, fformat="csv")


def test_blockedwells_from_stacked_file_csv_no_coordinates_raises(tmp_path):
    """CSV import should fail fast when horizontal coordinates are absent."""
    import pandas as pd

    df = pd.DataFrame({"WELLNAME": ["W1", "W1"], "Z_TVDSS": [0, 1], "FACIES": [1, 2]})
    csvfile = tmp_path / "no_xy.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="coordinate"):
        xtgeo.blockedwells_from_stacked_file(csvfile, fformat="csv")


def test_blockedwells_from_stacked_file_csv_no_depth_raises(tmp_path):
    """CSV import should fail fast when depth column is absent."""
    import pandas as pd

    df = pd.DataFrame(
        {"WELLNAME": ["W1", "W1"], "X_UTME": [0, 1], "Y_UTMN": [0, 1], "FACIES": [1, 2]}
    )
    csvfile = tmp_path / "no_depth.csv"
    df.to_csv(csvfile, index=False)

    with pytest.raises(ValueError, match="depth column"):
        xtgeo.blockedwells_from_stacked_file(csvfile, fformat="csv")


def test_blockedwells_from_stacked_file_auto_detect_csv(tmp_path, testblockedwells):
    """Test that CSV format is auto-detected from .csv extension."""
    outfile = tmp_path / "bwells.csv"
    testblockedwells.to_stacked_file(outfile, fformat="csv")

    bwells_read = xtgeo.blockedwells_from_stacked_file(outfile)  # fformat=None
    assert len(bwells_read.wells) == len(testblockedwells.wells)
    assert set(bwells_read.names) == set(testblockedwells.names)


def test_blockedwells_from_stacked_file_auto_detect_rms_ascii(
    tmp_path, testblockedwells
):
    """Test that RMS ASCII format is auto-detected from .rmswell extension."""
    outfile = tmp_path / "bwells.rmswell"
    testblockedwells.to_stacked_file(outfile, fformat="rms_ascii")

    bwells_read = xtgeo.blockedwells_from_stacked_file(outfile)  # fformat=None
    assert len(bwells_read.wells) == len(testblockedwells.wells)
    assert set(bwells_read.names) == set(testblockedwells.names)


def test_blockedwells_to_stacked_file_invalid_format_raises(tmp_path, testblockedwells):
    """Test that invalid format string raises clear error."""
    from xtgeo.common.exceptions import InvalidFileFormatError

    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        testblockedwells.to_stacked_file(
            tmp_path / "output.xyz", fformat="invalid_format"
        )


def test_blockedwells_from_stacked_file_invalid_format_raises(tmp_path):
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
    csvfile = tmp_path / "bwells.dat"
    df.to_csv(csvfile, index=False)

    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        xtgeo.blockedwells_from_stacked_file(csvfile, fformat="invalid_format")
