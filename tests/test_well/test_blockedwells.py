import logging
from os.path import join

import pytest

import xtgeo
from xtgeo.well import BlockedWells

logger = logging.getLogger(__name__)


@pytest.fixture(name="testblockedwells")
def fixture_testblockedwells(testdata_path):
    well_files = [join(testdata_path, "wells", "reek", "1", "OP_1.bw")]
    return xtgeo.blockedwells_from_files(well_files)


def test_blockedwells_empty_is_none():
    assert BlockedWells().wells is None


def test_blockedwells_non_well_setter_errors():
    with pytest.raises(ValueError, match="not valid Well"):
        BlockedWells().wells = [None]


def test_blockedwells_copy_same_names(testblockedwells):
    assert testblockedwells.copy().names == testblockedwells.names


def test_blockedwells_setter_same_names(testblockedwells):
    blockedwells2 = BlockedWells()
    blockedwells2.wells = testblockedwells.wells
    assert blockedwells2.names == testblockedwells.names


def test_well_names(testblockedwells):
    for well in testblockedwells.wells:
        assert well.name in testblockedwells.names
        assert testblockedwells.get_blocked_well(well.name) is well


def test_get_dataframe_allblockedwells(testblockedwells, snapshot, helpers):
    """Get a single dataframe for all blockedwells"""
    snapshot.assert_match(
        helpers.df2csv(testblockedwells.get_dataframe(filled=True).head(50).round()),
        "blockedwells.csv",
    )


def test_quickplot_blockedwells(tmp_path, testblockedwells, generate_plot):
    """Import blockedwells from file to BlockedWells and quick plot."""
    if not generate_plot:
        pytest.skip()
    testblockedwells.quickplot(filename=tmp_path / "quickblockedwells.png")


def test_wellintersections(snapshot, testblockedwells, helpers):
    """Find well crossing"""
    snapshot.assert_match(
        helpers.df2csv(testblockedwells.wellintersections().head(50).round()),
        "blockedwells_intersections.csv",
    )


def test_wellintersections_tvdrange_nowfilter(snapshot, testblockedwells, helpers):
    """Find well crossing using coarser sampling to Fence"""
    testblockedwells.limit_tvd(1300, 1400)
    testblockedwells.downsample(interval=6)

    snapshot.assert_match(
        helpers.df2csv(testblockedwells.wellintersections().head(50).round()),
        "tvdrange_blockedwells_intersections.csv",
    )


def test_wellintersections_tvdrange_wfilter(snapshot, testblockedwells, helpers):
    """Find well crossing using coarser sampling to Fence, with
    wfilter settings.
    """

    wfilter = {
        "parallel": {"xtol": 4.0, "ytol": 4.0, "ztol": 2.0, "itol": 10, "atol": 5.0}
    }

    testblockedwells.limit_tvd(1300, 1400)
    testblockedwells.downsample(interval=6)

    snapshot.assert_match(
        helpers.df2csv(
            testblockedwells.wellintersections(wfilter=wfilter).head(50).round()
        ),
        "filtered_blockedwells_intersections.csv",
    )


def test_blockedwells_from_stacked_file_unsupported_format():
    """Test that unsupported format raises error."""
    from xtgeo.common.exceptions import InvalidFileFormatError

    with pytest.raises(InvalidFileFormatError, match="unknown or unsupported"):
        xtgeo.blockedwells_from_stacked_file("dummy.txt", fformat="unsupported")


def test_blockedwells_from_stacked_file_valid_but_wrong_format(tmp_path):
    """Valid format that's unsupported for stacked files raises ValueError."""
    # Create a dummy file
    dummy_file = tmp_path / "dummy.rmswell"
    dummy_file.write_text("dummy content")

    # Try to use a valid file format that's not supported for stacked blocked wells
    # (e.g., single well format "rms_ascii" instead of "rms_ascii_stacked")
    with pytest.raises(ValueError, match="Unsupported format"):
        xtgeo.blockedwells_from_stacked_file(dummy_file, fformat="rms_ascii")


def test_blockedwells_to_stacked_file_rms_ascii(tmp_path, testdata_path):
    """Test exporting multiple blocked wells to a stacked RMS ASCII file."""
    # Create a BlockedWells collection with multiple wells
    well_files = [
        join(testdata_path, "wells", "reek", "1", "OP_1.bw"),
    ]
    bwells = xtgeo.blockedwells_from_files(well_files)

    outfile = tmp_path / "stacked_blockedwells.rmswell"

    result_file = bwells.to_stacked_file(outfile, fformat="rms_ascii_stacked")

    assert result_file.exists()

    # Re-import and verify
    reimported = xtgeo.blockedwells_from_stacked_file(result_file)
    assert len(reimported.wells) == len(bwells.wells)
    assert reimported.names == bwells.names


def test_blockedwells_copy_preserves_type(testblockedwells):
    """Test that copy() returns a BlockedWells instance with copied wells."""
    copied = testblockedwells.copy()

    assert isinstance(copied, BlockedWells)
    assert len(copied.wells) == len(testblockedwells.wells)

    # Verify independence
    if copied.wells:
        original_well = testblockedwells.wells[0]
        copied_well = copied.wells[0]

        # Modify copied well and verify original is unchanged
        original_name = original_well.name
        copied_well.name = "MODIFIED_NAME"
        assert original_well.name == original_name
        assert copied_well.name == "MODIFIED_NAME"


def test_get_blocked_well_returns_correct_well(testblockedwells):
    """Test that get_blocked_well returns the correct well by name."""
    if testblockedwells.wells:
        first_well = testblockedwells.wells[0]
        retrieved_well = testblockedwells.get_blocked_well(first_well.name)

        assert retrieved_well is first_well
        assert retrieved_well.name == first_well.name


def test_get_blocked_well_returns_none_for_nonexistent(testblockedwells):
    """Test that get_blocked_well returns None for non-existent well name."""
    result = testblockedwells.get_blocked_well("NONEXISTENT_WELL_NAME")
    assert result is None


def test_blockedwells_from_stacked_file_csv(tmp_path):
    """Test importing blocked wells from CSV format."""
    import pandas as pd

    # Create a CSV file with multiple blocked wells
    csv_file = tmp_path / "blocked_wells.csv"
    df = pd.DataFrame(
        {
            "WELLNAME": ["WELL1", "WELL1", "WELL2", "WELL2"],
            "X_UTME": [1000.0, 1001.0, 2000.0, 2001.0],
            "Y_UTMN": [5000.0, 5001.0, 6000.0, 6001.0],
            "Z_TVDSS": [1500.0, 1501.0, 1600.0, 1601.0],
            "I_INDEX": [10, 10, 20, 20],
            "J_INDEX": [15, 15, 25, 25],
            "K_INDEX": [5, 6, 7, 8],
            "PORO": [0.25, 0.26, 0.30, 0.31],
        }
    )
    df.to_csv(csv_file, index=False)

    # Import from CSV
    bwells = xtgeo.blockedwells_from_stacked_file(csv_file, fformat="csv")

    assert len(bwells.wells) == 2
    assert "WELL1" in bwells.names
    assert "WELL2" in bwells.names


def test_blockedwells_to_stacked_file_csv(tmp_path, testblockedwells):
    """Test exporting blocked wells to CSV format."""
    outfile = tmp_path / "blocked_wells.csv"

    result_file = testblockedwells.to_stacked_file(outfile, fformat="csv")

    assert result_file.exists()

    # Re-import and verify
    reimported = xtgeo.blockedwells_from_stacked_file(result_file, fformat="csv")
    assert len(reimported.wells) == len(testblockedwells.wells)
    # Names should be preserved
    for name in testblockedwells.names:
        assert name in reimported.names
