"""Test I/O of multiple wells from a single stacked file (RMS ASCII format)."""

import logging

import pytest

import xtgeo

logger = logging.getLogger(__name__)


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
