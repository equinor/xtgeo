"""Tests for RMS ASCII format I/O for WellData and BlockedWellData."""

from __future__ import annotations

import io

import numpy as np
import pytest

from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellFileFormat, WellLog

# ============================================================================
# Legacy stream tests (kept for specific stream-related edge cases)
# ============================================================================


def test_welldata_write_discrete_to_stream():
    """Test writing WellData with discrete log to StringIO stream."""
    import io

    from xtgeo.io._welldata._well_io import WellData, WellLog

    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    well = WellData(
        name="DiscreteStreamWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(facies_log,),
    )

    stream = io.StringIO()
    well.to_rms_ascii(filepath=stream)

    content = stream.getvalue()
    assert "DiscreteStreamWell" in content
    assert "FACIES DISC 1 SAND 2 SHALE" in content


def test_welldata_read_from_bytesio_stream():
    """Test reading WellData from BytesIO stream."""
    import io

    from xtgeo.io._welldata._well_io import WellData

    rms_content = b"""1.0
Test well from bytes
TestWell 100.0 200.0 0.0
1
PORO CONT lin
100.0 200.0 1000.0 0.25
101.0 201.0 1001.0 0.30
"""
    stream = io.BytesIO(rms_content)

    well = WellData.from_rms_ascii(filepath=stream)

    assert well.name == "TestWell"
    assert well.n_records == 2
    assert len(well.logs) == 1


def test_welldata_read_zero_logs():
    """Test reading WellData with zero logs (nlogs == 0)."""
    import io

    from xtgeo.io._welldata._well_io import WellData

    rms_content = """1.0
Well with no logs
NoLogsWell 150.0 250.0 10.0
0
150.0 250.0 1500.0
151.0 251.0 1501.0
"""
    stream = io.StringIO(rms_content)
    well = WellData.from_rms_ascii(filepath=stream)

    assert well.name == "NoLogsWell"
    assert well.xpos == 150.0
    assert well.ypos == 250.0
    assert well.zpos == 10.0
    assert len(well.logs) == 0
    assert well.n_records == 2


def test_welldata_read_well_name_with_spaces():
    """Test reading WellData with well name containing spaces."""
    import io

    from xtgeo.io._welldata._well_io import WellData

    # Test with RKB
    rms_content_with_rkb = """1.0
Well with space in name
31/4-2 A   4444030. 9937883.0 32.0
1
GR CONT lin
4444030.0 9937883.0 1000.0 75.5
4444031.0 9937884.0 1001.0 80.2
"""
    stream = io.StringIO(rms_content_with_rkb)
    well = WellData.from_rms_ascii(filepath=stream)

    assert well.name == "31/4-2 A"
    assert well.xpos == 4444030.0
    assert well.ypos == 9937883.0
    assert well.zpos == 32.0
    assert well.n_records == 2

    # Test without RKB
    rms_content_no_rkb = """1.0
Well with space in name no RKB
31/4-2 A   4444030. 9937883.0
1
GR CONT lin
4444030.0 9937883.0 1000.0 75.5
"""
    stream = io.StringIO(rms_content_no_rkb)
    well = WellData.from_rms_ascii(filepath=stream)

    assert well.name == "31/4-2 A"
    assert well.xpos == 4444030.0
    assert well.ypos == 9937883.0
    assert well.zpos == 0.0
    assert well.n_records == 1


# ============================================================================
# WellData.to_file() and WellData.from_file() with RMS_ASCII format
# ============================================================================


def test_welldata_to_file_from_file_basic(tmp_path):
    """Test basic WellData write and read using to_file/from_file."""
    # Create well data
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))
    poro_log = WellLog(name="PORO", values=np.array([0.15, 0.20, 0.25]))

    well = WellData(
        name="WELL-A",
        xpos=1000.0,
        ypos=2000.0,
        zpos=100.0,
        survey_x=np.array([1000.0, 1001.0, 1002.0]),
        survey_y=np.array([2000.0, 2001.0, 2002.0]),
        survey_z=np.array([1500.0, 1600.0, 1700.0]),
        logs=(gr_log, poro_log),
    )

    # Write to file
    filepath = tmp_path / "well_a.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    # Read back
    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    # Verify
    assert well_read.name == "WELL-A"
    assert well_read.xpos == 1000.0
    assert well_read.ypos == 2000.0
    assert well_read.zpos == 100.0
    assert well_read.n_records == 3
    assert len(well_read.logs) == 2
    assert well_read.log_names == ("GR", "PORO")
    np.testing.assert_array_almost_equal(well_read.survey_x, well.survey_x)
    np.testing.assert_array_almost_equal(well_read.survey_z, well.survey_z)


def test_welldata_to_file_from_file_with_discrete_log(tmp_path):
    """Test WellData with discrete log using to_file/from_file."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 3.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE", 3: "LIMESTONE"},
    )

    well = WellData(
        name="WELL-B",
        xpos=5000.0,
        ypos=6000.0,
        zpos=50.0,
        survey_x=np.array([5000.0, 5000.0, 5000.0, 5000.0]),
        survey_y=np.array([6000.0, 6000.0, 6000.0, 6000.0]),
        survey_z=np.array([2000.0, 2010.0, 2020.0, 2030.0]),
        logs=(facies_log,),
    )

    filepath = tmp_path / "well_b.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    assert well_read.name == "WELL-B"
    assert well_read.n_records == 4
    assert len(well_read.logs) == 1

    facies_read = well_read.get_log("FACIES")
    assert facies_read is not None
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE", 3: "LIMESTONE"}
    np.testing.assert_array_equal(facies_read.values, facies_log.values)


def test_welldata_to_file_from_file_no_rkb(tmp_path):
    """Test WellData without RKB (zpos=0) using to_file/from_file."""
    log = WellLog(name="DEPTH", values=np.array([1000.0, 1100.0]))

    well = WellData(
        name="WELL-NO-RKB",
        xpos=123.45,
        ypos=678.90,
        zpos=0.0,  # No RKB
        survey_x=np.array([123.45, 123.46]),
        survey_y=np.array([678.90, 678.91]),
        survey_z=np.array([1000.0, 1100.0]),
        logs=(log,),
    )

    filepath = tmp_path / "well_no_rkb.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    assert well_read.name == "WELL-NO-RKB"
    assert well_read.zpos == 0.0
    assert well_read.n_records == 2


def test_welldata_to_file_from_file_no_logs(tmp_path):
    """Test WellData with no logs using to_file/from_file."""
    well = WellData(
        name="EMPTY-LOGS",
        xpos=100.0,
        ypos=200.0,
        zpos=10.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1100.0, 1200.0]),
        logs=(),
    )

    filepath = tmp_path / "well_no_logs.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    assert well_read.name == "EMPTY-LOGS"
    assert len(well_read.logs) == 0
    assert well_read.n_records == 3


def test_welldata_to_file_from_file_precision(tmp_path):
    """Test WellData precision parameter in to_file."""
    log = WellLog(name="VALUE", values=np.array([1.23456789]))

    well = WellData(
        name="PRECISION-TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0]),
        survey_y=np.array([200.0]),
        survey_z=np.array([1000.0]),
        logs=(log,),
    )

    # Write with precision=2
    filepath = tmp_path / "well_precision.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII, precision=2)

    content = filepath.read_text()
    # Should have 2 decimal places (1.23)
    assert "1.23" in content
    # Should not have more precision
    assert "1.2345" not in content


def test_welldata_to_file_from_file_with_nan_values(tmp_path):
    """Test WellData with NaN values using to_file/from_file."""
    log = WellLog(name="INCOMPLETE", values=np.array([10.0, np.nan, 30.0]))

    well = WellData(
        name="NAN-TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1100.0, 1200.0]),
        logs=(log,),
    )

    filepath = tmp_path / "well_nan.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    log_read = well_read.get_log("INCOMPLETE")
    assert log_read is not None
    # Check values with NaN handling
    assert np.array_equal(log_read.values, log.values, equal_nan=True)


def test_welldata_to_file_from_file_multiple_logs(tmp_path):
    """Test WellData with multiple logs of different types."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))
    poro_log = WellLog(name="PORO", values=np.array([0.15, 0.20]))
    perm_log = WellLog(name="PERM", values=np.array([100.0, 200.0]))
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    well = WellData(
        name="MULTI-LOG",
        xpos=1000.0,
        ypos=2000.0,
        zpos=25.0,
        survey_x=np.array([1000.0, 1001.0]),
        survey_y=np.array([2000.0, 2001.0]),
        survey_z=np.array([1500.0, 1600.0]),
        logs=(gr_log, poro_log, perm_log, facies_log),
    )

    filepath = tmp_path / "well_multi_log.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    assert well_read.n_records == 2
    assert len(well_read.logs) == 4
    assert well_read.log_names == ("GR", "PORO", "PERM", "FACIES")

    # Check discrete log
    facies_read = well_read.get_log("FACIES")
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE"}


def test_welldata_metadata_roundtrip(tmp_path):
    """Test that continuous log metadata (unit, scale) is preserved in roundtrip."""
    # Create a log with specific metadata tuple (TYPE, UNIT, SCALE, ...)
    # Note: The first element 'CONT' is usually implied by is_discrete=False,
    # but the parser stores the whole tuple from the file.
    # When creating manually, we simulate what the parser produces.
    meta_tuple = ("CONT", "mMD", "log")

    perm_log = WellLog(
        name="PERM",
        values=np.array([100.0, 200.0]),
        is_discrete=False,
        code_names=meta_tuple,
    )

    well = WellData(
        name="META-TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 100.0]),
        survey_y=np.array([200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(perm_log,),
    )

    filepath = tmp_path / "well_meta.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    # Verify the file content has the metadata
    content = filepath.read_text()
    assert "PERM CONT mMD log" in content

    # Read back and verify
    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)
    perm_read = well_read.get_log("PERM")

    # The parser reads the whole line after name, so it should match
    assert perm_read.code_names == meta_tuple


def test_welldata_discrete_sorted_output(tmp_path):
    """Test that discrete log codes are written in sorted order."""
    # Create code_names with unsorted keys
    codes = {2: "SHALE", 1: "SAND", 3: "LIME"}

    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 3.0]),
        is_discrete=True,
        code_names=codes,
    )

    well = WellData(
        name="SORT-TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 100.0, 100.0]),
        survey_y=np.array([200.0, 200.0, 200.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(facies_log,),
    )

    filepath = tmp_path / "well_sort.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    content = filepath.read_text()
    # Check for the specific sorted string sequence
    expected_header = "FACIES DISC 1 SAND 2 SHALE 3 LIME"
    assert expected_header in content


def test_welldata_to_file_from_file_stream(tmp_path):
    """Test WellData using to_file/from_file with stream objects."""
    log = WellLog(name="TEST", values=np.array([1.0, 2.0]))

    well = WellData(
        name="STREAM-TEST",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1100.0]),
        logs=(log,),
    )

    # Write to stream
    stream_out = io.StringIO()
    well.to_file(filepath=stream_out, fformat=WellFileFormat.RMS_ASCII)

    # Read from stream
    stream_in = io.StringIO(stream_out.getvalue())
    well_read = WellData.from_file(filepath=stream_in, fformat=WellFileFormat.RMS_ASCII)

    assert well_read.name == "STREAM-TEST"
    assert well_read.n_records == 2
    assert len(well_read.logs) == 1


# ============================================================================
# BlockedWellData.to_file() and BlockedWellData.from_file() with RMS_ASCII
# ============================================================================


def test_blockedwell_to_file_from_file_basic(tmp_path):
    """Test basic BlockedWellData write and read using to_file/from_file."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    blocked_well = BlockedWellData(
        name="BLOCKED-A",
        xpos=1000.0,
        ypos=2000.0,
        zpos=100.0,
        survey_x=np.array([1000.0, 1001.0, 1002.0]),
        survey_y=np.array([2000.0, 2001.0, 2002.0]),
        survey_z=np.array([1500.0, 1600.0, 1700.0]),
        i_index=np.array([10.0, 11.0, 12.0]),
        j_index=np.array([20.0, 21.0, 22.0]),
        k_index=np.array([1.0, 2.0, 3.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "blocked_a.txt"
    blocked_well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    blocked_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII
    )

    assert blocked_read.name == "BLOCKED-A"
    assert blocked_read.xpos == 1000.0
    assert blocked_read.ypos == 2000.0
    assert blocked_read.zpos == 100.0
    assert blocked_read.n_records == 3
    assert len(blocked_read.logs) == 1
    assert blocked_read.n_blocked_cells == 3
    np.testing.assert_array_equal(blocked_read.i_index, blocked_well.i_index)
    np.testing.assert_array_equal(blocked_read.j_index, blocked_well.j_index)
    np.testing.assert_array_equal(blocked_read.k_index, blocked_well.k_index)


def test_blockedwell_to_file_from_file_with_nan_indices(tmp_path):
    """Test BlockedWellData with NaN indices using to_file/from_file."""
    blocked_well = BlockedWellData(
        name="BLOCKED-NAN",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0, 103.0]),
        survey_y=np.array([200.0, 201.0, 202.0, 203.0]),
        survey_z=np.array([1000.0, 1100.0, 1200.0, 1300.0]),
        i_index=np.array([10.0, np.nan, 11.0, 12.0]),
        j_index=np.array([20.0, np.nan, 21.0, 22.0]),
        k_index=np.array([1.0, np.nan, 2.0, 3.0]),
        logs=(),
    )

    filepath = tmp_path / "blocked_nan.txt"
    blocked_well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    blocked_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII
    )

    assert blocked_read.n_blocked_cells == 3  # One point has NaN indices
    # Check indices with NaN handling
    assert np.array_equal(blocked_read.i_index, blocked_well.i_index, equal_nan=True)
    assert np.array_equal(blocked_read.j_index, blocked_well.j_index, equal_nan=True)
    assert np.array_equal(blocked_read.k_index, blocked_well.k_index, equal_nan=True)


def test_blockedwell_to_file_from_file_multiple_logs(tmp_path):
    """Test BlockedWellData with multiple logs using to_file/from_file."""
    poro_log = WellLog(name="PORO", values=np.array([0.15, 0.20]))
    perm_log = WellLog(name="PERM", values=np.array([100.0, 200.0]))
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    blocked_well = BlockedWellData(
        name="BLOCKED-MULTI",
        xpos=5000.0,
        ypos=6000.0,
        zpos=50.0,
        survey_x=np.array([5000.0, 5001.0]),
        survey_y=np.array([6000.0, 6001.0]),
        survey_z=np.array([2000.0, 2100.0]),
        i_index=np.array([15.0, 16.0]),
        j_index=np.array([25.0, 26.0]),
        k_index=np.array([5.0, 6.0]),
        logs=(poro_log, perm_log, facies_log),
    )

    filepath = tmp_path / "blocked_multi.txt"
    blocked_well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    blocked_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII
    )

    assert blocked_read.n_records == 2
    assert len(blocked_read.logs) == 3
    assert blocked_read.log_names == ("PORO", "PERM", "FACIES")

    facies_read = blocked_read.get_log("FACIES")
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE"}


def test_blockedwell_to_file_from_file_no_logs(tmp_path):
    """Test BlockedWellData with no logs using to_file/from_file."""
    blocked_well = BlockedWellData(
        name="BLOCKED-NOLOG",
        xpos=100.0,
        ypos=200.0,
        zpos=10.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1100.0]),
        i_index=np.array([10.0, 11.0]),
        j_index=np.array([20.0, 21.0]),
        k_index=np.array([1.0, 2.0]),
        logs=(),
    )

    filepath = tmp_path / "blocked_nolog.txt"
    blocked_well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    blocked_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII
    )

    assert blocked_read.name == "BLOCKED-NOLOG"
    assert len(blocked_read.logs) == 0
    assert blocked_read.n_records == 2
    assert blocked_read.n_blocked_cells == 2


def test_blockedwell_to_file_from_file_precision(tmp_path):
    """Test BlockedWellData precision parameter in to_file."""
    log = WellLog(name="VALUE", values=np.array([1.23456789]))

    blocked_well = BlockedWellData(
        name="BLOCKED-PRECISION",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0]),
        survey_y=np.array([200.0]),
        survey_z=np.array([1000.0]),
        i_index=np.array([10.0]),
        j_index=np.array([20.0]),
        k_index=np.array([1.0]),
        logs=(log,),
    )

    filepath = tmp_path / "blocked_precision.txt"
    blocked_well.to_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII, precision=3
    )

    content = filepath.read_text()
    # Should have 3 decimal places
    assert "1.235" in content or "1.234" in content


def test_blockedwell_to_file_from_file_stream(tmp_path):
    """Test BlockedWellData using to_file/from_file with stream objects."""
    blocked_well = BlockedWellData(
        name="BLOCKED-STREAM",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1100.0]),
        i_index=np.array([10.0, 11.0]),
        j_index=np.array([20.0, 21.0]),
        k_index=np.array([1.0, 2.0]),
        logs=(),
    )

    # Write to stream
    stream_out = io.StringIO()
    blocked_well.to_file(filepath=stream_out, fformat=WellFileFormat.RMS_ASCII)

    # Read from stream
    stream_in = io.StringIO(stream_out.getvalue())
    blocked_read = BlockedWellData.from_file(
        filepath=stream_in, fformat=WellFileFormat.RMS_ASCII
    )

    assert blocked_read.name == "BLOCKED-STREAM"
    assert blocked_read.n_records == 2
    np.testing.assert_array_equal(blocked_read.i_index, blocked_well.i_index)


def test_blockedwell_to_file_from_file_roundtrip_complex(tmp_path):
    """Test complex BlockedWellData roundtrip with all features."""
    # Create complex blocked well with mixed logs
    gr_log = WellLog(name="GR", values=np.array([45.5, 67.8, 89.2, 101.3]))
    poro_log = WellLog(name="PORO", values=np.array([0.18, 0.22, np.nan, 0.16]))
    sw_log = WellLog(name="SW", values=np.array([0.45, 0.38, 0.52, 0.41]))
    zone_log = WellLog(
        name="ZONE",
        values=np.array([1.0, 1.0, 2.0, 3.0]),
        is_discrete=True,
        code_names={1: "UPPER", 2: "MIDDLE", 3: "LOWER"},
    )

    blocked_well = BlockedWellData(
        name="31/4-A-15 H",
        xpos=456789.12,
        ypos=6543210.98,
        zpos=25.5,
        survey_x=np.array([456789.12, 456789.5, 456790.0, 456790.5]),
        survey_y=np.array([6543210.98, 6543211.2, 6543211.5, 6543211.8]),
        survey_z=np.array([1850.5, 1855.0, 1859.5, 1864.0]),
        i_index=np.array([125.0, 125.0, 126.0, np.nan]),
        j_index=np.array([234.0, 234.0, 235.0, np.nan]),
        k_index=np.array([15.0, 16.0, 17.0, np.nan]),
        logs=(gr_log, poro_log, sw_log, zone_log),
    )

    filepath = tmp_path / "complex_blocked.txt"
    blocked_well.to_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII, precision=4
    )

    blocked_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.RMS_ASCII
    )

    # Verify all attributes
    assert blocked_read.name == "31/4-A-15 H"
    assert blocked_read.xpos == pytest.approx(456789.12, abs=1e-2)
    assert blocked_read.ypos == pytest.approx(6543210.98, abs=1e-2)
    assert blocked_read.zpos == pytest.approx(25.5, abs=1e-2)
    assert blocked_read.n_records == 4
    assert blocked_read.n_blocked_cells == 3  # One has NaN indices
    assert len(blocked_read.logs) == 4

    # Verify indices with NaN handling
    assert np.array_equal(blocked_read.i_index, blocked_well.i_index, equal_nan=True)
    assert np.array_equal(blocked_read.j_index, blocked_well.j_index, equal_nan=True)
    assert np.array_equal(blocked_read.k_index, blocked_well.k_index, equal_nan=True)

    # Verify logs
    poro_read = blocked_read.get_log("PORO")
    # Use allclose for float comparison with NaN handling
    assert np.allclose(poro_read.values, poro_log.values, equal_nan=True, rtol=1e-5)

    zone_read = blocked_read.get_log("ZONE")
    assert zone_read.is_discrete
    assert zone_read.code_names == {1: "UPPER", 2: "MIDDLE", 3: "LOWER"}


def test_blockedwell_from_file_missing_indices(tmp_path):
    """BlockedWellData.from_file shall raise ValueError when indices are missing."""
    # Create a regular WellData file (no I_INDEX, J_INDEX, K_INDEX)
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    well = WellData(
        name="REGULAR-WELL",
        xpos=1000.0,
        ypos=2000.0,
        zpos=100.0,
        survey_x=np.array([1000.0, 1001.0]),
        survey_y=np.array([2000.0, 2001.0]),
        survey_z=np.array([1500.0, 1600.0]),
        logs=(gr_log,),
    )

    # Write WellData to file
    filepath = tmp_path / "regular_well.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    # Try to read as BlockedWellData - should raise ValueError
    with pytest.raises(
        ValueError,
        match="File does not contain I_INDEX, J_INDEX, and K_INDEX logs",
    ):
        BlockedWellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)


def test_blockedwell_from_file_partial_indices(tmp_path):
    """BlockedWellData.from_file -> ValueError when only some indices present."""
    # Create a file with only I_INDEX and J_INDEX (missing K_INDEX)
    i_log = WellLog(name="I_INDEX", values=np.array([10.0, 11.0]))
    j_log = WellLog(name="J_INDEX", values=np.array([20.0, 21.0]))
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    well = WellData(
        name="PARTIAL-INDICES",
        xpos=1000.0,
        ypos=2000.0,
        zpos=100.0,
        survey_x=np.array([1000.0, 1001.0]),
        survey_y=np.array([2000.0, 2001.0]),
        survey_z=np.array([1500.0, 1600.0]),
        logs=(i_log, j_log, gr_log),
    )

    filepath = tmp_path / "partial_indices.txt"
    well.to_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)

    # Try to read as BlockedWellData - should raise ValueError
    with pytest.raises(
        ValueError,
        match="File does not contain I_INDEX, J_INDEX, and K_INDEX logs",
    ):
        BlockedWellData.from_file(filepath=filepath, fformat=WellFileFormat.RMS_ASCII)
