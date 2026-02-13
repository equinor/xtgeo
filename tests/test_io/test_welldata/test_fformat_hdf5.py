"""Tests for HDF5 format I/O for WellData and BlockedWellData."""

from __future__ import annotations

import numpy as np

from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellFileFormat, WellLog


def test_welldata_hdf5_basic_write_read(tmp_path):
    """Test basic WellData write and read using HDF5 format."""
    # Create well data
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))
    poro_log = WellLog(name="PORO", values=np.array([0.15, 0.20, 0.25]))

    well = WellData(
        name="TestWell_HDF5",
        xpos=460000.0,
        ypos=5930000.0,
        zpos=25.0,
        survey_x=np.array([460000.0, 460010.0, 460020.0]),
        survey_y=np.array([5930000.0, 5930010.0, 5930020.0]),
        survey_z=np.array([1000.0, 1010.0, 1020.0]),
        logs=(gr_log, poro_log),
    )

    # Write to HDF5
    filepath = tmp_path / "test_well.hdf5"
    well.to_hdf5(filepath=filepath)

    # Read back
    well_read = WellData.from_hdf5(filepath=filepath)

    # Verify
    assert well_read.name == well.name
    assert well_read.xpos == well.xpos
    assert well_read.ypos == well.ypos
    assert well_read.zpos == well.zpos
    assert well_read.n_records == well.n_records
    np.testing.assert_array_almost_equal(well_read.survey_x, well.survey_x)
    np.testing.assert_array_almost_equal(well_read.survey_y, well.survey_y)
    np.testing.assert_array_almost_equal(well_read.survey_z, well.survey_z)
    assert well_read.log_names == well.log_names

    # Check logs
    for log_name in well.log_names:
        log_orig = well.get_log(log_name)
        log_read = well_read.get_log(log_name)
        assert log_read is not None
        assert log_orig is not None
        np.testing.assert_array_almost_equal(log_read.values, log_orig.values)


def test_welldata_hdf5_with_discrete_log(tmp_path):
    """Test WellData HDF5 I/O with discrete log."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0, 3.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE", 3: "LIMESTONE"},
    )

    well = WellData(
        name="DiscreteWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0, 103.0]),
        survey_y=np.array([200.0, 201.0, 202.0, 203.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0, 1003.0]),
        logs=(facies_log,),
    )

    filepath = tmp_path / "discrete_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    facies_read = well_read.get_log("FACIES")
    assert facies_read is not None
    assert facies_read.is_discrete
    assert facies_read.code_names == facies_log.code_names
    np.testing.assert_array_almost_equal(facies_read.values, facies_log.values)


def test_welldata_hdf5_compression_blosc(tmp_path):
    """Test HDF5 I/O with blosc compression."""
    gr_log = WellLog(name="GR", values=np.random.rand(100))

    well = WellData(
        name="CompressedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=10.0,
        survey_x=np.linspace(100, 200, 100),
        survey_y=np.linspace(200, 300, 100),
        survey_z=np.linspace(1000, 1100, 100),
        logs=(gr_log,),
    )

    filepath = tmp_path / "compressed_well.hdf5"
    well.to_hdf5(filepath=filepath, compression="blosc")

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.n_records == 100
    gr_read = well_read.get_log("GR")
    assert gr_read is not None
    np.testing.assert_array_almost_equal(gr_read.values, gr_log.values)


def test_welldata_hdf5_compression_lzf(tmp_path):
    """Test HDF5 I/O with lzf compression (default)."""
    poro_log = WellLog(name="PORO", values=np.random.rand(50))

    well = WellData(
        name="LZFWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.linspace(100, 150, 50),
        survey_y=np.linspace(200, 250, 50),
        survey_z=np.linspace(1000, 1050, 50),
        logs=(poro_log,),
    )

    filepath = tmp_path / "lzf_well.hdf5"
    well.to_hdf5(filepath=filepath, compression="lzf")

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.n_records == 50


def test_welldata_hdf5_no_compression(tmp_path):
    """Test HDF5 I/O without compression."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    well = WellData(
        name="NoCompressionWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "no_compression_well.hdf5"
    well.to_hdf5(filepath=filepath, compression=None)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.n_records == 2


def test_welldata_hdf5_with_nan_values(tmp_path):
    """Test HDF5 I/O with NaN values in continuous log."""
    poro_log = WellLog(name="PORO", values=np.array([0.15, np.nan, 0.25, np.nan, 0.35]))

    well = WellData(
        name="NaNWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0, 103.0, 104.0]),
        survey_y=np.array([200.0, 201.0, 202.0, 203.0, 204.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0, 1003.0, 1004.0]),
        logs=(poro_log,),
    )

    filepath = tmp_path / "nan_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    poro_read = well_read.get_log("PORO")
    assert poro_read is not None
    assert np.isnan(poro_read.values[1])
    assert np.isnan(poro_read.values[3])
    assert poro_read.values[0] == 0.15


def test_welldata_hdf5_multiple_logs(tmp_path):
    """Test HDF5 I/O with multiple logs of different types."""
    n = 10
    gr_log = WellLog(name="GR", values=np.random.rand(n) * 100)
    poro_log = WellLog(name="PORO", values=np.random.rand(n) * 0.4)
    perm_log = WellLog(name="PERM", values=np.random.rand(n) * 1000)
    facies_log = WellLog(
        name="FACIES",
        values=np.random.randint(1, 4, n).astype(float),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE", 3: "LIMESTONE"},
    )

    well = WellData(
        name="MultiLogWell",
        xpos=100.0,
        ypos=200.0,
        zpos=15.0,
        survey_x=np.linspace(100, 200, n),
        survey_y=np.linspace(200, 300, n),
        survey_z=np.linspace(1000, 1100, n),
        logs=(gr_log, poro_log, perm_log, facies_log),
    )

    filepath = tmp_path / "multi_log_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.n_records == n
    assert len(well_read.logs) == 4
    assert set(well_read.log_names) == {"GR", "PORO", "PERM", "FACIES"}


def test_welldata_hdf5_using_to_file_from_file(tmp_path):
    """Test HDF5 I/O using to_file/from_file with HDF5 format enum."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    well = WellData(
        name="EnumFormatWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "enum_format_well.hdf5"
    well.to_file(filepath=filepath, fformat=WellFileFormat.HDF5)

    well_read = WellData.from_file(filepath=filepath, fformat=WellFileFormat.HDF5)

    assert well_read.name == well.name
    assert well_read.n_records == 3


# ============================================================================
# BlockedWellData HDF5 tests
# ============================================================================


def test_blockedwell_hdf5_basic_write_read(tmp_path):
    """Test basic BlockedWellData write and read using HDF5 format."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))
    poro_log = WellLog(name="PORO", values=np.array([0.15, 0.20, 0.25]))

    blocked_well = BlockedWellData(
        name="BlockedWell_HDF5",
        xpos=460000.0,
        ypos=5930000.0,
        zpos=25.0,
        survey_x=np.array([460000.0, 460010.0, 460020.0]),
        survey_y=np.array([5930000.0, 5930010.0, 5930020.0]),
        survey_z=np.array([1000.0, 1010.0, 1020.0]),
        logs=(gr_log, poro_log),
        i_index=np.array([10.0, 11.0, 12.0]),
        j_index=np.array([20.0, 21.0, 22.0]),
        k_index=np.array([5.0, 6.0, 7.0]),
    )

    filepath = tmp_path / "blocked_well.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    assert blocked_well_read.name == blocked_well.name
    assert blocked_well_read.n_records == 3
    assert blocked_well_read.n_blocked_cells == 3
    np.testing.assert_array_almost_equal(
        blocked_well_read.i_index, blocked_well.i_index
    )
    np.testing.assert_array_almost_equal(
        blocked_well_read.j_index, blocked_well.j_index
    )
    np.testing.assert_array_almost_equal(
        blocked_well_read.k_index, blocked_well.k_index
    )


def test_blockedwell_hdf5_with_discrete_log(tmp_path):
    """Test BlockedWellData HDF5 I/O with discrete log."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    blocked_well = BlockedWellData(
        name="BlockedDiscreteWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(facies_log,),
        i_index=np.array([5.0, 6.0, 7.0]),
        j_index=np.array([10.0, 11.0, 12.0]),
        k_index=np.array([1.0, 1.0, 2.0]),
    )

    filepath = tmp_path / "blocked_discrete_well.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    facies_read = blocked_well_read.get_log("FACIES")
    assert facies_read is not None
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE"}


def test_blockedwell_hdf5_using_to_file_from_file(tmp_path):
    """Test BlockedWellData HDF5 I/O using to_file/from_file."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    blocked_well = BlockedWellData(
        name="BlockedEnumWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(gr_log,),
        i_index=np.array([5.0, 6.0]),
        j_index=np.array([10.0, 11.0]),
        k_index=np.array([1.0, 1.0]),
    )

    filepath = tmp_path / "blocked_enum_well.hdf5"
    blocked_well.to_file(filepath=filepath, fformat=WellFileFormat.HDF5)

    blocked_well_read = BlockedWellData.from_file(
        filepath=filepath, fformat=WellFileFormat.HDF5
    )

    assert blocked_well_read.name == blocked_well.name
    assert blocked_well_read.n_records == 2
    assert blocked_well_read.n_blocked_cells == 2


def test_welldata_hdf5_no_logs(tmp_path):
    """Test HDF5 I/O with well containing no logs."""
    well = WellData(
        name="NoLogsWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(),
    )

    filepath = tmp_path / "no_logs_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.n_records == 3
    assert len(well_read.logs) == 0


def test_welldata_hdf5_no_rkb(tmp_path):
    """Test HDF5 I/O with well without RKB (zpos=0.0)."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    well = WellData(
        name="NoRKBWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "no_rkb_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.zpos == 0.0


def test_welldata_hdf5_continuous_log_with_metadata(tmp_path):
    """Test HDF5 I/O with continuous log containing metadata tuple."""
    # Create a log with metadata in code_names
    poro_log = WellLog(
        name="PORO",
        values=np.array([0.15, 0.20, 0.25]),
        is_discrete=False,
        code_names=("UNK", "lin"),
    )

    well = WellData(
        name="MetadataWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(poro_log,),
    )

    filepath = tmp_path / "metadata_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    poro_read = well_read.get_log("PORO")
    assert poro_read is not None
    assert not poro_read.is_discrete
    assert poro_read.code_names == ("UNK", "lin")


def test_welldata_hdf5_verify_discrete_flag(tmp_path):
    """Test that is_discrete flag is correctly preserved."""
    # Create a mix of continuous and discrete logs
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]), is_discrete=False)
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    well = WellData(
        name="MixedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log, facies_log),
    )

    filepath = tmp_path / "mixed_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    gr_read = well_read.get_log("GR")
    facies_read = well_read.get_log("FACIES")

    assert gr_read is not None
    assert not gr_read.is_discrete
    assert gr_read.code_names is None

    assert facies_read is not None
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE"}


def test_blockedwell_hdf5_no_logs(tmp_path):
    """Test BlockedWellData HDF5 I/O with no logs."""
    blocked_well = BlockedWellData(
        name="BlockedNoLogs",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(),
        i_index=np.array([5.0, 6.0]),
        j_index=np.array([10.0, 11.0]),
        k_index=np.array([1.0, 1.0]),
    )

    filepath = tmp_path / "blocked_no_logs.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    assert blocked_well_read.name == blocked_well.name
    assert blocked_well_read.n_records == 2
    assert len(blocked_well_read.logs) == 0
    np.testing.assert_array_almost_equal(
        blocked_well_read.i_index, blocked_well.i_index
    )


def test_blockedwell_hdf5_with_nan_indices(tmp_path):
    """Test BlockedWellData HDF5 I/O with NaN indices."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    blocked_well = BlockedWellData(
        name="BlockedNaNIndices",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
        i_index=np.array([5.0, np.nan, 7.0]),
        j_index=np.array([10.0, np.nan, 12.0]),
        k_index=np.array([1.0, np.nan, 2.0]),
    )

    filepath = tmp_path / "blocked_nan_indices.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    assert blocked_well_read.n_records == 3
    assert blocked_well_read.n_blocked_cells == 2  # Only 2 valid cells
    assert np.isnan(blocked_well_read.i_index[1])
    assert np.isnan(blocked_well_read.j_index[1])
    assert np.isnan(blocked_well_read.k_index[1])


def test_blockedwell_hdf5_compression_blosc(tmp_path):
    """Test BlockedWellData HDF5 I/O with blosc compression."""
    gr_log = WellLog(name="GR", values=np.random.rand(50))

    blocked_well = BlockedWellData(
        name="BlockedCompressed",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.linspace(100, 150, 50),
        survey_y=np.linspace(200, 250, 50),
        survey_z=np.linspace(1000, 1050, 50),
        logs=(gr_log,),
        i_index=np.random.randint(1, 20, 50).astype(float),
        j_index=np.random.randint(1, 30, 50).astype(float),
        k_index=np.random.randint(1, 10, 50).astype(float),
    )

    filepath = tmp_path / "blocked_compressed.hdf5"
    blocked_well.to_hdf5(filepath=filepath, compression="blosc")

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    assert blocked_well_read.name == blocked_well.name
    assert blocked_well_read.n_records == 50
    np.testing.assert_array_almost_equal(
        blocked_well_read.i_index, blocked_well.i_index
    )


def test_welldata_hdf5_metadata_roundtrip(tmp_path):
    """Test that all well metadata is preserved through HDF5 roundtrip."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 3.0, 1.0, 2.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE", 3: "LIMESTONE"},
    )
    poro_log = WellLog(
        name="PORO",
        values=np.array([0.15, 0.20, 0.25, 0.18, 0.22]),
        is_discrete=False,
    )

    well = WellData(
        name="ComplexWell_123",
        xpos=460123.45,
        ypos=5932456.78,
        zpos=32.5,
        survey_x=np.array([460123.45, 460124.0, 460125.0, 460126.0, 460127.0]),
        survey_y=np.array([5932456.78, 5932457.0, 5932458.0, 5932459.0, 5932460.0]),
        survey_z=np.array([1000.0, 1010.5, 1020.25, 1030.75, 1040.0]),
        logs=(facies_log, poro_log),
    )

    filepath = tmp_path / "roundtrip_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    # Verify all attributes
    assert well_read.name == well.name
    assert well_read.xpos == well.xpos
    assert well_read.ypos == well.ypos
    assert well_read.zpos == well.zpos
    assert well_read.n_records == well.n_records
    assert well_read.log_names == well.log_names

    # Verify survey arrays
    np.testing.assert_array_almost_equal(well_read.survey_x, well.survey_x)
    np.testing.assert_array_almost_equal(well_read.survey_y, well.survey_y)
    np.testing.assert_array_almost_equal(well_read.survey_z, well.survey_z)

    # Verify logs in detail
    for log_name in well.log_names:
        log_orig = well.get_log(log_name)
        log_read = well_read.get_log(log_name)
        assert log_read is not None
        assert log_orig is not None
        assert log_read.is_discrete == log_orig.is_discrete
        assert log_read.code_names == log_orig.code_names
        np.testing.assert_array_almost_equal(log_read.values, log_orig.values)


# ============================================================================
# Error handling tests
# ============================================================================


def test_welldata_hdf5_invalid_file_no_well_group(tmp_path):
    """Test reading HDF5 file without 'Well' group raises error."""
    import h5py
    import pytest

    filepath = tmp_path / "invalid_no_group.hdf5"

    # Create HDF5 file without the 'Well' group
    with h5py.File(filepath, "w") as fh5:
        fh5.create_group("SomeOtherGroup")

    with pytest.raises(ValueError, match="missing 'Well' group"):
        WellData.from_hdf5(filepath=filepath)


def test_welldata_hdf5_invalid_file_no_metadata(tmp_path):
    """Test reading HDF5 file without metadata raises error."""
    import h5py
    import pytest

    filepath = tmp_path / "invalid_no_metadata.hdf5"

    # Create HDF5 file with Well group but no metadata
    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        grp.attrs["columns"] = np.array(["X_UTME", "Y_UTMN", "Z_TVDSS"], dtype="S")

    with pytest.raises(ValueError, match="missing metadata"):
        WellData.from_hdf5(filepath=filepath)


def test_blockedwell_hdf5_missing_indices(tmp_path):
    """Test reading BlockedWellData from file missing grid indices."""
    import pytest

    # First create a regular well file
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0]))

    well = WellData(
        name="RegularWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0]),
        survey_y=np.array([200.0, 201.0]),
        survey_z=np.array([1000.0, 1001.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "regular_well_not_blocked.hdf5"
    well.to_hdf5(filepath=filepath)

    # Try to read it as BlockedWellData - should fail
    with pytest.raises(
        ValueError, match="does not contain I_INDEX, J_INDEX, and K_INDEX"
    ):
        BlockedWellData.from_hdf5(filepath=filepath)
