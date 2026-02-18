"""Tests for HDF5 format I/O for WellData and BlockedWellData."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellFileFormat, WellLog


def test_welldata_hdf5_basic_write_read(tmp_path):
    """Test basic WellData write and read using HDF5 format."""

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

    filepath = tmp_path / "test_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.xpos == well.xpos
    assert well_read.ypos == well.ypos
    assert well_read.zpos == well.zpos
    assert well_read.n_records == well.n_records
    np.testing.assert_array_almost_equal(well_read.survey_x, well.survey_x)
    np.testing.assert_array_almost_equal(well_read.survey_y, well.survey_y)
    np.testing.assert_array_almost_equal(well_read.survey_z, well.survey_z)
    assert well_read.log_names == well.log_names

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


# ======================================================================================
# BlockedWellData HDF5 tests
# ======================================================================================


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

    assert well_read.name == well.name
    assert well_read.xpos == well.xpos
    assert well_read.ypos == well.ypos
    assert well_read.zpos == well.zpos
    assert well_read.n_records == well.n_records
    assert well_read.log_names == well.log_names

    np.testing.assert_array_almost_equal(well_read.survey_x, well.survey_x)
    np.testing.assert_array_almost_equal(well_read.survey_y, well.survey_y)
    np.testing.assert_array_almost_equal(well_read.survey_z, well.survey_z)

    for log_name in well.log_names:
        log_orig = well.get_log(log_name)
        log_read = well_read.get_log(log_name)
        assert log_read is not None
        assert log_orig is not None
        assert log_read.is_discrete == log_orig.is_discrete
        assert log_read.code_names == log_orig.code_names
        np.testing.assert_array_almost_equal(log_read.values, log_orig.values)


# ======================================================================================
# Error handling tests
# ======================================================================================


def test_welldata_hdf5_invalid_file_no_well_group(tmp_path):
    """Test reading HDF5 file without 'Well' group raises error."""

    filepath = tmp_path / "invalid_no_group.hdf5"

    # Create HDF5 file without the 'Well' group
    with h5py.File(filepath, "w") as fh5:
        fh5.create_group("SomeOtherGroup")

    with pytest.raises(ValueError, match="missing 'Well' group"):
        WellData.from_hdf5(filepath=filepath)


def test_welldata_hdf5_invalid_file_no_metadata(tmp_path):
    """Test reading HDF5 file without metadata raises error."""

    filepath = tmp_path / "invalid_no_metadata.hdf5"

    # Create HDF5 file with Well group but no metadata
    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        grp.attrs["columns"] = np.array(["X_UTME", "Y_UTMN", "Z_TVDSS"], dtype="S")

    with pytest.raises(ValueError, match="missing metadata"):
        WellData.from_hdf5(filepath=filepath)


def test_blockedwell_hdf5_missing_indices(tmp_path):
    """Test reading BlockedWellData from file missing grid indices."""

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


def test_welldata_hdf5_legacy_format_compatibility(tmp_path):
    """Test HDF5 format compatibility with legacy importer expectations.

    Regression test for ensuring continuous logs with metadata tuples
    still use 'CONT' as the log type (not the first element of the tuple),
    maintaining backward compatibility with existing xtgeo.Well importers.
    """
    import json

    gr_log = WellLog(
        name="GR",
        values=np.array([50.0, 75.0, 100.0]),
        is_discrete=False,
        code_names=("UNK", "lin"),
    )
    poro_log = WellLog(
        name="PORO",
        values=np.array([0.15, 0.20, 0.25]),
        is_discrete=False,
        code_names=None,  # No code metadata
    )
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    well = WellData(
        name="LegacyCompatWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log, poro_log, facies_log),
    )

    filepath = tmp_path / "legacy_compat_well.hdf5"
    well.to_hdf5(filepath=filepath)

    with h5py.File(filepath, "r") as fh5:
        grp = fh5["Well"]
        jmeta = grp.attrs["metadata"]
        if isinstance(jmeta, bytes):
            jmeta = jmeta.decode()

        meta = json.loads(jmeta)
        wlogs = meta["_required_"]["wlogs"]

        # Verify GR log with metadata tuple uses "CONT" as type
        assert "GR" in wlogs
        gr_type, gr_rec = wlogs["GR"]
        assert gr_type == "CONT", (
            f"Expected 'CONT' for continuous log with metadata, got '{gr_type}'. "
            "This breaks backward compatibility with legacy importers."
        )
        assert gr_rec == ["UNK", "lin"], "Metadata tuple should be preserved"

        # Verify PORO log without metadata uses "CONT" as type
        assert "PORO" in wlogs
        poro_type, poro_rec = wlogs["PORO"]
        assert poro_type == "CONT"
        assert poro_rec is None

        # Verify FACIES discrete log uses "DISC" as type
        assert "FACIES" in wlogs
        facies_type, facies_rec = wlogs["FACIES"]
        assert facies_type == "DISC"
        assert facies_rec == {
            "1": "SAND",
            "2": "SHALE",
        }  # JSON converts int keys to strings

    # Verify round-trip and metadata
    well_read = WellData.from_hdf5(filepath=filepath)
    assert well_read.name == well.name
    assert well_read.n_records == 3

    gr_read = well_read.get_log("GR")
    assert gr_read is not None
    assert not gr_read.is_discrete
    assert gr_read.code_names == ("UNK", "lin")

    poro_read = well_read.get_log("PORO")
    assert poro_read is not None
    assert not poro_read.is_discrete
    assert poro_read.code_names is None

    facies_read = well_read.get_log("FACIES")
    assert facies_read is not None
    assert facies_read.is_discrete
    assert facies_read.code_names == {1: "SAND", 2: "SHALE"}


def test_welldata_hdf5_invalid_log_type(tmp_path):
    """Test that reading HDF5 with invalid log type raises ValueError."""
    import json

    from xtgeo.io._welldata._fformats._hdf5_xtgeo import (
        HDF5_FORMAT_IDCODE,
        HDF5_PROVIDER,
    )

    filepath = tmp_path / "invalid_logtype.hdf5"

    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": 0.0,
            "xpos": 100.0,
            "ypos": 200.0,
            "name": "InvalidTypeWell",
            "wlogs": {
                "GR": ("INVALID_TYPE", None),  # Invalid log type
            },
            "mdlogname": None,
            "zonelogname": None,
        },
    }
    jmeta = json.dumps(metadata).encode()

    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index)
        grp.create_dataset("column/X_UTME", data=np.array([100.0, 101.0, 102.0]))
        grp.create_dataset("column/Y_UTMN", data=np.array([200.0, 201.0, 202.0]))
        grp.create_dataset("column/Z_TVDSS", data=np.array([1000.0, 1001.0, 1002.0]))
        grp.create_dataset("column/GR", data=np.array([50.0, 75.0, 100.0]))
        grp.attrs["columns"] = np.array(
            ["X_UTME", "Y_UTMN", "Z_TVDSS", "GR"], dtype="S"
        )
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    with pytest.raises(ValueError, match="Invalid log type found in input"):
        WellData.from_hdf5(filepath=filepath)


def test_welldata_hdf5_invalid_log_record(tmp_path):
    """Test that reading HDF5 with invalid log record type raises ValueError."""
    import json

    from xtgeo.io._welldata._fformats._hdf5_xtgeo import (
        HDF5_FORMAT_IDCODE,
        HDF5_PROVIDER,
    )

    filepath = tmp_path / "invalid_logrecord.hdf5"

    # Create an HDF5 file with invalid log record type (e.g., a string instead of
    # dict/tuple/None)
    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": 0.0,
            "xpos": 100.0,
            "ypos": 200.0,
            "name": "InvalidRecordWell",
            "wlogs": {
                "GR": ("CONT", "invalid_string_record"),  # Invalid record type
            },
            "mdlogname": None,
            "zonelogname": None,
        },
    }
    jmeta = json.dumps(metadata).encode()

    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index)
        grp.create_dataset("column/X_UTME", data=np.array([100.0, 101.0, 102.0]))
        grp.create_dataset("column/Y_UTMN", data=np.array([200.0, 201.0, 202.0]))
        grp.create_dataset("column/Z_TVDSS", data=np.array([1000.0, 1001.0, 1002.0]))
        grp.create_dataset("column/GR", data=np.array([50.0, 75.0, 100.0]))
        grp.attrs["columns"] = np.array(
            ["X_UTME", "Y_UTMN", "Z_TVDSS", "GR"], dtype="S"
        )
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    with pytest.raises(ValueError, match="Invalid log record found in input"):
        WellData.from_hdf5(filepath=filepath)


def test_welldata_hdf5_invalid_compression(tmp_path):
    """Test that invalid compression method raises ValueError."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "invalid_compression.hdf5"

    # Test with invalid compression string
    with pytest.raises(ValueError, match="Unsupported compression 'gzip'"):
        well.to_hdf5(filepath=filepath, compression="gzip")

    # Test with another invalid compression string
    with pytest.raises(ValueError, match="Unsupported compression 'zstd'"):
        well.to_hdf5(filepath=filepath, compression="zstd")


def test_blockedwell_hdf5_invalid_compression(tmp_path):
    """Test that invalid compression method raises ValueError for blocked well."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    blocked_well = BlockedWellData(
        name="TestBlockedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(facies_log,),
        i_index=np.array([1.0, 1.0, 2.0]),
        j_index=np.array([2.0, 2.0, 3.0]),
        k_index=np.array([5.0, 6.0, 7.0]),
    )

    filepath = tmp_path / "invalid_compression_blocked.hdf5"

    with pytest.raises(ValueError, match="Unsupported compression 'bz2'"):
        blocked_well.to_hdf5(filepath=filepath, compression="bz2")


def test_welldata_hdf5_missing_required_metadata_fields(tmp_path):
    """Test that missing required metadata fields raise ValueError."""
    import json

    from xtgeo.io._welldata._fformats._hdf5_xtgeo import (
        HDF5_FORMAT_IDCODE,
        HDF5_PROVIDER,
    )

    filepath = tmp_path / "missing_name.hdf5"
    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": 0.0,
            "xpos": 100.0,
            "ypos": 200.0,
            # "name" is missing
            "wlogs": {},
        },
    }
    jmeta = json.dumps(metadata).encode()

    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index)
        grp.create_dataset("column/X_UTME", data=np.array([100.0, 101.0, 102.0]))
        grp.create_dataset("column/Y_UTMN", data=np.array([200.0, 201.0, 202.0]))
        grp.create_dataset("column/Z_TVDSS", data=np.array([1000.0, 1001.0, 1002.0]))
        grp.attrs["columns"] = np.array(["X_UTME", "Y_UTMN", "Z_TVDSS"], dtype="S")
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    with pytest.raises(ValueError, match="missing required metadata fields.*'name'"):
        WellData.from_hdf5(filepath=filepath)

    # Test missing 'wlogs'
    filepath = tmp_path / "missing_wlogs.hdf5"
    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": 0.0,
            "xpos": 100.0,
            "ypos": 200.0,
            "name": "TestWell",
            # "wlogs" is missing
        },
    }
    jmeta = json.dumps(metadata).encode()

    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index)
        grp.create_dataset("column/X_UTME", data=np.array([100.0, 101.0, 102.0]))
        grp.create_dataset("column/Y_UTMN", data=np.array([200.0, 201.0, 202.0]))
        grp.create_dataset("column/Z_TVDSS", data=np.array([1000.0, 1001.0, 1002.0]))
        grp.attrs["columns"] = np.array(["X_UTME", "Y_UTMN", "Z_TVDSS"], dtype="S")
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    with pytest.raises(ValueError, match="missing required metadata fields.*'wlogs'"):
        WellData.from_hdf5(filepath=filepath)


def test_blockedwell_hdf5_missing_required_metadata_fields(tmp_path):
    """Test that missing required metadata fields raise ValueError for blocked well."""
    import json

    from xtgeo.io._welldata._fformats._hdf5_xtgeo import (
        HDF5_FORMAT_IDCODE,
        HDF5_PROVIDER,
    )

    filepath = tmp_path / "missing_fields_blocked.hdf5"
    metadata = {
        "_class_": "BlockedWell",
        "_required_": {
            "rkb": 0.0,
            # "xpos", "ypos", "name", "wlogs" are all missing
        },
    }
    jmeta = json.dumps(metadata).encode()

    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index)
        grp.create_dataset("column/X_UTME", data=np.array([100.0, 101.0, 102.0]))
        grp.create_dataset("column/Y_UTMN", data=np.array([200.0, 201.0, 202.0]))
        grp.create_dataset("column/Z_TVDSS", data=np.array([1000.0, 1001.0, 1002.0]))
        grp.create_dataset("column/I_INDEX", data=np.array([1.0, 1.0, 2.0]))
        grp.create_dataset("column/J_INDEX", data=np.array([2.0, 2.0, 3.0]))
        grp.create_dataset("column/K_INDEX", data=np.array([5.0, 6.0, 7.0]))
        grp.attrs["columns"] = np.array(
            ["X_UTME", "Y_UTMN", "Z_TVDSS", "I_INDEX", "J_INDEX", "K_INDEX"],
            dtype="S",
        )
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    with pytest.raises(
        ValueError, match="missing required metadata fields.*'xpos'.*'ypos'"
    ):
        BlockedWellData.from_hdf5(filepath=filepath)


def test_welldata_hdf5_no_kwargs_accepted(tmp_path):
    """Test that from_hdf5 doesn't accept unsupported keyword arguments."""
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
    )

    filepath = tmp_path / "test_well.hdf5"
    well.to_hdf5(filepath=filepath)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        WellData.from_hdf5(filepath=filepath, unsupported_param="value")

    with pytest.raises(
        TypeError, match="from_hdf5\\(\\) does not accept keyword arguments"
    ):
        WellData.from_file(
            filepath=filepath, fformat=WellFileFormat.HDF5, unsupported_param="value"
        )


def test_blockedwell_hdf5_no_kwargs_accepted(tmp_path):
    """Test that from_hdf5 doesn't accept unsupported kw args for blocked well."""
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE"},
    )

    blocked_well = BlockedWellData(
        name="TestBlockedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(facies_log,),
        i_index=np.array([1.0, 1.0, 2.0]),
        j_index=np.array([2.0, 2.0, 3.0]),
        k_index=np.array([5.0, 6.0, 7.0]),
    )

    filepath = tmp_path / "test_blocked_well.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        BlockedWellData.from_hdf5(filepath=filepath, unsupported_param="value")

    with pytest.raises(
        TypeError, match="from_hdf5\\(\\) does not accept keyword arguments"
    ):
        BlockedWellData.from_file(
            filepath=filepath, fformat=WellFileFormat.HDF5, unsupported_param="value"
        )


# ======================================================================================
# Additional edge case tests
# ======================================================================================


def test_welldata_hdf5_empty_well_zero_records(tmp_path):
    """Test HDF5 I/O with well containing zero records (empty arrays)."""

    well = WellData(
        name="EmptyWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([]),
        survey_y=np.array([]),
        survey_z=np.array([]),
        logs=(),
    )

    filepath = tmp_path / "empty_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == well.name
    assert well_read.n_records == 0
    assert len(well_read.survey_x) == 0
    assert len(well_read.logs) == 0


def test_welldata_hdf5_special_characters_in_names(tmp_path):
    """Test HDF5 I/O with special characters in well and log names."""

    log1 = WellLog(name="GR-API", values=np.array([50.0, 75.0]))
    log2 = WellLog(name="PORO_%", values=np.array([0.15, 0.20]))
    log3 = WellLog(name="SW@65C", values=np.array([0.5, 0.6]))

    well = WellData(
        name="Well-31/2-E-4 AH #2",  # Realistic well name with special chars
        xpos=460000.0,
        ypos=5930000.0,
        zpos=25.5,
        survey_x=np.array([460000.0, 460010.0]),
        survey_y=np.array([5930000.0, 5930010.0]),
        survey_z=np.array([1000.0, 1010.0]),
        logs=(log1, log2, log3),
    )

    filepath = tmp_path / "special_chars_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == "Well-31/2-E-4 AH #2"
    assert "GR-API" in well_read.log_names
    assert "PORO_%" in well_read.log_names
    assert "SW@65C" in well_read.log_names

    log1_read = well_read.get_log("GR-API")
    assert log1_read is not None
    np.testing.assert_array_almost_equal(log1_read.values, log1.values)


def test_welldata_hdf5_very_long_log_names(tmp_path):
    """Test HDF5 I/O with very long log names."""

    long_name = "A" * 200  # 200 character log name
    log_long = WellLog(name=long_name, values=np.array([50.0, 75.0, 100.0]))

    well = WellData(
        name="LongNameWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(log_long,),
    )

    filepath = tmp_path / "long_name_well.hdf5"
    well.to_hdf5(filepath=filepath)

    well_read = WellData.from_hdf5(filepath=filepath)

    assert long_name in well_read.log_names
    log_read = well_read.get_log(long_name)
    assert log_read is not None
    np.testing.assert_array_almost_equal(log_read.values, log_long.values)


def test_welldata_hdf5_cross_compatibility_with_legacy(tmp_path):
    """Test that files written by new code can be read by legacy importer.

    This verifies backward compatibility by writing with the new WellData format
    and reading with the legacy import_hdf5_well function.
    """
    from xtgeo.io._file import FileWrapper
    from xtgeo.well._well_io import import_hdf5_well

    # Create well with new WellData format
    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0, 125.0]))
    poro_log = WellLog(
        name="PORO",
        values=np.array([0.15, 0.20, 0.25, 0.18]),
        is_discrete=False,
    )
    facies_log = WellLog(
        name="FACIES",
        values=np.array([1.0, 2.0, 1.0, 3.0]),
        is_discrete=True,
        code_names={1: "SAND", 2: "SHALE", 3: "LIMESTONE"},
    )

    well = WellData(
        name="CompatibilityWell",
        xpos=460123.45,
        ypos=5932456.78,
        zpos=32.5,
        survey_x=np.array([460123.45, 460124.0, 460125.0, 460126.0]),
        survey_y=np.array([5932456.78, 5932457.0, 5932458.0, 5932459.0]),
        survey_z=np.array([1000.0, 1010.0, 1020.0, 1030.0]),
        logs=(gr_log, poro_log, facies_log),
    )

    filepath = tmp_path / "cross_compat_well.hdf5"
    well.to_hdf5(filepath=filepath, compression="lzf")

    # Read using legacy importer
    wrapper = FileWrapper(filepath, mode="r")
    legacy_result = import_hdf5_well(wrapper)

    # Verify legacy importer can read the file correctly
    assert legacy_result["wname"] == "CompatibilityWell"
    assert legacy_result["xpos"] == 460123.45
    assert legacy_result["ypos"] == 5932456.78
    assert legacy_result["rkb"] == 32.5

    df = legacy_result["df"]
    assert len(df) == 4
    assert "X_UTME" in df.columns
    assert "Y_UTMN" in df.columns
    assert "Z_TVDSS" in df.columns
    assert "GR" in df.columns
    assert "PORO" in df.columns
    assert "FACIES" in df.columns

    assert legacy_result["wlogtypes"]["GR"] == "CONT"
    assert legacy_result["wlogtypes"]["PORO"] == "CONT"
    assert legacy_result["wlogtypes"]["FACIES"] == "DISC"

    assert legacy_result["wlogrecords"]["FACIES"] == {
        1: "SAND",
        2: "SHALE",
        3: "LIMESTONE",
    }

    np.testing.assert_array_almost_equal(df["GR"].values, gr_log.values)
    np.testing.assert_array_almost_equal(df["PORO"].values, poro_log.values)
    np.testing.assert_array_almost_equal(df["FACIES"].values, facies_log.values)


def test_welldata_hdf5_forward_compatibility_read_legacy(tmp_path):
    """Test that new code can read files written by legacy export_hdf5_well.

    This verifies forward compatibility by simulating a file structure
    that would be created by the legacy Well.to_hdf5() method.
    """
    import json

    from xtgeo.io._welldata._fformats._hdf5_xtgeo import (
        HDF5_FORMAT_IDCODE,
        HDF5_PROVIDER,
    )

    filepath = tmp_path / "legacy_format_well.hdf5"

    # Simulate legacy file structure as created by export_hdf5_well
    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": 25.0,
            "xpos": 460000.0,
            "ypos": 5930000.0,
            "name": "LegacyWell",
            "wlogs": {
                "GR": ("CONT", None),
                "FACIES": ("DISC", {"1": "SAND", "2": "SHALE"}),
            },
            "mdlogname": None,
            "zonelogname": None,
        },
    }
    jmeta = json.dumps(metadata).encode()

    # Create HDF5 file with legacy structure
    with h5py.File(filepath, "w") as fh5:
        grp = fh5.create_group("Well")

        # Legacy format uses dataframe index
        index = np.arange(3, dtype=np.int64)
        grp.create_dataset("index", data=index, chunks=True, compression="lzf")

        # Store columns
        grp.create_dataset(
            "column/X_UTME",
            data=np.array([460000.0, 460010.0, 460020.0]),
            chunks=True,
            compression="lzf",
        )
        grp.create_dataset(
            "column/Y_UTMN",
            data=np.array([5930000.0, 5930010.0, 5930020.0]),
            chunks=True,
            compression="lzf",
        )
        grp.create_dataset(
            "column/Z_TVDSS",
            data=np.array([1000.0, 1010.0, 1020.0]),
            chunks=True,
            compression="lzf",
        )
        grp.create_dataset(
            "column/GR",
            data=np.array([50.0, 75.0, 100.0]),
            chunks=True,
            compression="lzf",
        )
        grp.create_dataset(
            "column/FACIES",
            data=np.array([1.0, 2.0, 1.0]),
            chunks=True,
            compression="lzf",
        )

        grp.attrs["columns"] = np.array(
            ["X_UTME", "Y_UTMN", "Z_TVDSS", "GR", "FACIES"], dtype="S"
        )
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    # Read using new WellData importer and verify
    well_read = WellData.from_hdf5(filepath=filepath)

    assert well_read.name == "LegacyWell"
    assert well_read.xpos == 460000.0
    assert well_read.ypos == 5930000.0
    assert well_read.zpos == 25.0
    assert well_read.n_records == 3

    assert "GR" in well_read.log_names
    assert "FACIES" in well_read.log_names

    gr_log = well_read.get_log("GR")
    assert gr_log is not None
    assert not gr_log.is_discrete
    np.testing.assert_array_almost_equal(gr_log.values, np.array([50.0, 75.0, 100.0]))

    facies_log = well_read.get_log("FACIES")
    assert facies_log is not None
    assert facies_log.is_discrete
    assert facies_log.code_names == {1: "SAND", 2: "SHALE"}
    np.testing.assert_array_almost_equal(facies_log.values, np.array([1.0, 2.0, 1.0]))


def test_blockedwell_hdf5_empty_well_zero_records(tmp_path):
    """Test HDF5 I/O with blocked well containing zero records."""

    blocked_well = BlockedWellData(
        name="EmptyBlockedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([]),
        survey_y=np.array([]),
        survey_z=np.array([]),
        logs=(),
        i_index=np.array([]),
        j_index=np.array([]),
        k_index=np.array([]),
    )

    filepath = tmp_path / "empty_blocked_well.hdf5"
    blocked_well.to_hdf5(filepath=filepath)

    blocked_well_read = BlockedWellData.from_hdf5(filepath=filepath)

    assert blocked_well_read.name == blocked_well.name
    assert blocked_well_read.n_records == 0
    assert blocked_well_read.n_blocked_cells == 0
    assert len(blocked_well_read.i_index) == 0


def test_welldata_hdf5_rejects_bytesio(tmp_path):
    """Test that HDF5 I/O rejects BytesIO with clear error message."""
    from io import BytesIO

    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    well = WellData(
        name="TestWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
    )

    bytesio = BytesIO()
    with pytest.raises(TypeError, match="does not support in-memory streams"):
        well.to_hdf5(filepath=bytesio)

    with pytest.raises(TypeError, match="does not support in-memory streams"):
        WellData.from_hdf5(filepath=bytesio)


def test_blockedwell_hdf5_rejects_bytesio(tmp_path):
    """Test that BlockedWellData HDF5 I/O rejects BytesIO with clear error message."""
    from io import BytesIO

    gr_log = WellLog(name="GR", values=np.array([50.0, 75.0, 100.0]))

    blocked_well = BlockedWellData(
        name="TestBlockedWell",
        xpos=100.0,
        ypos=200.0,
        zpos=0.0,
        survey_x=np.array([100.0, 101.0, 102.0]),
        survey_y=np.array([200.0, 201.0, 202.0]),
        survey_z=np.array([1000.0, 1001.0, 1002.0]),
        logs=(gr_log,),
        i_index=np.array([1.0, 2.0, 3.0]),
        j_index=np.array([1.0, 2.0, 3.0]),
        k_index=np.array([1.0, 1.0, 1.0]),
    )

    bytesio = BytesIO()
    with pytest.raises(TypeError, match="does not support in-memory streams"):
        blocked_well.to_hdf5(filepath=bytesio)

    with pytest.raises(TypeError, match="does not support in-memory streams"):
        BlockedWellData.from_hdf5(filepath=bytesio)
