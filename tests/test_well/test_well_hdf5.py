"""Tests for Well HDF5 I/O functionality."""

import pathlib

import pandas as pd
import pytest

import xtgeo
from xtgeo.well._well_io import _import_wlogs_hdf5

WELL1 = pathlib.Path("wells/battle/1/WELL09.rmswell")


@pytest.fixture(name="loadwell1")
def fixture_loadwell1(testdata_path):
    """Fixture for loading a well (pytest setup)."""
    return xtgeo.well_from_file(testdata_path / pathlib.Path("wells/reek/1/OP_1.w"))


@pytest.fixture(name="simple_well")
def fixture_simple_well(string_to_well):
    wellstring = """1.01
Unknown
OP_1 0 0 0
4
Zonelog DISC 1 zone1 2 zone2 3 zone3
Poro UNK lin
Perm UNK lin
Facies DISC 0 Background 1 Channel 2 Crevasse
0 0 0 nan -999 -999 -999 -999
1 1 1 1 0.1 0.01 1
2 2 2 1 0.2 0.02 1
3 3 3 2 0.3 0.03 1
4 4 4 2 0.4 0.04 2
5 5 5 3 0.5 0.05 2"""
    well = string_to_well(wellstring)
    yield well


@pytest.mark.parametrize(
    "wlogs, expected_output",
    [
        ({}, {"wlogtypes": {}, "wlogrecords": {}}),
        (
            {"X_UTME": ("CONT", None)},
            {"wlogtypes": {"X_UTME": "CONT"}, "wlogrecords": {"X_UTME": None}},
        ),
        (
            {"ZONELOG": ("DISC", {"0": "ZONE00"})},
            {
                "wlogtypes": {"ZONELOG": "DISC"},
                "wlogrecords": {"ZONELOG": {0: "ZONE00"}},
            },
        ),
    ],
)
def test_import_wlogs_hdf5(wlogs, expected_output):
    assert _import_wlogs_hdf5(wlogs) == expected_output


def test_hdf_io_single(tmp_path, testdata_path):
    """Test HDF io, single well."""
    mywell = xtgeo.well_from_file(testdata_path / WELL1)

    wname = (tmp_path / "hdfwell").with_suffix(".hdf")
    mywell.to_hdf(wname)
    mywell2 = xtgeo.well_from_file(wname, fformat="hdf")
    assert mywell2.nrow == mywell.nrow


def test_import_as_rms_export_as_hdf_many(tmp_path, simple_well):
    """Import RMS and export as HDF5 and compare results."""
    wname = (tmp_path / "$random").with_suffix(".hdf")
    wuse = simple_well.to_hdf(wname, compression=None)

    result = xtgeo.well_from_file(wuse, fformat="hdf5")
    assert result.get_dataframe().equals(simple_well.get_dataframe())


def test_hdf_io_compression_blosc(tmp_path, simple_well):
    """Test HDF5 export/import with blosc compression."""
    wname = (tmp_path / "well_blosc").with_suffix(".hdf")
    simple_well.to_hdf(wname, compression="blosc")

    result = xtgeo.well_from_file(wname, fformat="hdf5")
    assert result.get_dataframe().equals(simple_well.get_dataframe())
    assert result.wname == simple_well.wname


def test_hdf_io_compression_lzf(tmp_path, simple_well):
    """Test HDF5 export/import with lzf compression (default)."""
    wname = (tmp_path / "well_lzf").with_suffix(".hdf")
    simple_well.to_hdf(wname, compression="lzf")

    result = xtgeo.well_from_file(wname, fformat="hdf5")
    assert result.get_dataframe().equals(simple_well.get_dataframe())


def test_hdf_io_metadata_preservation(tmp_path, loadwell1):
    """Test that well metadata is preserved through HDF5 export/import."""
    wname = (tmp_path / "well_metadata").with_suffix(".hdf")
    loadwell1.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check key metadata
    assert result.wname == loadwell1.wname
    assert result.xpos == loadwell1.xpos
    assert result.ypos == loadwell1.ypos
    assert result.rkb == loadwell1.rkb
    assert result.nrow == loadwell1.nrow
    assert result.ncol == loadwell1.ncol


def test_hdf_io_log_types_preservation(tmp_path, simple_well):
    """Test that log types (CONT/DISC) are preserved through HDF5 export/import."""
    wname = (tmp_path / "well_logtypes").with_suffix(".hdf")
    simple_well.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check that log types are preserved
    assert result._wdata.attr_types == simple_well._wdata.attr_types


def test_hdf_io_log_records_preservation(tmp_path, simple_well):
    """Test that log records (units, codes) are preserved through HDF5 export/import."""
    wname = (tmp_path / "well_logrecords").with_suffix(".hdf")
    simple_well.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check that log records are preserved
    for logname in simple_well.lognames:
        expected = simple_well._wdata.attr_records[logname]
        assert result._wdata.attr_records[logname] == expected


def test_hdf_io_with_nan_values(tmp_path, loadwell1):
    """Test HDF5 export/import with wells containing NaN values."""
    wname = (tmp_path / "well_nan").with_suffix(".hdf")
    loadwell1.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check that NaN values are preserved
    original_df = loadwell1.get_dataframe()
    result_df = result.get_dataframe()

    # Compare shapes
    assert result_df.shape == original_df.shape

    # Compare values including NaN handling
    pd.testing.assert_frame_equal(result_df, original_df)


def test_hdf_io_roundtrip_consistency(tmp_path, testdata_path):
    """Test that multiple round-trips through HDF5 maintain data consistency."""
    mywell = xtgeo.well_from_file(testdata_path / WELL1)

    # First round-trip
    wname1 = (tmp_path / "well_rt1").with_suffix(".hdf")
    mywell.to_hdf(wname1)
    well1 = xtgeo.well_from_file(wname1, fformat="hdf5")

    # Second round-trip
    wname2 = (tmp_path / "well_rt2").with_suffix(".hdf")
    well1.to_hdf(wname2)
    well2 = xtgeo.well_from_file(wname2, fformat="hdf5")

    # Both should be identical
    pd.testing.assert_frame_equal(well1.get_dataframe(), well2.get_dataframe())
    assert well1.wname == well2.wname


def test_hdf_io_discrete_logs(tmp_path, simple_well):
    """Test HDF5 export/import specifically for discrete logs with code mappings."""
    wname = (tmp_path / "well_discrete").with_suffix(".hdf")
    simple_well.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check discrete log (Facies) has correct code mapping
    facies_codes = result._wdata.attr_records["Facies"]
    expected_codes = simple_well._wdata.attr_records["Facies"]

    assert isinstance(facies_codes, dict)
    assert facies_codes == expected_codes
    assert facies_codes[0] == "Background"
    assert facies_codes[1] == "Channel"
    assert facies_codes[2] == "Crevasse"


def test_hdf_io_continuous_logs(tmp_path, simple_well):
    """Test HDF5 export/import specifically for continuous logs with unit/scale."""
    wname = (tmp_path / "well_continuous").with_suffix(".hdf")
    simple_well.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    # Check continuous log (Poro) has correct unit/scale tuple
    poro_record = result._wdata.attr_records["Poro"]
    expected_record = simple_well._wdata.attr_records["Poro"]

    assert isinstance(poro_record, tuple)
    assert poro_record == expected_record


def test_hdf_io_all_log_names(tmp_path, loadwell1):
    """Test that all log names are preserved through HDF5 export/import."""
    wname = (tmp_path / "well_lognames").with_suffix(".hdf")
    loadwell1.to_hdf(wname)

    result = xtgeo.well_from_file(wname, fformat="hdf5")

    assert result.lognames == loadwell1.lognames
    assert result.lognames_all == loadwell1.lognames_all
