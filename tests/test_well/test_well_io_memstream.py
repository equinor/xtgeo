"""Test in-memory stream (StringIO/BytesIO) I/O for Well and BlockedWell objects."""

from io import StringIO
from os.path import join

import pytest

import xtgeo


def test_well_from_file_stringio(testdata_path):
    """Test reading well from StringIO."""
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")

    # Read file content
    with open(wfile, "r") as f:
        content = f.read()

    # Create StringIO and read
    sio = StringIO(content)
    well = xtgeo.well_from_file(sio, fformat="rms_ascii")

    assert well.name == "WELLX"
    assert well.nrow > 0


def test_well_from_file_csv_stringio(testdata_path):
    """Test reading well from CSV StringIO."""
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")

    # Read well and export to CSV StringIO
    well1 = xtgeo.well_from_file(wfile)
    sio = StringIO()
    well1.to_file(sio, fformat="csv")

    # Read back from StringIO
    sio.seek(0)
    well2 = xtgeo.well_from_file(sio, fformat="csv", wellname=well1.name)

    assert well2.name == well1.name
    assert well2.nrow == well1.nrow


def test_blockedwell_from_file_stringio(testdata_path):
    """Test reading blocked well from StringIO."""
    bwfile = join(testdata_path, "wells/reek/1/OP_1.bw")

    # Read file content
    with open(bwfile, "r") as f:
        content = f.read()

    # Create StringIO and read
    sio = StringIO(content)
    bwell = xtgeo.blockedwell_from_file(sio, fformat="rms_ascii")

    assert bwell.name == "OP_1"
    assert bwell.nrow > 0


def test_well_to_file_rms_ascii_stringio(testdata_path):
    """Test writing well to StringIO with RMS ASCII format."""
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    well = xtgeo.well_from_file(wfile)

    # Export to StringIO
    sio = StringIO()
    well.to_file(sio, fformat="rms_ascii")

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.well_from_file(sio, fformat="rms_ascii")

    assert reimported.name == well.name
    assert reimported.nrow == well.nrow
    assert reimported.xpos == pytest.approx(well.xpos)
    assert reimported.ypos == pytest.approx(well.ypos)


def test_well_to_file_csv_stringio(testdata_path):
    """Test writing well to StringIO with CSV format."""
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    well = xtgeo.well_from_file(wfile)

    # Export to CSV StringIO
    sio = StringIO()
    well.to_file(sio, fformat="csv")

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.well_from_file(sio, fformat="csv", wellname=well.name)

    assert reimported.name == well.name
    assert reimported.nrow == well.nrow
    assert reimported.xpos == pytest.approx(well.xpos)


def test_blockedwell_to_file_rms_ascii_stringio(testdata_path):
    """Test writing blocked well to StringIO with RMS ASCII format."""
    bwfile = join(testdata_path, "wells/reek/1/OP_1.bw")
    bwell = xtgeo.blockedwell_from_file(bwfile)

    # Export to StringIO
    sio = StringIO()
    bwell.to_file(sio, fformat="rms_ascii")

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.blockedwell_from_file(sio, fformat="rms_ascii")

    assert reimported.name == bwell.name
    assert reimported.nrow == bwell.nrow

    # Verify grid indices preserved
    df_original = bwell.get_dataframe()
    df_reimported = reimported.get_dataframe()
    if "I_INDEX" in df_original.columns:
        assert "I_INDEX" in df_reimported.columns
        assert df_reimported["I_INDEX"].iloc[0] == pytest.approx(
            df_original["I_INDEX"].iloc[0]
        )


def test_blockedwell_to_file_csv_stringio(testdata_path):
    """Test writing blocked well to StringIO with CSV format."""
    bwfile = join(testdata_path, "wells/reek/1/OP_1.bw")
    bwell = xtgeo.blockedwell_from_file(bwfile)

    # Export to CSV StringIO
    sio = StringIO()
    bwell.to_file(sio, fformat="csv")

    # Read back from StringIO
    sio.seek(0)
    reimported = xtgeo.blockedwell_from_file(sio, fformat="csv", wellname=bwell.name)

    assert reimported.name == bwell.name
    assert reimported.nrow == bwell.nrow

    # Verify grid indices preserved in CSV
    df_original = bwell.get_dataframe()
    df_reimported = reimported.get_dataframe()
    if "I_INDEX" in df_original.columns:
        assert "I_INDEX" in df_reimported.columns


def test_well_roundtrip_stringio_preserves_logs(testdata_path):
    """Test that StringIO roundtrip preserves log values."""
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    well = xtgeo.well_from_file(wfile)

    # Export via StringIO and reimport
    sio = StringIO()
    well.to_file(sio, fformat="rms_ascii")
    sio.seek(0)
    reimported = xtgeo.well_from_file(sio, fformat="rms_ascii")

    # Compare dataframes
    df_original = well.get_dataframe()
    df_reimported = reimported.get_dataframe()

    assert len(df_original) == len(df_reimported)
    assert set(df_original.columns) == set(df_reimported.columns)

    # Check coordinate columns
    assert df_reimported["X_UTME"].iloc[0] == pytest.approx(
        df_original["X_UTME"].iloc[0]
    )
    assert df_reimported["Y_UTMN"].iloc[0] == pytest.approx(
        df_original["Y_UTMN"].iloc[0]
    )
    assert df_reimported["Z_TVDSS"].iloc[0] == pytest.approx(
        df_original["Z_TVDSS"].iloc[0]
    )
