"""Test StringIO/BytesIO support for well_from_file and blockedwell_from_file."""

from io import StringIO
from os.path import join

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
