"""Test in-memory stream (StringIO/BytesIO) I/O for Wells/BlockedWells collections."""

from io import StringIO
from os.path import join

import xtgeo


def test_wells_from_stacked_rms_ascii_stringio(testdata_path):
    """Test reading multiple wells from StringIO with RMS ASCII format."""
    # Read the actual file first
    wfile = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    with open(wfile, "r") as f:
        content = f.read()

    # Create a stacked file by duplicating the content
    stacked_content = content + "\n" + content

    # Create StringIO object
    sio = StringIO(stacked_content)

    # Import from StringIO
    wells = xtgeo.wells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(wells.wells) == 2
    assert all(isinstance(w, xtgeo.Well) for w in wells.wells)


def test_wells_to_stacked_rms_ascii_stringio(testdata_path):
    """Test writing multiple wells to StringIO with RMS ASCII format."""
    wfile1 = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    wfile2 = join(testdata_path, "wells/reek/1/OP_1.w")

    well1 = xtgeo.well_from_file(wfile1)
    well2 = xtgeo.well_from_file(wfile2)

    wells = xtgeo.Wells([well1, well2])

    sio = StringIO()
    wells.to_stacked_file(sio, fformat="rms_ascii_stacked")

    sio.seek(0)
    wells_loaded = xtgeo.wells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(wells_loaded.wells) == 2
    assert wells_loaded.wells[0].name == well1.name
    assert wells_loaded.wells[1].name == well2.name


def test_wells_csv_stringio_roundtrip(testdata_path):
    """Test CSV roundtrip with StringIO for multiple wells."""
    wfile1 = join(testdata_path, "wells/battle/1/WELLX.rmswell")
    wfile2 = join(testdata_path, "wells/reek/1/OP_1.w")

    well1 = xtgeo.well_from_file(wfile1)
    well2 = xtgeo.well_from_file(wfile2)

    wells = xtgeo.Wells([well1, well2])

    # Export to CSV StringIO
    sio = StringIO()
    wells.to_stacked_file(sio, fformat="csv")

    # Read back from StringIO
    sio.seek(0)
    wells_loaded = xtgeo.wells_from_stacked_file(sio, fformat="csv")

    assert len(wells_loaded.wells) == 2
    assert wells_loaded.wells[0].name == well1.name
    assert wells_loaded.wells[1].name == well2.name


def test_blockedwells_from_stacked_stringio(testdata_path):
    """Test reading blocked wells from StringIO."""
    bwfile = join(testdata_path, "wells/reek/1/OP_1.bw")

    # Read the actual file
    with open(bwfile, "r") as f:
        content = f.read()

    # Create stacked content
    stacked_content = content + "\n" + content

    # Create StringIO and read
    sio = StringIO(stacked_content)
    bwells = xtgeo.blockedwells_from_stacked_file(sio, fformat="rms_ascii_stacked")

    assert len(bwells.wells) == 2
    assert all(isinstance(bw, xtgeo.BlockedWell) for bw in bwells.wells)


def test_blockedwells_to_stacked_stringio(testdata_path):
    """Test writing blocked wells to StringIO."""
    bwfile1 = join(testdata_path, "wells/reek/1/OP_1.bw")

    # Use same file twice for simplicity
    bw1 = xtgeo.blockedwell_from_file(bwfile1)
    bw2 = xtgeo.blockedwell_from_file(bwfile1)

    bwells = xtgeo.BlockedWells([bw1, bw2])

    # Export to StringIO
    sio = StringIO()
    bwells.to_stacked_file(sio, fformat="rms_ascii_stacked")

    # Read back
    sio.seek(0)
    bwells_loaded = xtgeo.blockedwells_from_stacked_file(
        sio, fformat="rms_ascii_stacked"
    )

    assert len(bwells_loaded.wells) == 2
    assert bwells_loaded.wells[0].name == bw1.name
    assert bwells_loaded.wells[1].name == bw2.name
