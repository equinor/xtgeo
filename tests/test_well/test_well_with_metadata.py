"""Module for testing well with metadata."""

import xtgeo

xtg = xtgeo.XTGeoDialog()

TMPDX = xtg.tmpdirobj
TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

WFILE = TPATH / "wells/reek/1/OP_1.w"


def test_rms_file_with_metadata():
    """Import well from file, as a small metadata piece and export."""

    inwell = xtgeo.Well(WFILE)

    inwell.metadata.freeform = {"THIS": "ismetadata"}

    inwell.to_file(TMPDX / "well_w_metaddata.w", metadata=True)

    metadatafilename = "." + "well_w_metaddata.w".replace(".w", ".yml")

    assert (TMPDX / metadatafilename).exists() is True

    with open(TMPDX / metadatafilename) as stream:
        res = stream.readlines()

    assert "THIS" in str(res)
