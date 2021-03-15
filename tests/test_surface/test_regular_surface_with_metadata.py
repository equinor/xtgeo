"""Module for testing well with metadata."""

import xtgeo

xtg = xtgeo.XTGeoDialog()

TMPDX = xtg.tmpdirobj
TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

SFILE = TPATH / "surfaces/reek/1/reek_fwl.gri"


def test_regsurf_with_metadata():
    """Import regsurf from file, add a small metadata piece and export."""

    insurf = xtgeo.RegularSurface(SFILE)

    insurf.metadata.freeform = {"THIS": "ismetadata"}

    fname = "regsurf_w_metaddata.gri"
    insurf.to_file(TMPDX / fname, metadata=True)

    metadatafilename = "." + fname.replace(".gri", ".yml")

    assert (TMPDX / metadatafilename).exists() is True
    with open(TMPDX / metadatafilename) as stream:
        res = stream.readlines()

    assert "THIS" in str(res)
