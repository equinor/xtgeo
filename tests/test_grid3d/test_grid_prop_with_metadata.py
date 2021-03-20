"""Module for testing grid and gridproperty exporting metadata."""

import h5py
import xtgeo

xtg = xtgeo.XTGeoDialog()

TMPDX = xtg.tmpdirobj
TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

SFILE = TPATH / "3dgrids/etc/dual_grid_w_props.roff"


def test_grid_with_metadata_roff():
    """Import grid, add a small metadata piece and export to roff."""

    grd = xtgeo.Grid(SFILE)
    grd.metadata.freeform = {"THIS": "grid"}

    fname = "grd_w_metaddata.roff"
    grd.to_file(TMPDX / fname, metadata=True)

    metadatafilename = "." + fname.replace(".roff", ".yml")

    assert (TMPDX / metadatafilename).exists() is True
    with open(TMPDX / metadatafilename) as stream:
        res = stream.readlines()

    assert "THIS" in str(res)


def test_grid_with_metadata_hdf():
    """Import grid, add a small metadata piece and export to hdf."""

    grd = xtgeo.Grid(SFILE)
    grd.metadata.freeform = {"THIS": "grid"}

    fname = "grd_w_metaddata.hdf"
    grd.to_hdf(TMPDX / fname)

    with h5py.File(TMPDX / fname) as hstream:

        meta = hstream["CornerPointGeometry"].attrs["metadata"]
        assert "_freeform_" in meta
        assert '"THIS": "grid"' in meta


def test_gridprop_with_metadata_roff():
    """Import gridprop, add a small metadata piece and export."""

    grdprop = xtgeo.GridProperty(SFILE, name="POROM")
    grdprop.metadata.freeform = {"THIS": "grid"}

    fname = "grdprop_w_metaddata.roff"
    grdprop.to_file(TMPDX / fname, metadata=True)

    metadatafilename = "." + fname.replace(".roff", ".yml")

    assert (TMPDX / metadatafilename).exists() is True
    with open(TMPDX / metadatafilename) as stream:
        res = stream.readlines()

    assert "THIS" in str(res)


def test_gridprop_with_metadata_hdf():
    """Import grid property, add a small metadata piece and export/import to hdf."""

    grdp = xtgeo.GridProperty(SFILE, name="POROM")
    grdp.metadata.freeform = {"THIS": "grid"}

    fname = "grdprop_w_metaddata.hdf"
    grdp.to_hdf(TMPDX / fname, compression="blosc")

    # check in file
    with h5py.File(TMPDX / fname) as hstream:

        meta = hstream["CPGridProperty"].attrs["metadata"]
        assert "_freeform_" in meta
        assert '"THIS": "grid"' in meta

    # import file
    newp = xtgeo.GridProperty()
    newp.from_hdf(TMPDX / fname)

    assert newp._ncol == grdp._ncol
    print(newp.metadata.freeform)
