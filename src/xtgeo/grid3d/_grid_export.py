# -*- coding: utf-8 -*-
import struct
import json

from copy import deepcopy

import h5py
import hdf5plugin


import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

# valid byte settings for xtgeo/hdf5 based export e.g. 441 means
# 4byte float for coord, 4byte float for zcorn, and 1 byte Int for actnums.
# Native XTGeo is 844
VALIDXTGFMT = (221, 421, 441, 444, 841, 844, 881, 884)


def export_roff(self, gfile, option):
    """Export grid to ROFF format (binary)."""
    if self._xtgformat == 1:
        _export_roff_v1(self, gfile, option)
    else:
        _export_roff_v2(self, gfile, option)


def _export_roff_v1(self, gfile, option):
    """Export grid to ROFF format (binary)."""
    self._xtgformat1()

    gfile = xtgeo._XTGeoFile(gfile, mode="wb")
    gfile.check_folder(raiseerror=OSError)

    logger.debug("Export to ROFF...")

    nsubs = 0
    if self.subgrids is None:
        logger.debug("Create a pointer for subgrd_v ...")
        subgrd_v = _cxtgeo.new_intpointer()
    else:
        nsubs = len(self.subgrids)
        subgrd_v = _cxtgeo.new_intarray(nsubs)
        for inum, (sname, sarray) in enumerate(self.subgrids.items()):
            logger.info("INUM SUBGRID: %s %s", inum, sname)
            _cxtgeo.intarray_setitem(subgrd_v, inum, len(sarray))

    # get the geometrics list to find the xshift, etc
    gx = self.get_geometrics()

    _cxtgeo.grd3d_export_roff_grid(
        option,
        self._ncol,
        self._nrow,
        self._nlay,
        nsubs,
        0,
        gx[3],
        gx[5],
        gx[7],
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        subgrd_v,
        gfile.name,
    )

    # end tag
    _cxtgeo.grd3d_export_roff_end(option, gfile.name)


def _export_roff_v2(self, gfile, ascii_fmt):
    """Export grid to ROFF format (binary/ascii) _xtgformat=2."""
    self._xtgformat2()

    gfile = xtgeo._XTGeoFile(gfile, mode="wb")
    gfile.check_folder(raiseerror=OSError)
    cfhandle = gfile.get_cfhandle()

    logger.debug("Export to ROFF... ascii_fmt = %s", ascii_fmt)

    subs = self.get_subgrids()
    if subs:
        sublist = np.array(list(subs.values()), dtype=np.int32)
    else:
        sublist = np.zeros((1), dtype=np.int32)

    # for *shift values
    midi, midj, midk = (self.ncol // 2, self.nrow // 2, self.nlay // 2)
    midx = float(self._coordsv[midi, midj, 0])
    midy = float(self._coordsv[midi, midj, 1])
    midz = float(self._zcornsv[midi, midj, midk, 0])

    info = "#" + xtg.get_xtgeo_info() + "#$"  # last $ is for lineshift trick in roffasc

    _cxtgeo.grdcp3d_export_roff_bin_start_end(
        0, info, ascii_fmt, "grid", self.ncol, self.nrow, self.nlay, cfhandle
    )

    _cxtgeo.grdcp3d_export_roff_grid(
        ascii_fmt,
        self._ncol,
        self._nrow,
        self._nlay,
        midx,
        midy,
        midz,
        sublist,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        cfhandle,
    )

    # # TODO: export assosiated properties

    # end tag
    _cxtgeo.grdcp3d_export_roff_bin_start_end(
        1, info, ascii_fmt, "xxxx", self.ncol, self.nrow, self.nlay, cfhandle
    )
    gfile.cfclose()


def export_grdecl(self, gfile, mode):
    """Export grid to Eclipse GRDECL format (ascii, mode=1) or binary (mode=0)."""
    self._xtgformat1()

    logger.debug("Export to ascii or binary GRDECL...")

    _cxtgeo.grd3d_export_grdecl(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        gfile,
        mode,
    )


def export_egrid(self, gfile):
    """Export grid to Eclipse EGRID format, binary."""
    self._xtgformat1()

    logger.debug("Export to binary EGRID...")

    _cxtgeo.grd3d_export_egrid(
        self._ncol,
        self._nrow,
        self._nlay,
        self._coordsv,
        self._zcornsv,
        self._actnumsv,
        gfile,
        0,
    )


def export_xtgcpgeom(self, gfile, subformat=844):
    """Export grid to binary XTGeo xtgcpgeom format, in prep. and experimental."""
    logger.info("Export to native binary xtgeo...")

    self._xtgformat2()
    logger.info("Export to native binary xtgeo...(2)")

    self.metadata.required = self
    meta = self.metadata.get_metadata()

    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    dtypecoo, dtypezco, dtypeact, meta = _define_dtypes(self, subformat, meta)

    coordsv, zcornsv, actnumsv = _transform_vectors(
        self, meta, dtypecoo, dtypezco, dtypeact
    )

    prevalues = (1, 1301, int(subformat), self.ncol, self.nrow, self.nlay)
    mystruct = struct.Struct("= i i i q q q")
    hdr = mystruct.pack(*prevalues)

    with open(gfile.file, "wb") as fout:
        fout.write(hdr)

    with open(gfile.file, "ab") as fout:
        coordsv.tofile(fout)
        zcornsv.tofile(fout)
        actnumsv.tofile(fout)

    with open(gfile.file, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(gfile.file, "ab") as fout:
        fout.write(json.dumps(meta).encode())

    logger.info("Export to native binary xtgeo... done!")


def export_hdf5_cpgeom(self, gfile, compression=None, chunks=False, subformat=844):
    """Export grid to h5/hdf5, in prep. and experimental."""
    self._xtgformat2()

    logger.info("Export to hdf5 xtgeo layout...")

    self.metadata.required = self
    meta = self.metadata.get_metadata()

    if compression and compression == "blosc":
        compression = hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )

    if not chunks:
        chunks = None

    dtypecoo, dtypezco, dtypeact, meta = _define_dtypes(self, subformat, meta)

    coordsv, zcornsv, actnumsv = _transform_vectors(
        self, meta, dtypecoo, dtypezco, dtypeact
    )

    jmeta = json.dumps(meta).encode()

    with h5py.File(gfile.name, "w") as fh5:

        grp = fh5.create_group("CornerPointGeometry")
        grp.create_dataset(
            "coord",
            data=coordsv,
            compression=compression,
            chunks=chunks,
        )
        grp.create_dataset(
            "zcorn",
            data=zcornsv,
            compression=compression,
            chunks=chunks,
        )
        grp.create_dataset(
            "actnum",
            data=actnumsv,
            compression=compression,
            chunks=chunks,
        )

        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = "xtgeo"
        grp.attrs["format-idcode"] = 1301

    logger.info("Export to hdf5 xtgeo layout... done!")


def _define_dtypes(self, fmt, meta):
    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    logger.debug("Define dtypes...")

    newmeta = deepcopy(meta)

    if int(fmt) not in VALIDXTGFMT:
        raise ValueError(f"The subformat value is not valid, must be in: {VALIDXTGFMT}")

    if fmt <= 444:
        # the float() is for JSON serialization which cannot handle float32
        newmeta["_required_"]["xshift"] = float(self._coordsv[:, :, 0::3].mean())
        newmeta["_required_"]["yshift"] = float(self._coordsv[:, :, 1::3].mean())
        newmeta["_required_"]["zshift"] = float(self._zcornsv.mean())

    nbytecoord, nbytezcorn, nbyteactnum = [int(nbyte) for nbyte in str(fmt)]

    dtype_coord = "float" + str(nbytecoord * 8)
    dtype_zcorn = "float" + str(nbytezcorn * 8)
    dtype_actnum = "int" + str(nbyteactnum * 8)

    logger.debug("Define dtypes... done!")
    return dtype_coord, dtype_zcorn, dtype_actnum, newmeta


def _transform_vectors(self, meta, dtypecoo, dtypezco, dtypeact):

    logger.debug("Transform vectors...")

    xshift = meta["_required_"]["xshift"]
    yshift = meta["_required_"]["yshift"]
    zshift = meta["_required_"]["zshift"]

    if xshift != 0.0 or yshift != 0.0 or zshift != 0.0:
        coordsv = self._coordsv.copy()
        zcornsv = self._zcornsv.copy()
        actnumsv = self._actnumsv.copy()
        coordsv[:, :, 0::3] -= xshift
        coordsv[:, :, 1::3] -= yshift
        coordsv[:, :, 2::3] -= zshift
        zcornsv -= zshift
    else:
        coordsv = self._coordsv
        zcornsv = self._zcornsv
        actnumsv = self._actnumsv

    if dtypecoo != "float64":
        coordsv = coordsv.astype(dtypecoo)

    if dtypecoo != "float32":
        zcornsv = zcornsv.astype(dtypezco)

    if dtypecoo != "int32":
        actnumsv = actnumsv.astype(dtypeact)

    logger.debug("Transform vectors... done!")
    return coordsv, zcornsv, actnumsv
