# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import struct
import json
import numpy as np
import xtgeo
from xtgeo.common import XTGeoDialog
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def export_roff(self, gfile, option):
    """Export grid to ROFF format (binary)"""
    if self._xtgformat == 1:
        _export_roff_v1(self, gfile, option)
    else:
        _export_roff_v2(self, gfile, option)


def _export_roff_v1(self, gfile, option):
    """Export grid to ROFF format (binary)"""

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
    """Export grid to ROFF format (binary/ascii) _xtgformat=2"""

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
    """Export grid to binary XTGeo xtgcpgeom format, in prep. and experimental"""
    self._xtgformat2()
    self.metadata.required = self

    gfile = xtgeo._XTGeoFile(gfile, mode="wb")

    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    if int(subformat) not in (444, 844, 841, 881, 884):
        raise ValueError("The subformat value ins not valid")

    coordfmt, zcornfmt, actnumfmt = [int(nbyte) for nbyte in str(subformat)]

    coordsv = self._coordsv
    zcornsv = self._zcornsv
    actnumv = self._actnumsv

    if coordfmt != 8:
        coordsv = self._coordsv.astype("float" + str(coordfmt * 8))
    if zcornfmt != 4:
        zcornsv = self._zcornsv.astype("float" + str(zcornfmt * 8))
    if actnumfmt != 4:
        actnumv = self._actnumsv.astype("int" + str(actnumfmt * 8))

    prevalues = (1, 1301, int(subformat), self.ncol, self.nrow, self.nlay)
    mystruct = struct.Struct("= i i i q q q")
    hdr = mystruct.pack(*prevalues)

    meta = self.metadata.get_metadata()

    with open(gfile.name, "wb") as fout:
        fout.write(hdr)

    with open(gfile.name, "ab") as fout:
        coordsv.tofile(fout)
        zcornsv.tofile(fout)
        actnumv.tofile(fout)

    with open(gfile.name, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(gfile.name, "ab") as fout:
        fout.write(json.dumps(meta).encode())
