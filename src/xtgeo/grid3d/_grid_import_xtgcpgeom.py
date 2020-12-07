# coding: utf-8
"""Private module, Grid Import private functions for ROFF format."""

from collections import OrderedDict
from struct import unpack
import json

import numpy as np

import xtgeo.common.sys as xsys
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_xtgcpgeom(self, mfile):
    """Using pure python for experimental grid geometry import."""
    #
    offset = 36
    with open(mfile.file, "rb") as fhandle:
        buf = fhandle.read(offset)

    # unpack header
    swap, magic, nformat, ncol, nrow, nlay = unpack("= i i i q q q", buf)
    nncol = ncol + 1
    nnrow = nrow + 1
    nnlay = nlay + 1

    if swap != 1 or magic != 1301:
        raise ValueError(f"Error, swap magic are {swap} {magic}, expected is 1 1301")

    # subformat processing, indicating number of bytes per datatype
    # here, 844 is native XTGeo (float64, float32, int32)
    if nformat not in (444, 844, 841, 881, 884):
        raise ValueError(f"The subformat value {nformat} is not valid")

    coordfmt, zcornfmt, actnumfmt = [int(nbyte) for nbyte in str(nformat)]

    dtype_coordsv = "float" + str(coordfmt * 8)
    dtype_zcornsv = "float" + str(zcornfmt * 8)
    dtype_actnumv = "int" + str(actnumfmt * 8)

    ncoord = nncol * nnrow * 6
    nzcorn = nncol * nnrow * nnlay * 4
    nactnum = ncol * nrow * nlay

    # read numpy arrays from file
    coordsv = xsys.npfromfile(
        mfile.file, dtype=dtype_coordsv, count=ncoord, offset=offset
    )
    newoffset = offset + ncoord * coordfmt
    zcornsv = xsys.npfromfile(
        mfile.file, dtype=dtype_zcornsv, count=nzcorn, offset=newoffset
    )
    newoffset += nzcorn * zcornfmt
    actnumsv = xsys.npfromfile(
        mfile.file, dtype=dtype_actnumv, count=nactnum, offset=newoffset
    )
    newoffset += nactnum * actnumfmt

    # read metadata which will be at position offet + nfloat*narr +13
    pos = newoffset + 13

    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)
    req = meta["_required_"]

    # meta _optional_ *may* contain xshift, xscale etc which in case must be taken
    # into account
    opt = meta.get("_optional_", None)
    if opt:
        if {"xshift", "xscale"}.issubset(opt):
            shi = opt["xshift"]
            sca = opt["xscale"]
            coordsv[0::3] = np.where(shi != 0, coordsv[0::3] + shi, coordsv[0::3])
            coordsv[0::3] = np.where(sca != 1, coordsv[0::3] * sca, coordsv[0::3])
        if {"yshift", "yscale"}.issubset(opt):
            shi = opt["yshift"]
            sca = opt["yscale"]
            coordsv[1::3] = np.where(shi != 0, coordsv[1::3] + shi, coordsv[1::3])
            coordsv[1::3] = np.where(sca != 1, coordsv[1::3] * sca, coordsv[1::3])
        if {"zshift", "zscale"}.issubset(opt):
            shi = opt["zshift"]
            sca = opt["zscale"]
            coordsv[2::3] = np.where(shi != 0, coordsv[2::3] + shi, coordsv[2::3])
            coordsv[2::3] = np.where(sca != 1, coordsv[2::3] * sca, coordsv[2::3])
            zcornsv = (zcornsv + shi) * sca

    self._coordsv = coordsv.reshape((nncol, nnrow, 6)).astype(np.float64)
    self._zcornsv = zcornsv.reshape((nncol, nnrow, nnlay, 4)).astype(np.float32)
    self._actnumsv = actnumsv.reshape((ncol, nrow, nlay)).astype(np.int32)

    reqattrs = xtgeo.MetaDataCPGeometry.REQUIRED

    for myattr in reqattrs:
        if "subgrid" in myattr:
            self.set_subgrids(reqattrs["subgrids"])
        else:
            setattr(self, "_" + myattr, req[myattr])

    self._metadata.required = self
