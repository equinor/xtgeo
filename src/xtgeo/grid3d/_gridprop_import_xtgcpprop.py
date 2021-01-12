"""GridProperty import function of xtgcpprop format."""

from struct import unpack
import json
from collections import OrderedDict

import numpy as np

import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_xtgcpprop(self, mfile, ijrange=None, zerobased=False):
    """Using pure python for experimental xtgcpprop import.

    Args:
        self (obj): instance
        mfile (_XTGeoFile): Input file reference
        ijrange (list-like): List or tuple with 4 members [i_from, i_to, j_from, j_to]
            where cell indices are zero based (starts with 0)
        zerobased (bool): If ijrange basis is zero or one.

    """
    #
    offset = 36
    with open(mfile.file, "rb") as fhandle:
        buf = fhandle.read(offset)

    # unpack header
    swap, magic, nbyte, ncol, nrow, nlay = unpack("= i i i q q q", buf)

    if swap != 1 or magic not in (1351, 1352):
        raise ValueError("Invalid file format (wrong swap id or magic number).")

    if magic == 1351:
        dtype = np.float32 if nbyte == 4 else np.float64
    else:
        dtype = "int" + str(nbyte * 8)

    vals = None
    narr = ncol * nrow * nlay

    ncolnew = nrownew = 0

    if ijrange:
        vals, ncolnew, nrownew = _import_xtgcpprop_partial(
            mfile, nbyte, dtype, offset, ijrange, zerobased, ncol, nrow, nlay
        )

    else:
        vals = xsys.npfromfile(mfile.file, dtype=dtype, count=narr, offset=offset)

    # read metadata which will be at position offet + nfloat*narr +13
    pos = offset + nbyte * narr + 13

    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)
    req = meta["_required_"]

    reqattrs = xtgeo.MetaDataCPProperty.REQUIRED

    for myattr in reqattrs:
        if "discrete" in myattr:
            self._isdiscrete = req[myattr]
        else:
            setattr(self, "_" + myattr, req[myattr])

    if ijrange:
        self._ncol = ncolnew
        self._nrow = nrownew

    self._values = np.ma.masked_equal(
        vals.reshape(self._ncol, self._nrow, self._nlay), self._undef
    )

    self._metadata.required = self


def _import_xtgcpprop_partial(
    mfile, nbyte, dtype, offset, ijrange, zerobased, ncol, nrow, nlay
):
    """Partial import of a property."""
    i1, i2, j1, j2 = ijrange
    if not zerobased:
        i1 -= 1
        i2 -= 1
        j1 -= 1
        j2 -= 1

    ncolnew = i2 - i1 + 1
    nrownew = j2 - j1 + 1

    if ncolnew < 1 or ncolnew > ncol or nrownew < 1 or nrownew > nrow:
        raise ValueError("The ijrange spesification is invalid.")

    vals = np.zeros(ncolnew * nrownew * nlay, dtype=dtype)

    for newnum, inum in enumerate(range(i1, i2 + 1)):
        newpos = offset + (inum * nrow * nlay + j1 * nlay) * nbyte
        ncount = nrownew * nlay
        xvals = xsys.npfromfile(mfile.file, dtype=dtype, count=ncount, offset=newpos)
        vals[newnum * ncount : newnum * ncount + ncount] = xvals

    return vals, ncolnew, nrownew
