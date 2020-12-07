"""GridProperty import function of xtgcpprop format."""

from struct import unpack
import json
import numpy as np

import xtgeo
import xtgeo.common.sys as xsys

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_xtgcpprop(self, mfile):
    """Using pure python for experimental xtgcpprop import."""
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
    vals = xsys.npfromfile(mfile.file, dtype=dtype, count=narr, offset=offset)

    # read metadata which will be at position offet + nfloat*narr +13
    pos = offset + nbyte * narr + 13

    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta)
    req = meta["_required_"]

    reqattrs = xtgeo.MetaDataCPProperty.REQUIRED

    for myattr in reqattrs:
        if "discrete" in myattr:
            self._isdiscrete = req[myattr]
        else:
            setattr(self, "_" + myattr, req[myattr])

    self._values = np.ma.masked_equal(vals.reshape(ncol, nrow, nlay), self._undef)

    self._metadata.required = self
