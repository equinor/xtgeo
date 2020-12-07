"""GridProperty (not GridProperies) import functions"""

# from ._gridprop_import_eclrun import import_eclbinary
#  --> INIT, UNRST
#
# from ._gridprop_import_grdecl import import_grdecl_prop, import_bgrdecl_prop
#  --> ASCII and BINARY GRDECL format
# from ._gridprop_import_roff import import_roff
#  --> BINARY ROFF format

from __future__ import print_function, absolute_import

import os
from struct import unpack
import json

import numpy as np

import xtgeo
import xtgeo.common.sys as xsys

from ._gridprop_import_eclrun import import_eclbinary as impeclbin
from ._gridprop_import_grdecl import import_grdecl_prop, import_bgrdecl_prop
from ._gridprop_import_roff import import_roff

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)


def from_file(
    self,
    pfile,
    fformat=None,
    name="unknown",
    grid=None,
    date=None,
    fracture=False,
    _roffapiv=1,
):  # _roffapiv for devel.
    """Import grid property from file, and makes an instance of this."""
    # it may be that pfile already is an open file; hence a filehandle
    # instead. Check for this, and skip actions if so
    if not isinstance(pfile, xtgeo._XTGeoFile):
        raise RuntimeError("Internal error, pfile is not a _XTGeoFile instance")

    fformat = _chk_file(self, pfile.name, fformat)

    if fformat == "roff":
        logger.info("Importing ROFF...")
        import_roff(self, pfile, name, grid=grid, _roffapiv=_roffapiv)

    elif fformat.lower() == "init":
        impeclbin(
            self, pfile, name=name, etype=1, date=None, grid=grid, fracture=fracture
        )

    elif fformat.lower() == "unrst":
        if date is None:
            raise ValueError("Restart file, but no date is given")

        if isinstance(date, str):
            if "-" in date:
                date = int(date.replace("-", ""))
            elif date == "first":
                date = 0
            elif date == "last":
                date = 9
            else:
                date = int(date)

        if not isinstance(date, int):
            raise RuntimeError("Date is not int format")

        impeclbin(
            self, pfile, name=name, etype=5, date=date, grid=grid, fracture=fracture
        )

    elif fformat.lower() == "grdecl":
        import_grdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat.lower() == "bgrdecl":
        import_bgrdecl_prop(self, pfile, name=name, grid=grid)

    elif fformat.lower() == "xtgcpprop":
        import_xtgcpprop(self, pfile)

    else:
        logger.warning("Invalid file format")
        raise ValueError("Invalid file format")

    # if grid, then append this gridprop to the current grid object

    # ###################################TMP skipped""
    # if grid:
    #     grid.append_prop(self)

    return self


def _chk_file(self, pfile, fformat):

    self._filesrc = pfile

    if os.path.isfile(pfile):
        logger.debug("File %s exists OK", pfile)
    else:
        raise OSError("No such file: {}".format(pfile))

    # work on file extension
    _froot, fext = os.path.splitext(pfile)
    if fformat is None or fformat == "guess":
        if not fext:
            raise ValueError("File extension missing. STOP")

        fformat = fext.lower().replace(".", "")

    logger.debug("File name to be used is %s", pfile)
    logger.debug("File format is %s", fformat)

    return fformat


def import_xtgcpprop(self, mfile):
    """Using pure python for experimental import."""
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
