"""GridProperty (not GridProperies) export functions."""
import struct
import json

import numpy as np
import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def to_file(self, pfile, fformat="roff", name=None, append=False, dtype=None, fmt=None):
    """Export the grid property to file."""
    logger.info("Export property to file %s as %s", pfile, fformat)

    fobj = xtgeo._XTGeoFile(pfile, mode="rb")
    fobj.check_folder(raiseerror=OSError)

    if name is None:
        name = self.name

    if "roff" in fformat:

        binary = True
        if "asc" in fformat:
            binary = False

        # for later usage
        append = False
        last = True

        export_roff(self, fobj.name, name, append=append, last=last, binary=binary)

    elif fformat == "grdecl":
        export_grdecl(
            self, fobj.name, name, append=append, binary=False, dtype=dtype, fmt=fmt
        )

    elif fformat == "bgrdecl":
        export_grdecl(self, fobj.name, name, append=append, binary=True, dtype=dtype)

    elif fformat == "xtgcpprop":
        export_xtgcpprop(self, fobj.name)

    else:
        raise ValueError("Cannot export, invalid fformat: {}".format(fformat))


# Export ascii or binary ROFF format


def export_roff(self, pfile, name, append=False, last=True, binary=True):

    logger.info("Export roff to %s", pfile)
    if self._isdiscrete:
        _export_roff_discrete(
            self, pfile, name, append=append, last=last, binary=binary
        )
    else:
        _export_roff_continuous(
            self, pfile, name, append=append, last=last, binary=binary
        )


def _export_roff_discrete(self, pfile, name, append=False, last=True, binary=True):

    carray = _gridprop_lowlevel.update_carray(self, undef=-999)

    ptr_idum = _cxtgeo.new_intpointer()
    ptr_ddum = _cxtgeo.new_doublepointer()

    # codes:
    ptr_codes = _cxtgeo.new_intarray(256)
    ncodes = self.ncodes
    codenames = ""
    logger.info("Keys: %s", self.codes.keys())
    for inum, ckey in enumerate(sorted(self.codes.keys())):
        if ckey is not None:
            codenames += str(self.codes[ckey])
            codenames += "|"
            _cxtgeo.intarray_setitem(ptr_codes, inum, int(ckey))
        else:
            logger.warning("For some odd reason, None is a key. Check!")

    mode = 0
    if not binary:
        mode = 1

    if not append:
        _cxtgeo.grd3d_export_roff_pstart(
            mode, self._ncol, self._nrow, self._nlay, pfile
        )

    nsub = 0
    isub_to_export = 0
    _cxtgeo.grd3d_export_roff_prop(
        mode,
        self._ncol,
        self._nrow,
        self._nlay,
        nsub,
        isub_to_export,
        ptr_idum,
        name,
        "int",
        carray,
        ptr_ddum,
        ncodes,
        codenames,
        ptr_codes,
        pfile,
    )

    if last:
        _cxtgeo.grd3d_export_roff_end(mode, pfile)

    _gridprop_lowlevel.delete_carray(self, carray)


def _export_roff_continuous(self, pfile, name, append=False, last=True, binary=True):

    carray = _gridprop_lowlevel.update_carray(self, undef=-999.0)

    ptr_idum = _cxtgeo.new_intpointer()

    mode = 0
    if not binary:
        mode = 1

    if not append:
        _cxtgeo.grd3d_export_roff_pstart(
            mode, self._ncol, self._nrow, self._nlay, pfile
        )

    # now the actual data
    nsub = 0
    isub_to_export = 0

    _cxtgeo.grd3d_export_roff_prop(
        mode,
        self._ncol,
        self._nrow,
        self._nlay,
        nsub,
        isub_to_export,
        ptr_idum,
        name,
        "double",
        ptr_idum,
        carray,
        0,
        "",
        ptr_idum,
        pfile,
    )

    if last:
        _cxtgeo.grd3d_export_roff_end(mode, pfile)

    _gridprop_lowlevel.delete_carray(self, carray)


# Export ascii or binary GRDECL


def export_grdecl(self, pfile, name, append=False, binary=False, dtype=None, fmt=None):

    logger.info("Exporting %s to file %s, GRDECL format", name, pfile)

    if dtype is None:
        if self._isdiscrete:
            dtype = "int32"
        else:
            dtype = "float32"

    carray = _gridprop_lowlevel.update_carray(self, dtype=dtype)

    usefmt = " %13.4f"
    if fmt is not None:
        usefmt = " {}".format(fmt)

    iarr = _cxtgeo.new_intpointer()
    farr = _cxtgeo.new_floatpointer()
    darr = _cxtgeo.new_doublepointer()

    if "double" in str(carray):
        ptype = 3
        darr = carray
        if fmt is None:
            usefmt = " %e"

    elif "float" in str(carray):
        ptype = 2
        farr = carray
        if fmt is None:
            usefmt = " %e"

    else:
        ptype = 1
        iarr = carray
        if fmt is None:
            usefmt = " %d"

    mode = 0
    if not binary:
        mode = 1

    appendmode = 0
    if append:
        appendmode = 1

    _cxtgeo.grd3d_export_grdeclprop2(
        self._ncol,
        self._nrow,
        self._nlay,
        ptype,
        iarr,
        farr,
        darr,
        name,
        usefmt,
        pfile,
        mode,
        appendmode,
    )

    _gridprop_lowlevel.delete_carray(self, carray)


def export_xtgcpprop(self, mfile):
    """Export to experimental xtgcpproperty format, python version."""
    logger.info("Export as xtgcpprop...")
    self._metadata.required = self

    magic = 1351
    if self.isdiscrete:
        magic = 1352

    prevalues = (1, magic, 4, self.ncol, self.nrow, self.nlay)
    mystruct = struct.Struct("= i i i q q q")
    pre = mystruct.pack(*prevalues)

    meta = self.metadata.get_metadata()

    with open(mfile, "wb") as fout:
        fout.write(pre)

    with open(mfile, "ab") as fout:
        vv = self.get_npvalues1d(fill_value=self.undef)
        vv.astype(np.float32).tofile(fout)

    with open(mfile, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(mfile, "ab") as fout:
        fout.write(json.dumps(meta).encode())

    logger.info("Export as xtgcpprop... done")
