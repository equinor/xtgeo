"""GridProperty (not GridProperies) export functions."""
import json
import struct

import ecl_data_io as eclio
import numpy as np
import roffio

import xtgeo
from xtgeo.common import XTGeoDialog

from ._roff_parameter import RoffParameter

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def to_file(self, pfile, fformat="roff", name=None, append=False, dtype=None, fmt=None):
    """Export the grid property to file."""
    logger.info("Export property to file %s as %s", pfile, fformat)

    fobj = xtgeo._XTGeoFile(pfile, mode="rb")
    if not fobj.memstream:
        fobj.check_folder(raiseerror=OSError)

    if name is None:
        name = self.name

    if "roff" in fformat:

        binary = True
        if "asc" in fformat:
            binary = False

        if append:
            logger.warning(
                "Append is not implemented for roff format, defaulting to write."
            )

        export_roff(self, fobj.name, name, binary=binary)

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


def export_roff(self, pfile, name, binary=True):

    logger.info("Export roff to %s", pfile)
    roff_param = RoffParameter.from_xtgeo_grid_property(self)
    roff_param.name = name
    roff_format = roffio.Format.ASCII
    if binary:
        roff_format = roffio.Format.BINARY

    roff_param.to_file(pfile, roff_format)


def export_grdecl(self, pfile, name, append=False, binary=False, dtype=None, fmt=None):
    """Export ascii or binary GRDECL"""
    vals = self.values.ravel(order="F")
    if np.ma.isMaskedArray(vals):
        vals = np.ma.filled(vals, self.undef)

    if binary:
        mode = "wb"
        if append:
            mode = "ab"

        with open(pfile, mode) as fh:
            if dtype is not None:
                eclio.write(fh, [(name.ljust(8), vals.astype(dtype))])
            elif self.isdiscrete:
                eclio.write(fh, [(name.ljust(8), vals.astype(np.int32))])
            else:
                eclio.write(fh, [(name.ljust(8), vals.astype(np.float32))])

    else:
        mode = "w"
        if append:
            mode = "a"
        with open(pfile, mode) as fh:
            fh.write(name)
            fh.write("\n")
            for i, v in enumerate(vals):
                fh.write(" ")
                if fmt:
                    fh.write(fmt % v)
                elif self.isdiscrete:
                    fh.write(str(v))
                else:
                    fh.write("{:3e}".format(v))
                if i % 6 == 5:
                    fh.write("\n")

            fh.write(" /\n")


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
