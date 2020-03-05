# -*- coding: utf-8 -*-
"""Export RegularSurface data."""

# pylint: disable=protected-access
from __future__ import division, absolute_import
from __future__ import print_function
from struct import pack
import numpy as np

import xtgeo
from xtgeo.common.constants import UNDEF_MAP_IRAPB
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # pylint: disable=import-error
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


DEBUG = 0

if DEBUG < 0:
    DEBUG = 0


def export_irap_ascii(self, mfile):
    """Export to Irap RMS ascii format."""

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    zmin = self.values.min()
    zmax = self.values.max()

    vals = self.get_values1d(fill_value=xtgeo.UNDEF)
    logger.debug("SHAPE %s %s", vals.shape, vals.dtype)

    ier = _cxtgeo.surf_export_irap_ascii(
        fout.fhandle,
        self._ncol,
        self._nrow,
        self._xori,
        self._yori,
        self._xinc,
        self._yflip * self._yinc,
        self._rotation,
        vals,
        zmin,
        zmax,
        0,
    )
    if ier != 0:
        raise RuntimeError("Export to Irap Ascii went wrong, " "code is {}".format(ier))

    del vals

    fout.close()


def export_irap_binary(self, mfile, engine="cxtgeo", bstream=False):
    """Export to Irap RMS binary format.

    Note that mfile can also a be a BytesIO instance
    """

    if engine == "cxtgeo":
        _export_irap_binary_cxtgeo(self, mfile)
    else:
        _export_irap_binary_python(self, mfile, bstream=bstream)


def _export_irap_binary_cxtgeo(self, mfile):
    """Export to Irap RMS binary format."""

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPB, order="F")
    ier = _cxtgeo.surf_export_irap_bin(
        fout.fhandle,
        self._ncol,
        self._nrow,
        self._xori,
        self._yori,
        self._xinc,
        self._yflip * self._yinc,
        self._rotation,
        vals,
        0,
    )

    if ier != 0:
        raise RuntimeError("Export to Irap Binary went wrong, code is {}".format(ier))

    fout.close()


def _export_irap_binary_python(self, mfile, bstream=False):
    """Export to Irap RMS binary format but use python only.

    This is approx 2-5 times slower than the C method, but may a be more robust in cases
    with BytesIO.
    """

    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPB, order="F")

    ap = pack(
        ">3i6f3i3f10i",  # > means big endian storage
        32,
        -996,
        self.nrow,
        self.xori,
        self.xori + self.xinc * (self.ncol - 1),
        self.yori,
        self.yori + self.yinc * (self.nrow - 1),
        self.xinc,
        self.yinc,
        32,
        16,
        self.ncol,
        self.rotation,
        self.xori,
        self.yori,
        16,
        28,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        28,
    )
    inum = self.nrow * self.ncol

    # export to Irap binary in ncol chunks (only chunk size accepted by RMS)
    nchunk = self.ncol
    chunks = [nchunk] * (inum // nchunk)
    if (inum % nchunk) > 0:
        chunks.append(inum % nchunk)
    start = 0
    for chunk in chunks:
        ap += pack(">i", chunk * 4)
        ap += pack(">{:d}f".format(chunk), *vals[start : start + chunk])
        ap += pack(">i", chunk * 4)
        start += chunk

    if bstream:
        mfile.write(ap)
    else:
        with open(mfile, "wb") as fout:
            fout.write(ap)


def export_ijxyz_ascii(self, mfile):
    """Export to DSG IJXYZ ascii format."""

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    vals = self.get_values1d(fill_value=xtgeo.UNDEF)
    ier = _cxtgeo.surf_export_ijxyz(
        fout.fhandle,
        self._ncol,
        self._nrow,
        self._xori,
        self._yori,
        self._xinc,
        self._yinc,
        self._rotation,
        self._yflip,
        self._ilines,
        self._xlines,
        vals,
        0,
    )

    if ier != 0:
        raise RuntimeError(
            "Export to IJXYZ format went wrong, " "code is {}".format(ier)
        )

    fout.close()


def export_zmap_ascii(self, mfile):
    """Export to ZMAP ascii format (non-rotated)."""

    # zmap can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    scopy = self.copy()

    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    zmin = scopy.values.min()
    zmax = scopy.values.max()

    yinc = scopy._yinc * scopy._yflip

    vals = scopy.get_values1d(order="F", asmasked=False, fill_value=xtgeo.UNDEF)

    ier = _cxtgeo.surf_export_zmap_ascii(
        fout.fhandle,
        scopy._ncol,
        scopy._nrow,
        scopy._xori,
        scopy._yori,
        scopy._xinc,
        yinc,
        vals,
        zmin,
        zmax,
        0,
    )
    if ier != 0:
        raise RuntimeError("Export to ZMAP Ascii went wrong, " "code is {}".format(ier))
    del scopy

    fout.close()


def export_storm_binary(self, mfile):
    """Export to Storm binary format (non-rotated)."""

    # storm can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    scopy = self.copy()

    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    zmin = scopy.values.min()
    zmax = scopy.values.max()

    yinc = scopy._yinc * scopy._yflip

    ier = _cxtgeo.surf_export_storm_bin(
        fout.fhandle,
        scopy._ncol,
        scopy._nrow,
        scopy._xori,
        scopy._yori,
        scopy._xinc,
        yinc,
        scopy.get_zval(),
        zmin,
        zmax,
        0,
    )
    if ier != 0:
        raise RuntimeError(
            "Export to Storm binary went wrong, " "code is {}".format(ier)
        )
    del scopy

    fout.close()


def export_petromod_binary(self, mfile):
    """Export to petromod binary format."""

    undef = 99999
    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    dsc = "Content=Map,"
    dsc += "DataUnitDistance=15,"
    dsc += "DataUnitZ=10,"
    dsc += "GridNoX={},".format(self.ncol)
    dsc += "GridNoY={},".format(self.nrow)
    dsc += "GridStepX={},".format(self.xinc)
    dsc += "GridStepY={},".format(self.yinc)
    dsc += "MapType=GridMap,"
    dsc += "OriginX={},".format(self.xori)
    dsc += "OriginY={},".format(self.yori)
    dsc += "RotationAngle={},".format(self.rotation)
    dsc += "RotationOriginX={},".format(self.xori)
    dsc += "RotationOriginY={},".format(self.yori)
    dsc += "Undefined={},".format(undef)
    dsc += "Version=1.0"

    values = np.ma.filled(self.values1d, fill_value=undef)

    _cxtgeo.surf_export_petromod_bin(
        fout.fhandle, dsc, values,
    )

    fout.close()
