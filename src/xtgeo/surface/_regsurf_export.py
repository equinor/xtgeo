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

PMD_DATAUNITDISTANCE = {
    15: "meter",
    16: "km",
    17: "feet",
    18: "yard",
    19: "mile",
    221: "global",
}

PMD_DATAUNITZ = {
    10: "10",
    31: "31",
    44: "44",
    300: "300",
}


def export_irap_ascii(self, mfile):
    """Export to Irap RMS ascii format."""

    vals = self.get_values1d(fill_value=xtgeo.UNDEF)

    ier = _cxtgeo.surf_export_irap_ascii(
        mfile.get_cfhandle(),
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
        raise RuntimeError("Export to Irap Ascii went wrong, " "code is {}".format(ier))

    del vals

    mfile.cfclose()


def export_irap_binary(self, mfile, engine="cxtgeo", bstream=False):
    """Export to Irap RMS binary format.

    Note that mfile can also a be a BytesIO instance
    """

    if engine == "cxtgeo":
        _export_irap_binary_cxtgeo(self, mfile)
    elif engine == "cxtgeotest":
        _export_irap_binary_cxtgeotest(self, mfile)
    else:
        _export_irap_binary_python(self, mfile, bstream=bstream)


def _export_irap_binary_cxtgeo(self, mfile):
    """Export to Irap binary using C backend

    Args:
        mfile (_XTGeoFile): xtgeo file instance

    Raises:
        RuntimeError: Export to Irap Binary went wrong...
    """

    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPB, order="F")
    ier = _cxtgeo.surf_export_irap_bin(
        mfile.get_cfhandle(),
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
        mfile.cfclose(strict=False)  # strict False as C routine may have closed
        raise RuntimeError("Export to Irap Binary went wrong, code is {}".format(ier))

    mfile.cfclose()


def _export_irap_binary_cxtgeotest(self, mfile):
    """Export to Irap RMS binary format. TEST SWIG FLAT"""

    print(self.values.mask.astype(np.uint8).mean())

    ier = _cxtgeo.surf_export_irap_bin_test(
        mfile.get_cfhandle(),
        self._ncol,
        self._nrow,
        self._xori,
        self._yori,
        self._xinc,
        self._yflip * self._yinc,
        self._rotation,
        self.values.data,
        self.values.mask,
    )

    if ier != 0:
        raise RuntimeError("Export to Irap Binary went wrong, code is {}".format(ier))

    mfile.cfclose()


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
        mfile.file.write(ap)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(ap)


def export_ijxyz_ascii(self, mfile):
    """Export to DSG IJXYZ ascii format."""

    vals = self.get_values1d(fill_value=xtgeo.UNDEF)
    ier = _cxtgeo.surf_export_ijxyz(
        mfile.get_cfhandle(),
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

    mfile.cfclose()


def export_zmap_ascii(self, mfile):
    """Export to ZMAP ascii format (non-rotated)."""

    # zmap can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

    scopy = self.copy()

    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    zmin = scopy.values.min()
    zmax = scopy.values.max()

    yinc = scopy._yinc * scopy._yflip

    vals = scopy.get_values1d(order="F", asmasked=False, fill_value=xtgeo.UNDEF)

    ier = _cxtgeo.surf_export_zmap_ascii(
        mfile.get_cfhandle(),
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

    mfile.cfclose()


def export_storm_binary(self, mfile):
    """Export to Storm binary format (non-rotated)."""

    # storm can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

    scopy = self.copy()

    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    zmin = scopy.values.min()
    zmax = scopy.values.max()

    yinc = scopy._yinc * scopy._yflip

    ier = _cxtgeo.surf_export_storm_bin(
        mfile.get_cfhandle(),
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

    mfile.cfclose()


def export_petromod_binary(self, mfile, pmd_dataunits):
    """Export to petromod binary format."""

    validunits = False
    if isinstance(pmd_dataunits, tuple) and len(pmd_dataunits) == 2:
        unitd, unitz = pmd_dataunits
        if isinstance(unitd, int) and isinstance(unitz, int):
            if unitd in PMD_DATAUNITDISTANCE.keys() and unitz in PMD_DATAUNITZ.keys():
                validunits = True

            if unitd <= 0 or unitz <= 0:
                raise ValueError("Values for pmd_dataunits cannot be negative!")

    if not validunits:
        UserWarning(
            "Format or values for pmd_dataunits out of range: Pair should be in ranges "
            "{} and {}".format(PMD_DATAUNITDISTANCE, PMD_DATAUNITZ)
        )

    undef = 99999

    dsc = "Content=Map,"
    dsc += "DataUnitDistance={},".format(unitd)
    dsc += "DataUnitZ={},".format(unitz)
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
        mfile.get_cfhandle(), dsc, values,
    )

    mfile.cfclose()
