# -*- coding: utf-8 -*-
"""Export RegularSurface data."""

import json

# pylint: disable=protected-access
# import hashlib
import struct

import h5py
import hdf5plugin
import numpy as np
import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # pylint: disable=import-error
from xtgeo.common import XTGeoDialog
from xtgeo.common.constants import UNDEF_MAP_IRAPA, UNDEF_MAP_IRAPB

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


def export_irap_ascii(self, mfile, engine="cxtgeo"):
    """Export to Irap RMS ascii format."""
    if mfile.memstream is True or engine == "python":
        _export_irap_ascii_purepy(self, mfile)
    else:
        _export_irap_ascii(self, mfile)


def _export_irap_ascii_purepy(self, mfile):
    """Export to Irap RMS ascii using pure python, slower? but safer for memstreams."""
    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPA, order="F")

    xmax = self.xori + (self.ncol - 1) * self.xinc
    ymax = self.yori + (self.nrow - 1) * self.yinc

    buf = "-996 {} {} {}\n".format(self.nrow, self.xinc, self.yinc)
    buf += "{} {} {} {}\n".format(self.xori, xmax, self.yori, ymax)
    buf += "{} {} {} {}\n".format(self.ncol, self.rotation, self.xori, self.yori)
    buf += "0  0  0  0  0  0  0\n"
    vals = vals.astype("str").tolist()
    nrow = 0
    for val in vals:
        buf += val
        nrow += 1
        if nrow == 6:
            buf += "\n"
            nrow = 0
        else:
            buf += " "

    if nrow != 0:
        buf += "\n"

    # convert buffer to ascii
    buf = buf.encode("latin1")

    if mfile.memstream:
        mfile.file.write(buf)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(buf)

    del vals


def _export_irap_ascii(self, mfile):
    """Export to Irap RMS ascii format using cxtgeo."""
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
        raise RuntimeError("Export to Irap Ascii went wrong, code is {}".format(ier))

    del vals

    mfile.cfclose()


def export_irap_binary(self, mfile, engine="cxtgeo"):
    """Export to Irap RMS binary format.

    Note that mfile can also a be a BytesIO instance
    """
    if mfile.memstream or engine == "python":
        _export_irap_binary_python(self, mfile)
    else:
        _export_irap_binary_cxtgeo(self, mfile)


def _export_irap_binary_python(self, mfile):
    """Export to Irap RMS binary format but use python only.

    This is approx 2-5 times slower than the C method, but may a be more robust in cases
    with BytesIO.
    """
    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPB, order="F")

    ap = struct.pack(
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

    # export to Irap binary in ncol chunks (the only chunk size accepted by RMS)
    nchunk = self.ncol
    chunks = [nchunk] * (inum // nchunk)
    if (inum % nchunk) > 0:
        chunks.append(inum % nchunk)
    start = 0
    for chunk in chunks:
        ap += struct.pack(">i", chunk * 4)
        ap += struct.pack(">{:d}f".format(chunk), *vals[start : start + chunk])
        ap += struct.pack(">i", chunk * 4)
        start += chunk

    if mfile.memstream:
        mfile.file.write(ap)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(ap)


def _export_irap_binary_cxtgeo(self, mfile):
    """Export to Irap binary using C backend.

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


def export_zmap_ascii(self, mfile, engine="cxtgeo"):
    """Export to ZMAP ascii format (non-rotated)."""

    # zmap can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

    if mfile.memstream or engine == "python":
        _export_zmap_ascii_purepy(self, mfile)
    else:
        _export_zmap_ascii(self, mfile)


def _export_zmap_ascii_purepy(self, mfile):
    """Export to ZMAP ascii format (non-rotated), pure python for memstreams"""

    scopy = self.copy()

    undef = -99999.0
    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    yinc = scopy._yinc * scopy._yflip

    vals = scopy.get_values1d(order="C", asmasked=False, fill_value=undef)

    xmax = scopy.xori + (scopy.ncol - 1) * scopy.xinc
    ymax = scopy.yori + (scopy.nrow - 1) * yinc

    fcode = 8
    if scopy.values.min() > -10 and scopy.values.max() < 10:
        fcode = 4

    nfrow = scopy.nrow if scopy.nrow < 5 else 5

    buf = "! Export from XTGeo (python engine)\n"
    buf += f"@ GRIDFILE, GRID, {nfrow}\n"
    buf += f"20, {undef}, , {fcode}, 1\n"
    buf += f"{scopy.nrow}, {scopy.ncol}, {scopy.xori}, {xmax}, {scopy.yori}, {ymax}\n"

    buf += "0.0, 0.0, 0.0\n"
    buf += "@\n"

    vals = vals.tolist()
    ncol = 0
    for icol in range(scopy.ncol):
        for jrow in range(scopy.nrow - 1, -1, -1):
            ic = icol * scopy.nrow + jrow
            buf += f" {vals[ic]:19.{fcode}f}"
            ncol += 1
            if ncol == 5 or jrow == 0:
                buf += "\n"
                ncol = 0

    # convert buffer to ascii
    buf = buf.encode("latin1")

    if mfile.memstream:
        mfile.file.write(buf)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(buf)

    del vals


def _export_zmap_ascii(self, mfile):
    """Export to ZMAP ascii format (non-rotated)."""

    scopy = self.copy()

    if abs(scopy.rotation) > 1.0e-20:
        scopy.unrotate()

    zmin = scopy.values.min()
    zmax = scopy.values.max()

    yinc = scopy._yinc * scopy._yflip

    vals = scopy.get_values1d(order="C", asmasked=False, fill_value=xtgeo.UNDEF)

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
    unitd = 15
    unitz = 10
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
        mfile.get_cfhandle(),
        dsc,
        values,
    )

    mfile.cfclose()


def export_xtgregsurf(self, mfile):
    """Export to experimental xtgregsurf format, python version."""
    logger.info("Export to xtgregsurf format...")

    self.metadata.required = self

    vals = self.get_values1d(fill_value=self._undef, order="C").astype(np.float32)

    prevalues = (1, 1101, 4, self.ncol, self.nrow)
    mystruct = struct.Struct("= i i i q q")
    hdr = mystruct.pack(*prevalues)

    meta = self.metadata.get_metadata()

    jmeta = json.dumps(meta).encode()

    with open(mfile.name, "wb") as fout:
        fout.write(hdr)

    with open(mfile.name, "ab") as fout:
        vals.tofile(fout)

    with open(mfile.name, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(mfile.name, "ab") as fout:
        fout.write(jmeta)

    logger.info("Export to xtgregsurf format... done!")


def export_hdf5_regsurf(self, mfile, compression="lzf", dtype="float32"):
    """Export to experimental hdf5 format."""
    logger.info("Export to hdf5 format...")

    self.metadata.required = self

    meta = self.metadata.get_metadata()
    jmeta = json.dumps(meta).encode()

    if compression and compression == "blosc":
        compression = hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )

    if dtype not in ("float32", "float64", np.float32, np.float64):
        raise ValueError("Wrong dtype input, must be 'float32' or 'float64'")

    with h5py.File(mfile.name, "w") as fh5:
        grp = fh5.create_group("RegularSurface")
        grp.create_dataset(
            "values",
            data=np.ma.filled(self.values, fill_value=self.undef).astype(dtype),
            compression=compression,
            chunks=True,
        )
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = "xtgeo"
        grp.attrs["format-idcode"] = 1101

    logger.info("Export to hdf5 format... done!")
