"""Export RegularSurface data."""

from __future__ import annotations

import io
import json
import struct
from typing import TYPE_CHECKING

import h5py
import hdf5plugin
import numpy as np

from xtgeo import _cxtgeo
from xtgeo.common.constants import UNDEF_MAP_IRAPA, UNDEF_MAP_IRAPB
from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.io._file import FileWrapper
    from xtgeo.surface.regular_surface import RegularSurface

logger = null_logger(__name__)


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


def export_irap_ascii(self: RegularSurface, mfile: FileWrapper) -> None:
    """Export to Irap RMS ascii format."""

    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPA, order="F")

    yinc = self.yinc * self.yflip

    xmax = self.xori + (self.ncol - 1) * self.xinc
    ymax = self.yori + (self.nrow - 1) * yinc

    header = (
        f"-996 {self.nrow} {self.xinc} {yinc}\n"
        f"{self.xori} {xmax} {self.yori} {ymax}\n"
        f"{self.ncol} {self.rotation} {self.xori} {self.yori}\n"
        "0  0  0  0  0  0  0\n"
    )

    def _optimal_shape(vals, start=9):
        """Optimal shape for the data.

        It seems by reverse engineering that RMS accepts only 9 or less items per line
        """
        # Check if divisible by a number
        size = len(vals)
        # Find the nearest factorization
        for i in range(start, 1, -1):
            if size % i == 0:
                return (int(size // i), int(i))  # Ensure integers are returned

        # If we can't find a perfect divisor, return a valid shape
        return (int(size), 1)

    buffer = io.StringIO()

    np.savetxt(buffer, vals.reshape(_optimal_shape(vals)), fmt="%f", delimiter=" ")
    data = buffer.getvalue()

    # Combine header and data
    buf = (header + data).encode("latin1")

    if mfile.memstream:
        mfile.file.write(buf)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(buf)

    del vals


def export_irap_binary(self: RegularSurface, mfile: FileWrapper) -> None:
    """Export to Irap RMS binary format."""

    vals = self.get_values1d(fill_value=UNDEF_MAP_IRAPB, order="F")

    yinc = self.yinc * self.yflip

    header = struct.pack(
        ">3i6f3i3f10i",  # > means big endian storage
        32,
        -996,
        self.nrow,
        self.xori,
        self.xori + self.xinc * (self.ncol - 1),
        self.yori,
        self.yori + yinc * (self.nrow - 1),
        self.xinc,
        yinc,
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
    data = bytearray(header)
    chunk_size = nchunk * 4

    # Precompute the struct.pack format for chunk size
    chunk_size_pack = struct.pack(">i", chunk_size)

    for chunk in chunks:
        chunk_data = np.array(vals[start : start + chunk], dtype=">f4").tobytes()
        data.extend(chunk_size_pack)
        data.extend(chunk_data)
        data.extend(chunk_size_pack)
        start += chunk

    if mfile.memstream:
        mfile.file.write(data)
    else:
        with open(mfile.name, "wb") as fout:
            fout.write(data)


def export_ijxyz_ascii(self: RegularSurface, mfile: FileWrapper) -> None:
    """Export to DSG IJXYZ ascii format."""

    dfr = self.get_dataframe(ij=True, order="F")  # order F since this was in cxtgeo
    ix = dfr.IX - 1  # since dataframe indexing starts at 1
    jy = dfr.JY - 1
    inlines = self.ilines[ix]
    xlines = self.xlines[jy]
    dfr["IL"] = inlines
    dfr["XL"] = xlines
    dfr = dfr[["IL", "XL", "X_UTME", "Y_UTMN", "VALUES"]]

    fmt = "%f"

    if mfile.memstream:
        buf = dfr.to_csv(sep="\t", float_format=fmt, index=False, header=False)
        buf = buf.encode("latin1")
        mfile.file.write(buf)
    else:
        dfr.to_csv(mfile.name, sep="\t", float_format=fmt, index=False, header=False)


def export_zmap_ascii(self: RegularSurface, mfile: FileWrapper) -> None:
    """Export to ZMAP ascii format (non-rotated)."""

    # zmap can only deal with non-rotated formats; hence make a copy
    # of the instance and derotate that prior to export, so that the
    # original instance is unchanged

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


def export_storm_binary(self: RegularSurface, mfile: FileWrapper) -> None:
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
        scopy.get_values1d(order="F", asmasked=False, fill_value=self.undef).astype(
            np.float64
        ),
        zmin,
        zmax,
        0,
    )
    if ier != 0:
        raise RuntimeError(f"Export to Storm binary went wrong, code is {ier}")
    del scopy

    mfile.cfclose()


def export_petromod_binary(
    self: RegularSurface, mfile: FileWrapper, pmd_dataunits: tuple[int, int]
):
    """Export to petromod binary format."""
    validunits = False
    unitd = 15
    unitz = 10
    if isinstance(pmd_dataunits, tuple) and len(pmd_dataunits) == 2:
        unitd, unitz = pmd_dataunits
        if isinstance(unitd, int) and isinstance(unitz, int):
            if unitd in PMD_DATAUNITDISTANCE and unitz in PMD_DATAUNITZ:
                validunits = True

            if unitd <= 0 or unitz <= 0:
                raise ValueError("Values for pmd_dataunits cannot be negative!")

    if not validunits:
        UserWarning(
            "Format or values for pmd_dataunits out of range: Pair should be in ranges "
            f"{PMD_DATAUNITDISTANCE} and {PMD_DATAUNITZ}"
        )

    undef = 99999

    dsc = "Content=Map,"
    dsc += f"DataUnitDistance={unitd},"
    dsc += f"DataUnitZ={unitz},"
    dsc += f"GridNoX={self.ncol},"
    dsc += f"GridNoY={self.nrow},"
    dsc += f"GridStepX={self.xinc},"
    dsc += f"GridStepY={self.yinc},"
    dsc += "MapType=GridMap,"
    dsc += f"OriginX={self.xori},"
    dsc += f"OriginY={self.yori},"
    dsc += f"RotationAngle={self.rotation},"
    dsc += f"RotationOriginX={self.xori},"
    dsc += f"RotationOriginY={self.yori},"
    dsc += f"Undefined={undef},"
    dsc += "Version=1.0"

    values = np.ma.filled(self.values1d, fill_value=undef)

    _cxtgeo.surf_export_petromod_bin(
        mfile.get_cfhandle(),
        dsc,
        values.astype(np.float64),
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
