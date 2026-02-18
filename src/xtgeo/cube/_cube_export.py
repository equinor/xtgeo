"""Export Cube data via SegyIO library or XTGeo CLIB."""

from __future__ import annotations

import io
import json
import math
import struct
from typing import TYPE_CHECKING

import numpy as np
import segyio

from xtgeo import _cxtgeo
from xtgeo.common import XTGeoDialog, null_logger

logger = null_logger(__name__)
xtg = XTGeoDialog()

if TYPE_CHECKING:
    from xtgeo import Cube


def export_segy(cube: Cube, sfile: str) -> None:
    """Export on SEGY using segyio library.

    Args:
        cube (:class:`xtgeo.cube.Cube`): The instance
        sfile (str): File name to export to.

    Raises:
        TypeError: If ``sfile`` is an in-memory stream (e.g. ``BytesIO``).
    """
    if isinstance(sfile, (io.StringIO, io.BytesIO)):
        raise TypeError(
            "SEGY export requires a filesystem path; in-memory streams are not "
            "supported."
        )

    logger.debug("Exporting segy format using segyio")
    cvalues = cube.values

    spec = segyio.spec()
    spec.sorting = 2  # inline
    spec.format = 5  # ieee floats
    spec.samples = np.arange(cube.nlay) * cube.zinc + cube.zori
    spec.ilines = np.array(cube._ilines)
    spec.xlines = np.array(cube._xlines)

    dt_us = int(round(cube.zinc * 1000))
    delrt = int(round(cube.zori))

    rotation_rad = math.radians(cube.rotation)
    cos_rot = math.cos(rotation_rad)
    sin_rot = math.sin(rotation_rad)

    coord_scalar = -100  # divide stored values by 100 to get real value

    with segyio.create(sfile, spec) as f:
        tr = 0
        for il_idx, il in enumerate(spec.ilines):
            for xl_idx, xl in enumerate(spec.xlines):
                dx = il_idx * cube.xinc
                dy = xl_idx * cube.yinc * cube.yflip

                # ij to xy
                x = cube.xori + dx * cos_rot - dy * sin_rot
                y = cube.yori + dx * sin_rot + dy * cos_rot

                f.header[tr] = {
                    segyio.TraceField.INLINE_3D: int(il),
                    segyio.TraceField.CROSSLINE_3D: int(xl),
                    segyio.TraceField.CDP_X: int(round(x * 100)),
                    segyio.TraceField.CDP_Y: int(round(y * 100)),
                    segyio.TraceField.SourceGroupScalar: coord_scalar,
                    segyio.TraceField.TRACE_SAMPLE_INTERVAL: dt_us,
                    segyio.TraceField.TRACE_SAMPLE_COUNT: cube.nlay,
                    segyio.TraceField.DelayRecordingTime: delrt,
                    segyio.TraceField.TraceIdentificationCode: int(
                        cube._traceidcodes[il_idx, xl_idx]
                    ),
                }
                f.trace[tr] = cvalues[il_idx, xl_idx, :]
                tr += 1

        f.bin[segyio.BinField.Interval] = dt_us
        f.bin[segyio.BinField.Samples] = cube.nlay
        f.bin[segyio.BinField.SortingCode] = 4  # trace sorting from C: needed?
        # TODO: Make this read from cube._measurement (or something)
        f.bin[segyio.BinField.MeasurementSystem] = 1

    with open(sfile, "r+b") as stream:
        stream.seek(3200)
        header = bytearray(stream.read(400))
        if len(header) == 400:
            # SEG-Y rev1 unassigned ranges in binary header.
            header[60:300] = b"\x00" * 240
            header[306:400] = b"\x00" * 94
            stream.seek(3200)
            stream.write(header)


def export_rmsreg(self, sfile):
    """Export on RMS regular format."""

    logger.debug("Export to RMS regular format...")
    values1d = self.values.reshape(-1)

    status = _cxtgeo.cube_export_rmsregular(
        self.ncol,
        self.nrow,
        self.nlay,
        self.xori,
        self.yori,
        self.zori,
        self.xinc,
        self.yinc * self.yflip,
        self.zinc,
        self.rotation,
        self.yflip,
        values1d,
        sfile,
    )

    if status != 0:
        raise RuntimeError("Error when exporting to RMS regular")


def export_xtgregcube(self, mfile):
    """Export to experimental xtgregcube format, python version."""
    logger.info("Export as xtgregcube...")
    self.metadata.required = self

    prevalues = (1, 1201, 4, self.ncol, self.nrow, self.nlay)
    mystruct = struct.Struct("= i i i q q q")
    pre = mystruct.pack(*prevalues)

    meta = self.metadata.get_metadata()

    jmeta = json.dumps(meta).encode()

    with open(mfile, "wb") as fout:
        fout.write(pre)

    with open(mfile, "ab") as fout:
        # TODO. Treat dead traces as undef
        self.values.tofile(fout)

    with open(mfile, "ab") as fout:
        fout.write("\nXTGMETA.v01\n".encode())

    with open(mfile, "ab") as fout:
        fout.write(jmeta)

    logger.info("Export as xtgregcube... done")
