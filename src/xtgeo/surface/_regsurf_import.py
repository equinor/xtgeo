"""Import RegularSurface data."""

from __future__ import annotations

import json
import mmap
from struct import unpack
from typing import TYPE_CHECKING

import h5py
import numpy as np

import xtgeo.common.sys as xsys
from xtgeo import _cxtgeo
from xtgeo.common.constants import UNDEF, UNDEF_LIMIT, UNDEF_MAP_IRAPA, UNDEF_MAP_IRAPB
from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGeoDialog
from xtgeo.metadata.metadata import MetaDataRegularSurface

from ._regsurf_ijxyz_parser import parse_ijxyz
from ._zmap_parser import parse_zmap

if TYPE_CHECKING:
    from xtgeo.cube.cube1 import Cube
    from xtgeo.io._file import FileWrapper
    from xtgeo.surface.regular_surface import RegularSurface

xtg = XTGeoDialog()

logger = null_logger(__name__)


def import_irap_binary(mfile: FileWrapper, values: bool = True, **_):
    """Using pure python from version 4.X.

    Reverse engineering says that the BINARY header is
    <32> IDFLAG NY XORI XMAX YORI YMAX XINC YINC <32>
    <16> NX ROT X0ORI Y0ORI<16>
    <28> 0 0 0 0 0 0 0 <28>
    ---data---
    Note, XMAX and YMAX are based on unroted distances and are
    not used directly? =>
    XINC = (XMAX-XORI)/(NX-1) etc
    X0ORI/Y0ORI seems to be rotation origin? Set them equal to XORI/YORI

    """

    logger.info("Enter function %s", __name__)

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
    else:
        with open(mfile.file, "rb") as fhandle:
            buf = mmap.mmap(fhandle.fileno(), 0, access=mmap.ACCESS_READ)

    # Ensure buffer is large enough for header
    header_size = 100
    if len(buf) < header_size:
        raise ValueError("Buffer size is too small for header")

    # unpack header with big-endian format string (cf. docstring info)
    hed = np.frombuffer(
        buf[:header_size],
        dtype=">i4,>i4,>i4,>f4,>f4,>f4,>f4,>f4,>f4,>i4,"  # <32> IDFLAG NY XORI ... <32>
        + ">i4,>i4,>f4,>f4,>f4,>i4,"  # <16> NX ROT X0ORI Y0ORI<16>
        + ">i4,>i4,>i4,>i4,>i4,>i4,>i4,>i4,>i4",  # <28> 0 0 0 0 0 0 0 <28>
    )

    args = {}
    args["nrow"] = int(hed[0][2])
    args["xori"] = float(hed[0][3])
    args["yori"] = float(hed[0][5])
    args["xinc"] = float(hed[0][7])
    args["yinc"] = float(hed[0][8])
    args["ncol"] = int(hed[0][11])
    args["rotation"] = float(hed[0][12])

    args["yflip"] = 1
    if args["yinc"] < 0.0:
        args["yinc"] *= -1
        args["yflip"] = -1

    if not values:
        return args

    # Values: traverse through data blocks
    stv = header_size  # Starting byte
    datav = []

    while stv < len(buf):
        # start block integer - number of bytes of floats in following block
        blockv = np.frombuffer(buf[stv : stv + 4], dtype=">i4")[0]
        stv += 4
        # floats
        datav.append(np.frombuffer(buf[stv : stv + blockv], dtype=">f4"))
        stv += blockv
        # end block integer not needed really
        stv += 4

    values = np.hstack(datav)
    values = np.reshape(values, (args["ncol"], args["nrow"]), order="F")
    values = np.array(values, order="C")
    values = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPB)
    args["values"] = np.ma.masked_invalid(values)

    del buf
    return args


def import_irap_ascii(mfile: FileWrapper, **_):
    """Import Irap in pure python code, suitable for memstreams, and now efficient.
    -996  2010      5.000000      5.000000
    461587.553724   467902.553724  5927061.430176  5937106.430176
    1264       30.000011   461587.553724  5927061.430176
       0     0     0     0     0     0     0
      1677.3239    1677.3978    1677.4855    1677.5872    1677.7034    1677.8345
      1677.9807    1678.1420    1678.3157    1678.5000    1678.6942    1678.8973
      1679.1086    1679.3274    1679.5524    1679.7831    1680.0186    1680.2583
      1680.5016    1680.7480    1680.9969    1681.2479    1681.5004    1681.7538
       ....
    """

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read().decode()
    else:
        with open(mfile.file) as fhandle:
            buf = fhandle.read()

    buf = buf.split(maxsplit=19)
    args = {}
    args["nrow"] = int(buf[1])
    args["xinc"] = float(buf[2])
    args["yinc"] = float(buf[3])
    args["xori"] = float(buf[4])
    args["yori"] = float(buf[6])
    args["ncol"] = int(buf[8])
    args["rotation"] = float(buf[9])

    nvalues = args["nrow"] * args["ncol"]
    values = np.fromstring(buf[19], dtype=np.double, count=nvalues, sep=" ")

    values = np.reshape(values, (args["ncol"], args["nrow"]), order="F")
    values = np.array(values, order="C")
    args["values"] = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPA)

    args["yflip"] = 1
    if args["yinc"] < 0.0:
        args["yinc"] *= -1
        args["yflip"] = -1

    del buf
    return args


def import_ijxyz(
    mfile: FileWrapper,
    template: RegularSurface | Cube | None = None,
    **_,
) -> dict:
    """Import OW/DSG IJXYZ ascii format."""
    ijxyz_data = parse_ijxyz(mfile, template)

    return {
        "ncol": ijxyz_data.ncol,
        "nrow": ijxyz_data.nrow,
        "xori": ijxyz_data.xori,
        "yori": ijxyz_data.yori,
        "xinc": ijxyz_data.xinc,
        "yinc": ijxyz_data.yinc,
        "values": ijxyz_data.values,
        "ilines": ijxyz_data.ilines,
        "xlines": ijxyz_data.xlines,
        "yflip": ijxyz_data.yflip,
        "rotation": ijxyz_data.rotation,
    }


def import_petromod(mfile: FileWrapper, **_):
    """Import Petromod binary format."""

    cfhandle = mfile.get_cfhandle()

    logger.info("Enter function %s", __name__)

    # read with mode 0, to get mx my and other metadata
    dsc, _ = _cxtgeo.surf_import_petromod_bin(cfhandle, 0, 0.0, 0, 0, 0)

    fields = dsc.split(",")

    rota_xori = 0
    rota_yori = 0
    undef = 999999.0
    args = {}
    for field in fields:
        key, value = field.split("=")
        if key == "GridNoX":
            args["ncol"] = int(value)
        if key == "GridNoY":
            args["nrow"] = int(value)
        if key == "OriginX":
            args["xori"] = float(value)
        if key == "OriginY":
            args["yori"] = float(value)
        if key == "RotationOriginX":
            rota_xori = float(value)
        if key == "RotationOriginY":
            rota_yori = float(value)
        if key == "GridStepX":
            args["xinc"] = float(value)
        if key == "GridStepY":
            args["yinc"] = float(value)
        if key == "RotationAngle":
            args["rotation"] = float(value)
        if key == "Undefined":
            undef = float(value)

    if args["rotation"] != 0.0 and (
        rota_xori != args["xori"] or rota_yori != args["yori"]
    ):
        xtg.warnuser("Rotation origin and data origin do match")

    # reread file for map values

    dsc, values = _cxtgeo.surf_import_petromod_bin(
        cfhandle, 1, undef, args["ncol"], args["nrow"], args["ncol"] * args["nrow"]
    )

    values = np.ma.masked_greater(values, UNDEF_LIMIT)

    args["values"] = values.reshape(args["ncol"], args["nrow"])

    mfile.cfclose()
    return args


def import_zmap_ascii(mfile: FileWrapper, values: bool = True, **_):
    """Importing ZMAP + ascii files, in pure python only.

    Some sources

    https://mycarta.wordpress.com/2019/03/23/working-with-zmap-grid-files-in-python/
    https://blog.nitorinfotech.com/what-is-zmap-plus-file-format/

    """
    zmap_data = parse_zmap(mfile.file, load_values=values)
    try:
        args = {
            "ncol": zmap_data.ncol,
            "nrow": zmap_data.nrow,
            "xori": zmap_data.xmin,
            "yori": zmap_data.ymin,
            "xinc": (zmap_data.xmax - zmap_data.xmin) / (zmap_data.ncol - 1),
            "yinc": (zmap_data.ymax - zmap_data.ymin) / (zmap_data.nrow - 1),
        }
    except ZeroDivisionError as err:
        raise ValueError(
            f"A zmap surface must have ncol ({zmap_data.ncol}) "
            f"and nrow ({zmap_data.ncol}) > 1"
        ) from err
    if values:
        loaded_values = np.reshape(
            zmap_data.values, (zmap_data.ncol, zmap_data.nrow), order="C"
        )
        loaded_values = np.flip(loaded_values, axis=1)
        args["values"] = loaded_values

    return args


def import_xtg(mfile, values=True, **kwargs):
    """Using pure python for experimental XTGEO import."""
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)

    offset = 28
    with open(mfile.file, "rb") as fhandle:
        buf = fhandle.read(offset)

    # unpack header
    swap, magic, nfloat, ncol, nrow = unpack("= i i i q q", buf)

    if swap != 1 or magic != 1101:
        raise ValueError("Invalid file format (wrong swap id or magic number).")

    dtype = np.float32 if nfloat == 4 else np.float64

    vals = None
    narr = ncol * nrow
    if values:
        vals = xsys.npfromfile(mfile.file, dtype=dtype, count=narr, offset=offset)

    # read metadata which will be at position offet + nfloat*narr +13
    pos = offset + nfloat * narr + 13

    with open(mfile.file, "rb") as fhandle:
        fhandle.seek(pos)
        jmeta = fhandle.read().decode()

    meta = json.loads(jmeta, object_pairs_hook=dict)
    req = meta["_required_"]

    reqattrs = MetaDataRegularSurface.REQUIRED

    args = {}
    for myattr in reqattrs:
        args[myattr] = req[myattr]

    if values:
        args["values"] = np.ma.masked_equal(
            vals.reshape(args["ncol"], args["nrow"]), UNDEF
        )

    return args


def import_hdf5_regsurf(mfile: FileWrapper, values=True, **_):
    """Importing h5/hdf5 storage."""
    reqattrs = MetaDataRegularSurface.REQUIRED

    invalues = None
    with h5py.File(mfile.name, "r") as h5h:
        grp = h5h["RegularSurface"]
        idcode = grp.attrs["format-idcode"]
        provider = grp.attrs["provider"]
        if idcode != 1101:
            raise ValueError(f"Wrong id code: {idcode}")
        logger.info("Provider is %s", provider)

        if values:
            invalues = grp["values"][:]

        jmeta = grp.attrs["metadata"]
        meta = json.loads(jmeta, object_pairs_hook=dict)

        req = meta["_required_"]

    args = {}
    for myattr in reqattrs:
        args[myattr] = req[myattr]

    if values:
        args["values"] = np.ma.masked_equal(
            invalues.reshape(args["ncol"], args["nrow"]), UNDEF
        )

    return args
