"""Import RegularSurface data."""
# pylint: disable=protected-access

import json
from collections import OrderedDict
from struct import unpack

import h5py
import numpy as np
import numpy.ma as ma
import xtgeo
import xtgeo.common.sys as xsys
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # pylint: disable=no-name-in-module
from xtgeo.common import XTGeoDialog
from xtgeo.common.constants import UNDEF_MAP_IRAPA, UNDEF_MAP_IRAPB
from xtgeo.surface._zmap_parser import parse_zmap

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_irap_binary(mfile, values=True, engine="cxtgeo", **_):
    """Import Irap binary format.

    Args:
        mfile (_XTGeoFile): Instance of xtgeo file class
        values (bool, optional): Getting values or just scan. Defaults to True.

    Raises:
        RuntimeError: Error in reading Irap binary file
        RuntimeError: Problem....
    """
    if mfile.memstream is True or engine == "python":
        return _import_irap_binary_purepy(mfile)
    else:
        return _import_irap_binary(mfile, values=values)


def _import_irap_binary_purepy(mfile, values=True):
    """Using pure python, better for memorymapping/threading."""
    # Borrowed some code from https://github.com/equinor/grequi/../fmt_irapbin.py

    logger.info("Enter function %s", __name__)

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
    else:
        with open(mfile.file, "rb") as fhandle:
            buf = fhandle.read()

    # unpack header with big-endian format string
    hed = unpack(">3i6f3i3f10i", buf[:100])

    args = {}
    args["nrow"] = hed[2]
    args["xori"] = hed[3]
    args["yori"] = hed[5]
    args["xinc"] = hed[7]
    args["yinc"] = hed[8]
    args["ncol"] = hed[11]
    args["rotation"] = hed[12]

    args["yflip"] = 1
    if args["yinc"] < 0.0:
        args["yinc"] *= -1
        args["yflip"] = -1

    if not values:
        return args

    # Values: traverse through data blocks
    stv = 100  # Starting byte
    datav = []

    while True:
        # start block integer - number of bytes of floats in following block
        blockv = unpack(">i", buf[stv : stv + 4])[0]
        stv += 4
        # floats
        datav.append(
            np.array(unpack(">" + str(int(blockv / 4)) + "f", buf[stv : blockv + stv]))
        )
        stv += blockv
        # end block integer not needed really
        _ = unpack(">i", buf[stv : stv + 4])[0]
        stv += 4
        if stv == len(buf):
            break

    values = np.hstack(datav)
    values = np.reshape(values, (args["ncol"], args["nrow"]), order="F")
    values = np.array(values, order="C")
    values = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPB)
    args["values"] = np.ma.masked_invalid(values)

    del buf
    return args


def _import_irap_binary(mfile, values=True):

    logger.info("Enter function %s", __name__)

    cfhandle = mfile.get_cfhandle()
    args = {}
    # read with mode 0, to get mx my and other metadata
    (
        ier,
        args["ncol"],
        args["nrow"],
        _,
        args["xori"],
        args["yori"],
        args["xinc"],
        args["yinc"],
        args["rotation"],
        val,
    ) = _cxtgeo.surf_import_irap_bin(cfhandle, 0, 1, 0)

    if ier != 0:
        mfile.cfclose()
        raise RuntimeError("Error in reading Irap binary file")

    args["yflip"] = 1
    if args["yinc"] < 0.0:
        args["yinc"] *= -1
        args["yflip"] = -1

    # lazy loading, not reading the arrays
    if values:
        nval = args["ncol"] * args["nrow"]
        xlist = _cxtgeo.surf_import_irap_bin(cfhandle, 1, nval, 0)
        if xlist[0] != 0:
            mfile.cfclose()
            raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

        val = xlist[-1]

        val = np.reshape(val, (args["ncol"], args["nrow"]), order="C")

        val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)

        if np.isnan(val).any():
            logger.info("NaN values are found, will mask...")
            val = ma.masked_invalid(val)

        args["values"] = val

    mfile.cfclose()
    return args


def import_irap_ascii(mfile, engine="cxtgeo", **_):
    """Import Irap ascii format, where mfile is a _XTGeoFile instance."""
    #   -996  2010      5.000000      5.000000
    #    461587.553724   467902.553724  5927061.430176  5937106.430176
    #   1264       30.000011   461587.553724  5927061.430176
    #      0     0     0     0     0     0     0
    #     1677.3239    1677.3978    1677.4855    1677.5872    1677.7034    1677.8345
    #     1677.9807    1678.1420    1678.3157    1678.5000    1678.6942    1678.8973
    #     1679.1086    1679.3274    1679.5524    1679.7831    1680.0186    1680.2583
    #     1680.5016    1680.7480    1680.9969    1681.2479    1681.5004    1681.7538
    #

    if mfile.memstream is True or engine == "python":
        return _import_irap_ascii_purepy(mfile)
    else:
        return _import_irap_ascii(mfile)


def _import_irap_ascii_purepy(mfile):
    """Import Irap in pure python code, suitable for memstreams, but less efficient."""
    # timer tests suggest approx double load time compared with cxtgeo method

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
        buf = buf.decode().split()
    else:
        with open(mfile.file) as fhandle:
            buf = fhandle.read().split()
    args = {}
    args["nrow"] = int(buf[1])
    args["xinc"] = float(buf[2])
    args["yinc"] = float(buf[3])
    args["xori"] = float(buf[4])
    args["yori"] = float(buf[6])
    args["ncol"] = int(buf[8])
    args["rotation"] = float(buf[9])

    values = np.array(buf[19:]).astype(np.float64)
    values = np.reshape(values, (args["ncol"], args["nrow"]), order="F")
    values = np.array(values, order="C")
    args["values"] = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPA)

    args["yflip"] = 1
    if args["yinc"] < 0.0:
        args["yinc"] *= -1
        args["yflip"] = -1

    del buf
    return args


def _import_irap_ascii(mfile):
    """Import Irap ascii format via C routines (fast, but less suited for bytesio)."""
    logger.debug("Enter function...")

    cfhandle = mfile.get_cfhandle()

    # read with mode 0, scan to get mx my
    xlist = _cxtgeo.surf_import_irap_ascii(cfhandle, 0, 1, 0)

    nvn = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_ascii(cfhandle, 1, nvn, 0)

    ier, ncol, nrow, _, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        mfile.cfclose()
        raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order="C")

    val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info("NaN values are found, will mask...")
        val = ma.masked_invalid(val)

    yflip = 1
    if yinc < 0.0:
        yinc = yinc * -1
        yflip = -1
    args = {}
    args["ncol"] = ncol
    args["nrow"] = nrow
    args["xori"] = xori
    args["yori"] = yori
    args["xinc"] = xinc
    args["yinc"] = yinc
    args["yflip"] = yflip
    args["rotation"] = rot

    args["values"] = val

    mfile.cfclose()
    return args


def import_ijxyz(mfile, template=None, **_):
    """Import OW/DSG IJXYZ ascii format."""

    if not template:
        return _import_ijxyz(mfile)
    else:
        return _import_ijxyz_tmpl(mfile, template)


def _import_ijxyz(mfile):  # pylint: disable=too-many-locals
    """Import OW/DSG IJXYZ ascii format."""
    # import of seismic column system on the form:
    # 2588	1179	476782.2897888889	6564025.6954	1000.0
    # 2588	1180	476776.7181777778	6564014.5058	1000.0
    logger.debug("Read data from file... (scan for dimensions)")

    cfhandle = mfile.get_cfhandle()

    xlist = _cxtgeo.surf_import_ijxyz(cfhandle, 0, 1, 1, 1, 0)

    (
        ier,
        ncol,
        nrow,
        _,
        xori,
        yori,
        xinc,
        yinc,
        rot,
        iln,
        xln,
        val,
        yflip,
    ) = xlist

    if ier != 0:
        mfile.cfclose()
        raise RuntimeError("Import from C is wrong...")

    # now real read mode
    xlist = _cxtgeo.surf_import_ijxyz(cfhandle, 1, ncol, nrow, ncol * nrow, 0)

    ier, ncol, nrow, _, xori, yori, xinc, yinc, rot, iln, xln, val, yflip = xlist

    if ier != 0:
        raise RuntimeError("Import from C is wrong...")

    logger.info(xlist)

    val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)
    args = {}
    args["xori"] = xori
    args["xinc"] = xinc
    args["yori"] = yori
    args["yinc"] = yinc
    args["ncol"] = ncol
    args["nrow"] = nrow
    args["rotation"] = rot
    args["yflip"] = yflip

    args["values"] = val.reshape((args["ncol"], args["nrow"]))

    args["ilines"] = iln
    args["xlines"] = xln

    mfile.cfclose()
    return args


def _import_ijxyz_tmpl(mfile, template):
    """Import OW/DSG IJXYZ ascii format, with a Cube or RegularSurface as template."""
    cfhandle = mfile.get_cfhandle()

    if isinstance(template, (xtgeo.cube.Cube, xtgeo.surface.RegularSurface)):
        logger.info("OK template")
    else:
        raise ValueError("Template is of wrong type: {}".format(type(template)))

    nxy = template.ncol * template.nrow
    ier, val = _cxtgeo.surf_import_ijxyz_tmpl(
        cfhandle, template.ilines, template.xlines, nxy, 0
    )

    if ier == -1:
        raise ValueError(
            f"The file {mfile.name} and template map or cube has inconsistent "
            "inline and/or xlines numbering. Try importing without template "
            "and use e.g. resampling instead."
        )

    elif ier != 0:
        raise RuntimeError("Unknown error when trying to import the IJXYZ based file!")

    val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)

    args = {}
    args["xori"] = template.xori
    args["xinc"] = template.xinc
    args["yori"] = template.yori
    args["yinc"] = template.yinc
    args["ncol"] = template.ncol
    args["nrow"] = template.nrow
    args["rotation"] = template.rotation
    args["yflip"] = template.yflip
    args["values"] = val.reshape((args["ncol"], args["nrow"]))

    args["ilines"] = template._ilines.copy()
    args["xlines"] = template._xlines.copy()

    mfile.cfclose()
    return args


def import_petromod(mfile, **_):
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

    values = np.ma.masked_greater(values, xtgeo.UNDEF_LIMIT)

    args["values"] = values.reshape(args["ncol"], args["nrow"])

    mfile.cfclose()
    return args


def import_zmap_ascii(mfile, values=True, **_):
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

    meta = json.loads(jmeta, object_pairs_hook=OrderedDict)
    req = meta["_required_"]

    reqattrs = xtgeo.MetaDataRegularSurface.REQUIRED

    args = {}
    for myattr in reqattrs:
        args[myattr] = req[myattr]

    if values:
        args["values"] = np.ma.masked_equal(
            vals.reshape(args["ncol"], args["nrow"]), xtgeo.UNDEF
        )

    return args


def import_hdf5_regsurf(mfile, values=True, **_):
    """Importing h5/hdf5 storage."""
    reqattrs = xtgeo.MetaDataRegularSurface.REQUIRED

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
        meta = json.loads(jmeta, object_pairs_hook=OrderedDict)

        req = meta["_required_"]

    args = {}
    for myattr in reqattrs:
        args[myattr] = req[myattr]

    if values:
        args["values"] = np.ma.masked_equal(
            invalues.reshape(args["ncol"], args["nrow"]), xtgeo.UNDEF
        )

    return args
