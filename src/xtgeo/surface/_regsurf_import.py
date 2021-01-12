"""Import RegularSurface data."""
# pylint: disable=protected-access

import json
from collections import OrderedDict

import numpy as np
import numpy.ma as ma
from struct import unpack
import h5py

from xtgeo.common.constants import UNDEF_MAP_IRAPB, UNDEF_MAP_IRAPA
import xtgeo
import xtgeo.common.sys as xsys
import xtgeo.cxtgeo._cxtgeo as _cxtgeo  # pylint: disable=no-name-in-module
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def import_irap_binary(self, mfile, values=True, engine="cxtgeo", **kwargs):
    """Import Irap binary format.

    Args:
        mfile (_XTGeoFile): Instance of xtgeo file class
        values (bool, optional): Getting values or just scan. Defaults to True.

    Raises:
        RuntimeError: Error in reading Irap binary file
        RuntimeError: Problem....
    """
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)
    if mfile.memstream is True or engine == "python":
        _import_irap_binary_purepy(self, mfile)
    else:
        _import_irap_binary(self, mfile, values=values)

    self._metadata.required = self
    self._isloaded = values


def _import_irap_binary_purepy(self, mfile, values=True):
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

    self._nrow = hed[2]
    self._xori = hed[3]
    self._yori = hed[5]
    self._xinc = hed[7]
    self._yinc = hed[8]
    self._ncol = hed[11]
    self._rotation = hed[12]

    self._yflip = 1
    if self._yinc < 0.0:
        self._yinc *= -1
        self._yflip = -1

    if not values:
        self._isloaded = False
        self._values = None
        return

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
    values = np.reshape(values, (self._ncol, self._nrow), order="F")
    values = np.array(values, order="C")
    values = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPB)
    self._values = np.ma.masked_invalid(values)

    self._ilines = np.array(range(1, self.ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self.nrow + 1), dtype=np.int32)

    del buf


def _import_irap_binary(self, mfile, values=True):

    logger.info("Enter function %s", __name__)

    cfhandle = mfile.get_cfhandle()

    # read with mode 0, to get mx my and other metadata
    (
        ier,
        self._ncol,
        self._nrow,
        _,
        self._xori,
        self._yori,
        self._xinc,
        self._yinc,
        self._rotation,
        val,
    ) = _cxtgeo.surf_import_irap_bin(cfhandle, 0, 1, 0)

    if ier != 0:
        mfile.cfclose()
        raise RuntimeError("Error in reading Irap binary file")

    self._yflip = 1
    if self._yinc < 0.0:
        self._yinc *= -1
        self._yflip = -1

    # lazy loading, not reading the arrays
    if not values:
        self._values = None
        mfile.cfclose()
        return

    nval = self._ncol * self._nrow
    xlist = _cxtgeo.surf_import_irap_bin(cfhandle, 1, nval, 0)
    if xlist[0] != 0:
        mfile.cfclose()
        raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

    val = xlist[-1]

    val = np.reshape(val, (self._ncol, self._nrow), order="C")

    val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info("NaN values are found, will mask...")
        val = ma.masked_invalid(val)

    self._isloaded = True
    self.values = val

    self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

    mfile.cfclose()


def import_irap_ascii(self, mfile, engine="cxtgeo", **kwargs):
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
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)

    if mfile.memstream is True or engine == "python":
        _import_irap_ascii_purepy(self, mfile)
    else:
        _import_irap_ascii(self, mfile)

    self._metadata.required = self


def _import_irap_ascii_purepy(self, mfile):
    """Import Irap in pure python code, suitable for memstreams, but less efficient."""
    # timer tests suggest approx double load time compared with cxtgeo method

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
        buf = buf.decode().split()
    else:
        with open(mfile.file) as fhandle:
            buf = fhandle.read().split()

    self._nrow = int(buf[1])
    self._xinc = float(buf[2])
    self._yinc = float(buf[3])
    self._xori = float(buf[4])
    self._yori = float(buf[6])
    self._ncol = int(buf[8])
    self._rotation = float(buf[9])

    values = np.array(buf[19:]).astype(np.float64)
    values = np.reshape(values, (self._ncol, self._nrow), order="F")
    values = np.array(values, order="C")
    self.values = np.ma.masked_greater_equal(values, UNDEF_MAP_IRAPA)

    self._yflip = 1
    if self._yinc < 0.0:
        self._yinc *= -1
        self._yflip = -1

    self._ilines = np.array(range(1, self.ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self.nrow + 1), dtype=np.int32)

    del buf


def _import_irap_ascii(self, mfile):
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

    self._ncol = ncol
    self._nrow = nrow
    self._xori = xori
    self._yori = yori
    self._xinc = xinc
    self._yinc = yinc
    self._yflip = yflip
    self._rotation = rot

    self.values = val

    self._ilines = np.array(range(1, ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, nrow + 1), dtype=np.int32)

    mfile.cfclose()


def import_ijxyz(self, mfile, template=None, **kwargs):
    """Import OW/DSG IJXYZ ascii format."""
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)

    if not template:
        _import_ijxyz(self, mfile)
    else:
        _import_ijxyz_tmpl(self, mfile, template)


def _import_ijxyz(self, mfile):  # pylint: disable=too-many-locals
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

    self._xori = xori
    self._xinc = xinc
    self._yori = yori
    self._yinc = yinc
    self._ncol = ncol
    self._nrow = nrow
    self._rotation = rot
    self._yflip = yflip

    self.values = val.reshape((self._ncol, self._nrow))

    self._ilines = iln
    self._xlines = xln

    mfile.cfclose()
    self._metadata.required = self


def _import_ijxyz_tmpl(self, mfile, template):
    """Import OW/DSG IJXYZ ascii format, with a Cube or RegularSurface as template."""
    cfhandle = mfile.get_cfhandle()

    if isinstance(template, (xtgeo.cube.Cube, xtgeo.surface.RegularSurface)):
        logger.info("OK template")
    else:
        raise ValueError("Template is of wrong type: {}".format(type(template)))

    nxy = template.ncol * template.nrow
    _, val = _cxtgeo.surf_import_ijxyz_tmpl(
        cfhandle, template.ilines, template.xlines, nxy, 0
    )

    val = ma.masked_greater(val, xtgeo.UNDEF_LIMIT)

    self._xori = template.xori
    self._xinc = template.xinc
    self._yori = template.yori
    self._yinc = template.yinc
    self._ncol = template.ncol
    self._nrow = template.nrow
    self._rotation = template.rotation
    self._yflip = template.yflip
    self.values = val.reshape((self._ncol, self._nrow))

    self._ilines = template._ilines.copy()
    self._xlines = template._xlines.copy()

    mfile.cfclose()
    self._metadata.required = self


def import_petromod(self, mfile, values=True, **kwargs):
    """Import Petromod binary format."""
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)

    cfhandle = mfile.get_cfhandle()

    logger.info("Enter function %s", __name__)

    # read with mode 0, to get mx my and other metadata
    dsc, _ = _cxtgeo.surf_import_petromod_bin(cfhandle, 0, 0.0, 0, 0, 0)

    fields = dsc.split(",")

    rota_xori = 0
    rota_yori = 0
    undef = 999999.0

    for field in fields:
        key, value = field.split("=")
        if key == "GridNoX":
            self._ncol = int(value)
        if key == "GridNoY":
            self._nrow = int(value)
        if key == "OriginX":
            self._xori = float(value)
        if key == "OriginY":
            self._yori = float(value)
        if key == "RotationOriginX":
            rota_xori = float(value)
        if key == "RotationOriginY":
            rota_yori = float(value)
        if key == "GridStepX":
            self._xinc = float(value)
        if key == "GridStepY":
            self._yinc = float(value)
        if key == "RotationAngle":
            self._rotation = float(value)
        if key == "Undefined":
            undef = float(value)

    if self._rotation != 0.0 and (rota_xori != self._xori or rota_yori != self._yori):
        xtg.warnuser("Rotation origin and data origin do match")

    # reread file for map values

    dsc, values = _cxtgeo.surf_import_petromod_bin(
        cfhandle, 1, undef, self._ncol, self._nrow, self._ncol * self._nrow
    )

    values = np.ma.masked_greater(values, xtgeo.UNDEF_LIMIT)

    values = values.reshape(self._ncol, self._nrow)

    self.values = values

    self._ilines = np.array(range(1, self.ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self.nrow + 1), dtype=np.int32)

    mfile.cfclose()
    self._metadata.required = self


def import_zmap_ascii(self, mfile, values=True, **kwargs):
    """Importing ZMAP + ascii files, in pure python only.

    Some sources

    https://mycarta.wordpress.com/2019/03/23/working-with-zmap-grid-files-in-python/
    https://blog.nitorinfotech.com/what-is-zmap-plus-file-format/

    """
    logger.debug("Additional, probably unused kwargs: %s", **kwargs)

    if not mfile.memstream:
        logger.info("Reading zmap+ from %s", mfile.file)
    else:
        logger.info("Reading zmap+ from memory stream")

    count = 0
    buffer = ""
    self._values = None

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
        buf = buf.decode().split("\n")
        for line in buf:
            buffer += line
            count += 1
            if line.startswith("@\n"):
                break
    else:
        with open(mfile.file, "r", encoding="utf-8") as fhandle:
            for line in fhandle:
                buffer += line
                count += 1
                if line.startswith("@\n"):
                    break

    # interpret header
    correctformat = False
    header = []
    lines = buffer.split("\n")
    for line in lines:
        if line.startswith("!"):
            continue
        elif line.startswith("@ GRIDFILE, GRID"):
            correctformat = True
        elif line.startswith("@\n"):
            break
        else:
            hdr = [xxx.strip() for xxx in line.split(",")]
            header.extend(hdr)

    if not correctformat:
        raise ValueError("Input file does not seem to be a correct zmap file")
    print(header)

    self._ncol = int(header[6])
    self._nrow = int(header[5])
    self._xori = float(header[7])
    self._yori = float(header[9])
    self._xinc = float(header[8]) - float(header[7]) / (self._ncol + 1)
    self._yinc = float(header[10]) - float(header[9]) / (self._nrow + 1)

    if header[2]:
        undef = float(header[2])  # user defined undef
    else:
        undef = float(header[1])

    logger.info("UNDEF value is %s", undef)
    print(self._xinc)

    if values is False:
        self.isloaded = False
        return

    if mfile.memstream:
        mfile.file.seek(0)
        buf = mfile.file.read()
        buf = buf.decode().split()
        indexes = [i for i, x in enumerate(buf) if x == "@"]
        buf = buf[indexes[-1] + 1 :]
    else:
        with open(mfile.file, "r", encoding="utf-8") as fhx:
            buf = fhx.read()

            buf = buf.split()
            indexes = [i for i, x in enumerate(buf) if x == "@"]
            buf = buf[indexes[-1] + 1 :]

    values = np.array(buf, dtype=np.float64)
    values = np.reshape(values, (self._ncol, self._nrow), order="C")
    values = np.flip(values, axis=1)
    self._values = np.ma.masked_equal(values, undef)

    self._ilines = np.array(range(1, self.ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self.nrow + 1), dtype=np.int32)

    logger.info("Reading zmap+... done")
    self._metadata.required = self


def import_xtg(self, mfile, values=True, **kwargs):
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

    for myattr in reqattrs:
        setattr(self, "_" + myattr, req[myattr])

    if values:
        self.values = np.ma.masked_equal(
            vals.reshape(self.ncol, self.nrow), self._undef
        )
    else:
        self._values = None

    self._metadata.required = self


def import_hdf5_regsurf(self, mfile, values=True):
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

    for myattr in reqattrs:
        setattr(self, "_" + myattr, req[myattr])

    if values:
        self.values = np.ma.masked_equal(
            invalues.reshape(self.ncol, self.nrow), self._undef
        )
    else:
        self._values = None

    self._metadata.required = self
