"""Import RegularSurface data."""
# pylint: disable=protected-access

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo  # pylint: disable=import-error
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

DEBUG = 0
if DEBUG < 0:
    DEBUG = 0


def import_irap_binary(self, mfile, values=True):
    """Import Irap binary format."""

    ifile = xtgeo._XTGeoCFile(mfile)

    logger.debug("Enter function...")
    # read with mode 0, to get mx my and other metadata
    (
        ier,
        self._ncol,
        self._nrow,
        _ndef,
        self._xori,
        self._yori,
        self._xinc,
        self._yinc,
        self._rotation,
        val,
    ) = _cxtgeo.surf_import_irap_bin(ifile.fhandle, 0, 1, 0)

    if ier != 0:
        ifile.close()
        raise RuntimeError("Error in reading Irap binary file")

    self._yflip = 1
    if self._yinc < 0.0:
        self._yinc *= -1
        self._yflip = -1

    self._filesrc = mfile

    self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

    # lazy loading, not reading the arrays
    if not values:
        self._values = None
        ifile.close()
        return

    nval = self._ncol * self._nrow
    xlist = _cxtgeo.surf_import_irap_bin(ifile.fhandle, 1, nval, 0)
    if xlist[0] != 0:
        ifile.close()
        raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

    val = xlist[-1]

    val = np.reshape(val, (self._ncol, self._nrow), order="C")

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info("NaN values are found, will mask...")
        val = ma.masked_invalid(val)

    self._values = val

    ifile.close()


def import_irap_binary2(self, mfile, values=True):
    """Import Irap binary format, variant 2."""

    logger.debug("Enter function (v2)...")

    pmx = _cxtgeo.new_intpointer()
    pmy = _cxtgeo.new_intpointer()
    pll = _cxtgeo.new_longpointer()
    pxori = _cxtgeo.new_doublepointer()
    pyori = _cxtgeo.new_doublepointer()
    pxinc = _cxtgeo.new_doublepointer()
    pyinc = _cxtgeo.new_doublepointer()
    prot = _cxtgeo.new_doublepointer()
    pmap = _cxtgeo.new_doublearray(1)
    logger.debug("Enter function (v2)... scanning start")

    _cxtgeo.surf_import_irap_bin2(mfile, 0, pmx, pmy, pll, pxori, pyori, pxinc, pyinc,
                                  prot, pmap, 1, 0)

    logger.debug("Enter function (v2)... scanning done")


    # # read with mode 0, to get mx my and other metadata
    # (
    #     ier,
    #     self._ncol,
    #     self._nrow,
    #     _ndef,
    #     self._xori,
    #     self._yori,
    #     self._xinc,
    #     self._yinc,
    #     self._rotation,
    #     val,
    # ) = _cxtgeo.surf_import_irap_bin2(mfile, 0, 1, 0)

    # logger.debug("Enter function (v2)... scanning done")

    # if ier != 0:
    #     raise RuntimeError("Error in reading Irap binary file")

    # self._yflip = 1
    # if self._yinc < 0.0:
    #     self._yinc *= -1
    #     self._yflip = -1

    # self._filesrc = mfile

    # self._ilines = np.array(range(1, self._ncol + 1), dtype=np.int32)
    # self._xlines = np.array(range(1, self._nrow + 1), dtype=np.int32)

    # # lazy loading, not reading the arrays
    # if not values:
    #     self._values = None
    #     return

    # nval = self._ncol * self._nrow
    # xlist = _cxtgeo.surf_import_irap_bin2(mfile, 1, nval, 0)
    # if xlist[0] != 0:
    #     raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

    # val = xlist[-1]

    # val = np.reshape(val, (self._ncol, self._nrow), order="C")

    # val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    # if np.isnan(val).any():
    #     logger.info("NaN values are found, will mask...")
    #     val = ma.masked_invalid(val)

    # self._values = val


def import_irap_ascii(self, mfile):
    """Import Irap ascii format."""
    # version using swig type mapping

    logger.debug("Enter function...")
    ifile = xtgeo._XTGeoCFile(mfile)

    # read with mode 0, scan to get mx my
    xlist = _cxtgeo.surf_import_irap_ascii(ifile.fhandle, 0, 1, 0)

    nvn = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_ascii(ifile.fhandle, 1, nvn, 0)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
        ifile.close()
        raise RuntimeError("Problem in {}, code {}".format(__name__, ier))

    val = np.reshape(val, (ncol, nrow), order="C")

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

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
    self._values = val
    self._filesrc = mfile

    self._ilines = np.array(range(1, ncol + 1), dtype=np.int32)
    self._xlines = np.array(range(1, nrow + 1), dtype=np.int32)

    ifile.close()


def import_ijxyz_ascii(self, mfile):  # pylint: disable=too-many-locals
    """Import OW/DSG IJXYZ ascii format."""

    # import of seismic column system on the form:
    # 2588	1179	476782.2897888889	6564025.6954	1000.0
    # 2588	1180	476776.7181777778	6564014.5058	1000.0

    logger.debug("Read data from file... (scan for dimensions)")

    fin = xtgeo._XTGeoCFile(mfile)

    xlist = _cxtgeo.surf_import_ijxyz(fin.fhandle, 0, 1, 1, 1, 0)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, iln, xln, val, yflip = xlist

    if ier != 0:
        fin.close()
        raise RuntimeError("Import from C is wrong...")

    # now real read mode
    xlist = _cxtgeo.surf_import_ijxyz(fin.fhandle, 1, ncol, nrow, ncol * nrow, 0)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, iln, xln, val, yflip = xlist

    if ier != 0:
        raise RuntimeError("Import from C is wrong...")

    logger.info(xlist)

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    self._xori = xori
    self._xinc = xinc
    self._yori = yori
    self._yinc = yinc
    self._ncol = ncol
    self._nrow = nrow
    self._rotation = rot
    self._yflip = yflip
    self._values = val.reshape((self._ncol, self._nrow))
    self._filesrc = mfile

    self._ilines = iln
    self._xlines = xln

    fin.close()


def import_ijxyz_ascii_tmpl(self, mfile, template):
    """Import OW/DSG IJXYZ ascii format, with a Cube or RegularSurface
    instance as template."""

    fin = xtgeo._XTGeoCFile(mfile)

    if isinstance(template, (xtgeo.cube.Cube, xtgeo.surface.RegularSurface)):
        logger.info("OK template")
    else:
        raise ValueError("Template is of wrong type: {}".format(type(template)))

    nxy = template.ncol * template.nrow
    _iok, val = _cxtgeo.surf_import_ijxyz_tmpl(
        fin.fhandle, template.ilines, template.xlines, nxy, 0
    )

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    self._xori = template.xori
    self._xinc = template.xinc
    self._yori = template.yori
    self._yinc = template.yinc
    self._ncol = template.ncol
    self._nrow = template.nrow
    self._rotation = template.rotation
    self._yflip = template.yflip
    self._values = val.reshape((self._ncol, self._nrow))
    self._filesrc = mfile

    self._ilines = template._ilines.copy()
    self._xlines = template._xlines.copy()

    fin.close()
