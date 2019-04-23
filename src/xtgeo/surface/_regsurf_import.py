"""Import RegularSurface data."""
# pylint: disable=protected-access

import numpy as np
import numpy.ma as ma

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo  # pylint: disable=import-error
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

DEBUG = xtg.get_syslevel()
if DEBUG < 0:
    DEBUG = 0

_cxtgeo.xtg_verbose_file("NONE")


def import_irap_binary(self, mfile):
    """Import Irap binary format."""
    # using swig type mapping

    logger.debug("Enter function...")
    # need to call the C function...
    _cxtgeo.xtg_verbose_file("NONE")

    # read with mode 0, to get mx my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 0, 1, 0, DEBUG)

    nval = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 1, nval, 0, DEBUG)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
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


def import_irap_ascii(self, mfile):
    """Import Irap ascii format."""
    # version using swig type mapping

    logger.debug("Enter function...")
    # need to call the C function...
    _cxtgeo.xtg_verbose_file("NONE")

    # read with mode 0, scan to get mx my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 0, 1, 0, DEBUG)

    nvn = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_ascii(mfile, 1, nvn, 0, DEBUG)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, val = xlist

    if ier != 0:
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


def import_ijxyz_ascii(self, mfile):  # pylint: disable=too-many-locals
    """Import OW/DSG IJXYZ ascii format."""

    # import of seismic column system on the form:
    # 2588	1179	476782.2897888889	6564025.6954	1000.0
    # 2588	1180	476776.7181777778	6564014.5058	1000.0

    logger.debug("Read data from file... (scan for dimensions)")

    xlist = _cxtgeo.surf_import_ijxyz(mfile, 0, 1, 1, 1, 0, DEBUG)

    ier, ncol, nrow, _ndef, xori, yori, xinc, yinc, rot, iln, xln, val, yflip = xlist

    if ier != 0:
        raise RuntimeError("Import from C is wrong...")

    # now real read mode
    xlist = _cxtgeo.surf_import_ijxyz(mfile, 1, ncol, nrow, ncol * nrow, 0, DEBUG)

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


def import_ijxyz_ascii_tmpl(self, mfile, template):
    """Import OW/DSG IJXYZ ascii format, with a Cube or RegularSurface
    instance as template."""

    if isinstance(template, (xtgeo.cube.Cube, xtgeo.surface.RegularSurface)):
        logger.info("OK template")
    else:
        raise ValueError("Template is of wrong type: {}".format(type(template)))

    nxy = template.ncol * template.nrow
    _iok, val = _cxtgeo.surf_import_ijxyz_tmpl(
        mfile, template.ilines, template.xlines, nxy, 0, DEBUG
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
