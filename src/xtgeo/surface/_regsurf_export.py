# -*- coding: utf-8 -*-
"""Export RegularSurface data."""

# pylint: disable=protected-access
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo
import xtgeo.cxtgeo.cxtgeo as _cxtgeo  # pylint: disable=import-error
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


def export_irap_binary(self, mfile):
    """Export to Irap RMS binary format."""

    fout = xtgeo._XTGeoCFile(mfile, mode="wb")

    vals = self.get_values1d(fill_value=xtgeo.UNDEF)
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
        raise RuntimeError(
            "Export to Irap Binary went wrong, " "code is {}".format(ier)
        )

    fout.close()


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
