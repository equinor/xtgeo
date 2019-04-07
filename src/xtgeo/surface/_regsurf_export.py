# -*- coding: utf-8 -*-
"""Export RegularSurface data."""

# pylint: disable=protected-access
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo.cxtgeo.cxtgeo as _cxtgeo  # pylint: disable=import-error
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


DEBUG = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file("NONE")

if DEBUG < 0:
    DEBUG = 0


def export_irap_ascii(self, mfile):
    """Export to Irap RMS ascii format."""
    zmin = self.values.min()
    zmax = self.values.max()

    vals = self.get_values1d(fill_value=self.undef)
    logger.debug("SHAPE %s %s", vals.shape, vals.dtype)

    ier = _cxtgeo.surf_export_irap_ascii(
        mfile,
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
        DEBUG,
    )
    if ier != 0:
        raise RuntimeError("Export to Irap Ascii went wrong, " "code is {}".format(ier))

    del vals


def export_irap_binary(self, mfile):
    """Export to Irap RMS binary format."""

    vals = self.get_values1d(fill_value=self.undef)
    ier = _cxtgeo.surf_export_irap_bin(
        mfile,
        self._ncol,
        self._nrow,
        self._xori,
        self._yori,
        self._xinc,
        self._yflip * self._yinc,
        self._rotation,
        vals,
        0,
        DEBUG,
    )

    if ier != 0:
        raise RuntimeError(
            "Export to Irap Binary went wrong, " "code is {}".format(ier)
        )


def export_ijxyz_ascii(self, mfile):
    """Export to DSG IJXYZ ascii format."""

    vals = self.get_values1d(fill_value=self.undef)
    ier = _cxtgeo.surf_export_ijxyz(
        mfile,
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
        DEBUG,
    )

    if ier != 0:
        raise RuntimeError(
            "Export to IJXYZ format went wrong, " "code is {}".format(ier)
        )


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

    vals = scopy.get_values1d(order="F", asmasked=False, fill_value=self.undef)

    ier = _cxtgeo.surf_export_zmap_ascii(
        mfile,
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
        DEBUG,
    )
    if ier != 0:
        raise RuntimeError("Export to ZMAP Ascii went wrong, " "code is {}".format(ier))
    del scopy


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
        mfile,
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
        DEBUG,
    )
    if ier != 0:
        raise RuntimeError(
            "Export to Storm binary went wrong, " "code is {}".format(ier)
        )
    del scopy
