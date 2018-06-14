# -*- coding: utf-8 -*-
"""Export RegularSurface data."""
from __future__ import division, absolute_import
from __future__ import print_function

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')

if xtg_verbose_level < 0:
    xtg_verbose_level = 0


def export_irap_ascii(self, mfile):

    zmin = self.values.min()
    zmax = self.values.max()

    vals = self.get_values1d(fill_value=self.undef)
    print('SHAPE', vals.shape, vals.dtype)

    ier = _cxtgeo.surf_export_irap_ascii(mfile, self._ncol, self._nrow,
                                         self._xori, self._yori,
                                         self._xinc, self._yinc,
                                         self._rotation, vals,
                                         zmin, zmax, 0,
                                         xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Export to Irap Ascii went wrong, '
                           'code is {}'.format(ier))

    del vals


def export_irap_binary(self, mfile):

    ier = _cxtgeo.surf_export_irap_bin(mfile, self._ncol, self._nrow,
                                       self._xori,
                                       self._yori, self._xinc, self._yinc,
                                       self._rotation, self.get_zval(), 0,
                                       xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Export to Irap Binary went wrong, '
                           'code is {}'.format(ier))


def export_zmap_ascii(self, mfile):

    if abs(self.rotation) > 1.0e-20:
        self.unrotate()

    zmin = self.values.min()
    zmax = self.values.max()

    ier = _cxtgeo.surf_export_zmap_ascii(mfile, self._ncol, self._nrow,
                                         self._xori, self._yori,
                                         self._xinc, self._yinc,
                                         self.get_zval(),
                                         zmin, zmax, 0,
                                         xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Export to ZMAP Ascii went wrong, '
                           'code is {}'.format(ier))


def export_storm_binary(self, mfile):

    if abs(self.rotation) > 1.0e-20:
        self.unrotate()

    zmin = self.values.min()
    zmax = self.values.max()

    ier = _cxtgeo.surf_export_storm_bin(mfile, self._ncol, self._nrow,
                                        self._xori, self._yori,
                                        self._xinc, self._yinc,
                                        self.get_zval(),
                                        zmin, zmax, 0,
                                        xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Export to Storm binary went wrong, '
                           'code is {}'.format(ier))
