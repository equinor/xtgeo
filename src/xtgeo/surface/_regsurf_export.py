"""Export RegularSurface data."""

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def export_irap_ascii(self, mfile):

    zmin = self.values.min()
    zmax = self.values.max()

    xtg_verbose_level = xtg.get_syslevel()
    _cxtgeo.xtg_verbose_file('NONE')
    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    ier = _cxtgeo.surf_export_irap_ascii(mfile, self._ncol, self._nrow,
                                         self._xori, self._yori,
                                         self._xinc, self._yinc,
                                         self._rotation, self.get_zval(),
                                         zmin, zmax, 0,
                                         xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Export to Irap Ascii went wrong, '
                           'code is {}'.format(ier))


def export_irap_binary(self, mfile):

    # update numpy to c_array
    self._update_cvalues()

    xtg_verbose_level = xtg.get_syslevel()
    _cxtgeo.xtg_verbose_file('NONE')

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    ier = _cxtgeo.surf_export_irap_bin(mfile, self._ncol, self._nrow,
                                       self._xori,
                                       self._yori, self._xinc, self._yinc,
                                       self._rotation, self.get_zval(), 0,
                                       xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Export to Irap Ascii went wrong, '
                           'code is {}'.format(ier))
