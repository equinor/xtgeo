"""Export RegularSurface data."""
import logging
import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)


def export_irap_ascii(surf, mfile):

    zmin = surf.values.min()
    zmax = surf.values.max()

    xtg_verbose_level = xtg.get_syslevel()
    _cxtgeo.xtg_verbose_file('NONE')
    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    ier = _cxtgeo.surf_export_irap_ascii(mfile, surf._ncol, surf._nrow,
                                         surf._xori, surf._yori,
                                         surf._xinc, surf._yinc,
                                         surf._rotation, surf.get_zval(),
                                         zmin, zmax, 0,
                                         xtg_verbose_level)
    if ier != 0:
        raise RuntimeError('Export to Irap Ascii went wrong, '
                           'code is {}'.format(ier))


def export_irap_binary(surf, mfile):

    # update numpy to c_array
    surf._update_cvalues()

    xtg_verbose_level = xtg.get_syslevel()
    _cxtgeo.xtg_verbose_file('NONE')

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    ier = _cxtgeo.surf_export_irap_bin(mfile, surf._ncol, surf._nrow,
                                       surf._xori,
                                       surf._yori, surf._xinc, surf._yinc,
                                       surf._rotation, surf.get_zval(), 0,
                                       xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Export to Irap Ascii went wrong, '
                           'code is {}'.format(ier))
