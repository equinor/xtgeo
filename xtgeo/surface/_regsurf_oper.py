"""Various operations"""
from __future__ import print_function, absolute_import

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
#from xtgeo.surface import RegularSurface

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
if xtg_verbose_level < 0:
    xtg_verbose_level = 0

_cxtgeo.xtg_verbose_file('NONE')


def resample(surf, other):
    """Resample from other surface object to this surf"""

    logger.info('Resampling...')

    svalues = surf.get_zval()

    ier = _cxtgeo.surf_resample(other._ncol, other._nrow,
                                other._xori, other._xinc,
                                other._yori, other._yinc,
                                other._yflip, other._rotation,
                                other.get_zval(),
                                surf._ncol, surf._nrow,
                                surf._xori, surf._xinc,
                                surf._yori, surf._yinc,
                                surf._yflip, surf._rotation,
                                svalues,
                                0,
                                xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Resampling went wrong, '
                           'code is {}'.format(ier))

    surf.set_zval(svalues)
