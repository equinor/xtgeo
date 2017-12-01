"""Import RegularSurface data."""
import logging
import numpy as np
import numpy.ma as ma

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)
xtg_verbose_level = xtg.get_syslevel()
_cxtgeo.xtg_verbose_file('NONE')




def import_irap_binary(mfile):
    # version using swig type mapping

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    # read with mode 0, to get mx my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 0, 1, 0, xtg_verbose_level)

    nval = xlist[1] * xlist[2]  # mx * my
    xlist = _cxtgeo.surf_import_irap_bin(mfile, 1, nval, 0, xtg_verbose_level)

    ier, ncol, nrow, ndef, xori, yori, xinc, yinc, rot, val = xlist

    val = np.reshape(val, (ncol, nrow), order='F')

    val = ma.masked_greater(val, _cxtgeo.UNDEF_LIMIT)

    if np.isnan(val).any():
        logger.info('NaN values are found, will mask...')
        val = ma.masked_invalid(val)

    sdata = dict()

    sdata['ncol'] = ncol
    sdata['nrow'] = nrow
    sdata['xori'] = xori
    sdata['yori'] = yori
    sdata['xinc'] = xinc
    sdata['yinc'] = yinc
    sdata['rotation'] = rot
    sdata['values'] = val
    sdata['cvalues'] = None

    return sdata
