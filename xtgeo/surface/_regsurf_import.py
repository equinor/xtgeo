"""Import RegularSurface data."""
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_cxtgeo.xtg_verbose_file('NONE')

xtg = XTGeoDialog()
xtg_verbose_level = xtg.get_syslevel()


def import_irap_binary(mfile):

    logger.debug('Enter function...')
    # need to call the C function...
    _cxtgeo.xtg_verbose_file('NONE')

    xtg_verbose_level = xtg.get_syslevel()

    if xtg_verbose_level < 0:
        xtg_verbose_level = 0

    ptr_mx = _cxtgeo.new_intpointer()
    ptr_my = _cxtgeo.new_intpointer()
    ptr_xori = _cxtgeo.new_doublepointer()
    ptr_yori = _cxtgeo.new_doublepointer()
    ptr_xinc = _cxtgeo.new_doublepointer()
    ptr_yinc = _cxtgeo.new_doublepointer()
    ptr_rot = _cxtgeo.new_doublepointer()
    ptr_dum = _cxtgeo.new_doublepointer()
    ptr_ndef = _cxtgeo.new_intpointer()

    # read with mode 0, to get mx my
    _cxtgeo.surf_import_irap_bin(mfile, 0, ptr_mx, ptr_my, ptr_xori,
                                 ptr_yori, ptr_xinc, ptr_yinc, ptr_rot,
                                 ptr_dum, ptr_ndef, 0, xtg_verbose_level)

    mx = _cxtgeo.intpointer_value(ptr_mx)
    my = _cxtgeo.intpointer_value(ptr_my)

    cvalues = _cxtgeo.new_doublearray(mx * my)

    # read with mode 1, to get the map
    _cxtgeo.surf_import_irap_bin(mfile, 1, ptr_mx, ptr_my, ptr_xori,
                                 ptr_yori, ptr_xinc, ptr_yinc, ptr_rot,
                                 cvalues, ptr_ndef, 0,
                                 xtg_verbose_level)

    sdata = dict()

    sdata['ncol'] = _cxtgeo.intpointer_value(ptr_mx)
    sdata['nrow'] = _cxtgeo.intpointer_value(ptr_my)
    sdata['xori'] = _cxtgeo.doublepointer_value(ptr_xori)
    sdata['yori'] = _cxtgeo.doublepointer_value(ptr_yori)
    sdata['xinc'] = _cxtgeo.doublepointer_value(ptr_xinc)
    sdata['yinc'] = _cxtgeo.doublepointer_value(ptr_yinc)
    sdata['rotation'] = _cxtgeo.doublepointer_value(ptr_rot)
    sdata['cvalues'] = cvalues

    return sdata
