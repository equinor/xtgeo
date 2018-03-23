"""GridProperty (not GridProperies) export functions"""

from __future__ import print_function, absolute_import

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import _gridprop_lowlevel

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Export binary ROFF format (NB Int NOT supported YET)!
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def export_roff(self, pfile, name):

    self.logger.debug('Exporting {} to file {}'.format(name, pfile))

    carray = _gridprop_lowlevel.update_carray(self)

    ptr_idum = _cxtgeo.new_intpointer()

    mode = 0  # binary
    if not self._isdiscrete:
        _cxtgeo.grd3d_export_roff_pstart(mode, self._ncol, self._nrow,
                                         self._nlay, pfile,
                                         xtg_verbose_level)

    # now the actual data
    # only float data are supported for now!
    nsub = 0
    isub_to_export = 0
    if not self._isdiscrete:
        _cxtgeo.grd3d_export_roff_prop(mode, self._ncol, self._nrow,
                                       self._nlay, nsub, isub_to_export,
                                       ptr_idum, name, 'double', ptr_idum,
                                       carray, 0, '',
                                       ptr_idum, pfile, xtg_verbose_level)
    else:
        self.logger.critical('INT export not supported yet')
        raise NotImplementedError('INT export not supported yet')

    _cxtgeo.grd3d_export_roff_end(mode, pfile, xtg_verbose_level)

    _gridprop_lowlevel.delete_carray(self, carray)
