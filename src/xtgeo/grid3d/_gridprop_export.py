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
# Export ascii or binary ROFF format
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def export_roff(self, pfile, name, append=False, last=True, binary=True):

    logger.debug('Exporting {} to file {}'.format(name, pfile))

    if self._isdiscrete:
        _export_roff_discrete(self, pfile, name, append=append, last=last,
                              binary=binary)
    else:
        _export_roff_continuous(self, pfile, name, append=append, last=last,
                                binary=binary)


def _export_roff_discrete(self, pfile, name, append=False, last=True,
                          binary=True):

    logger.debug('Exporting {} to file {}'.format(name, pfile))

    carray = _gridprop_lowlevel.update_carray(self, undef=-999)

    ptr_idum = _cxtgeo.new_intpointer()
    ptr_ddum = _cxtgeo.new_doublepointer()

    # codes:
    ptr_codes = _cxtgeo.new_intarray(256)
    ncodes = self.ncodes
    codenames = ""
    logger.info(self.codes.keys())
    for inum, ckey in enumerate(sorted(self.codes.keys())):
        codenames += self.codes[ckey]
        codenames += '|'
        _cxtgeo.intarray_setitem(ptr_codes, inum, ckey)

    mode = 0
    if not binary:
        mode = 1

    if not append:
        _cxtgeo.grd3d_export_roff_pstart(mode, self._ncol, self._nrow,
                                         self._nlay, pfile,
                                         xtg_verbose_level)

    nsub = 0
    isub_to_export = 0
    _cxtgeo.grd3d_export_roff_prop(mode, self._ncol, self._nrow,
                                   self._nlay, nsub, isub_to_export,
                                   ptr_idum, name, 'int', carray,
                                   ptr_ddum, ncodes, codenames,
                                   ptr_codes, pfile, xtg_verbose_level)

    if last:
        _cxtgeo.grd3d_export_roff_end(mode, pfile, xtg_verbose_level)

    _gridprop_lowlevel.delete_carray(self, carray)


def _export_roff_continuous(self, pfile, name, append=False, last=True,
                            binary=True):

    logger.debug('Exporting {} to file {}'.format(name, pfile))

    carray = _gridprop_lowlevel.update_carray(self, undef=-999.0)

    ptr_idum = _cxtgeo.new_intpointer()

    mode = 0
    if not binary:
        mode = 1

    if not append:
        _cxtgeo.grd3d_export_roff_pstart(mode, self._ncol, self._nrow,
                                         self._nlay, pfile,
                                         xtg_verbose_level)

    # now the actual data
    nsub = 0
    isub_to_export = 0

    _cxtgeo.grd3d_export_roff_prop(mode, self._ncol, self._nrow,
                                   self._nlay, nsub, isub_to_export,
                                   ptr_idum, name, 'double', ptr_idum,
                                   carray, 0, '',
                                   ptr_idum, pfile, xtg_verbose_level)

    if last:
        _cxtgeo.grd3d_export_roff_end(mode, pfile, xtg_verbose_level)

    _gridprop_lowlevel.delete_carray(self, carray)
