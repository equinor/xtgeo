# coding: utf-8
"""Private module, Grid Import private functions"""

from __future__ import print_function, absolute_import

import re
import os
from tempfile import mkstemp
from collections import OrderedDict
import numpy as np

import xtgeo.cxtgeo.cxtgeo as _cxtgeo
import xtgeo

xtg = xtgeo.common.XTGeoDialog()

logger = xtg.functionlogger(__name__)

_cxtgeo.xtg_verbose_file('NONE')
xtg_verbose_level = xtg.get_syslevel()

#
# NOTE:
# self is the xtgeo.grid3d.Grid instance
#


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import roff binary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_roff(self, gfile):

    logger.info('Working with file {}'.format(gfile))

    logger.info('Scanning...')
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()
    ptr_nsubs = _cxtgeo.new_intpointer()

    _cxtgeo.grd3d_scan_roff_bingrid(ptr_ncol, ptr_nrow, ptr_nlay,
                                    ptr_nsubs, gfile, xtg_verbose_level)

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)
    nsubs = _cxtgeo.intpointer_value(ptr_nsubs)

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    logger.info('NCOORD {}'.format(ncoord))
    logger.info('NZCORN {}'.format(nzcorn))
    logger.info('Reading...')

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)
    subgrd_v = _cxtgeo.new_intarray(nsubs)

    _cxtgeo.grd3d_import_roff_grid(ptr_num_act, ptr_nsubs, self._p_coord_v,
                                   self._p_zcorn_v, self._p_actnum_v,
                                   subgrd_v, nsubs, gfile,
                                   xtg_verbose_level)

    logger.info('Number of active cells: {}'.format(self.nactive))
    logger.info('Number of subgrids: {}'.format(nsubs))

    if nsubs > 1:
        self._subgrids = OrderedDict()
        prev = 1
        for irange in range(nsubs):
            val = _cxtgeo.intarray_getitem(subgrd_v, irange)

            logger.info('VAL is %s', val)
            logger.info('RANGE is %s', range(prev, val + prev))
            self._subgrids['subgrid_' + str(irange)] = range(prev, val + prev)
            prev = val + prev
    else:
        self._subgrids = None

    logger.info('Subgrids array %s', self._subgrids)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import eclipse output .GRID
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_output(self, gfile, gtype):

    # gtype=0 GRID, gtype=1 FGRID, 2=EGRID, 3=FEGRID ...not all supported
    if gtype == 1 or gtype == 3:
        logger.error(
            'Other than GRID and EGRID format not supported'
            ' yet. Return')
        return

    logger.info('Working with file {}'.format(gfile))

    logger.info('Scanning...')
    ptr_ncol = _cxtgeo.new_intpointer()
    ptr_nrow = _cxtgeo.new_intpointer()
    ptr_nlay = _cxtgeo.new_intpointer()

    if gtype == 0:
        _cxtgeo.grd3d_scan_ecl_grid_hd(gtype, ptr_ncol, ptr_nrow, ptr_nlay,
                                       gfile, xtg_verbose_level)
    elif gtype == 2:
        _cxtgeo.grd3d_scan_ecl_egrid_hd(gtype, ptr_ncol, ptr_nrow,
                                        ptr_nlay, gfile, xtg_verbose_level)

    self._ncol = _cxtgeo.intpointer_value(ptr_ncol)
    self._nrow = _cxtgeo.intpointer_value(ptr_nrow)
    self._nlay = _cxtgeo.intpointer_value(ptr_nlay)

    logger.info('NX NY NZ {} {} {}'.format(self._ncol, self._nrow,
                                           self._nlay))

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    logger.info('NTOT NCCORD NZCORN {} {} {}'.format(ntot, ncoord,
                                                     nzcorn))

    logger.info('Reading... ncoord is {}'.format(ncoord))

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    if gtype == 0:
        # GRID
        _cxtgeo.grd3d_import_ecl_grid(0, ntot, ptr_num_act,
                                      self._p_coord_v, self._p_zcorn_v,
                                      self._p_actnum_v, gfile,
                                      xtg_verbose_level)
    elif gtype == 2:
        # EGRID
        _cxtgeo.grd3d_import_ecl_egrid(0, self._ncol, self._nrow,
                                       self._nlay, ptr_num_act,
                                       self._p_coord_v, self._p_zcorn_v,
                                       self._p_actnum_v, gfile,
                                       xtg_verbose_level)

    nact = _cxtgeo.intpointer_value(ptr_num_act)

    logger.info('Number of active cells: {}'.format(nact))
    self._subgrids = None


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import eclipse run suite: EGRID + properties from INIT and UNRST
# For the INIT and UNRST, props dates shall be selected
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_run(self, groot, initprops=None,
                   restartprops=None, restartdates=None):

    ecl_grid = groot + '.EGRID'
    ecl_init = groot + '.INIT'
    ecl_rsta = groot + '.UNRST'

    # import the grid
    import_ecl_output(self, ecl_grid, 2)

    # import the init properties unless list is empty
    if initprops:
        iprops = xtgeo.grid3d.GridProperties()
        iprops.from_file(ecl_init, names=initprops, fformat='init',
                         dates=None, grid=self)

        for prop in iprops.props:
            self._props.append(prop)

    # import the restart properties for dates unless lists are empty
    if restartprops and restartdates:
        restprops = xtgeo.grid3d.GridProperties()
        restprops.from_file(ecl_rsta, names=restartprops,
                            fformat='unrst', dates=restartdates,
                            grid=self)
        for prop in restprops.props:
            self._props.append(prop)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import eclipse input .GRDECL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def import_ecl_grdecl(self, gfile):

    # make a temporary file
    fds, tmpfile = mkstemp()
    # make a temporary

    with open(gfile) as oldfile, open(tmpfile, 'w') as newfile:
        for line in oldfile:
            if not (re.search(r'^--', line) or re.search(r'^\s+$', line)):
                newfile.write(line)

    newfile.close()
    oldfile.close()

    # find ncol nrow nz
    mylist = []
    found = False
    with open(tmpfile) as xfile:
        for line in xfile:
            if found:
                logger.info(line)
                mylist = line.split()
                break
            if re.search(r'^SPECGRID', line):
                found = True

    if not found:
        logger.error('SPECGRID not found. Nothing imported!')
        return
    xfile.close()

    self._ncol, self._nrow, self._nlay = \
        int(mylist[0]), int(mylist[1]), int(mylist[2])

    logger.info('NX NY NZ in grdecl file: {} {} {}'
                .format(self._ncol, self._nrow, self._nlay))

    ntot = self._ncol * self._nrow * self._nlay
    ncoord = (self._ncol + 1) * (self._nrow + 1) * 2 * 3
    nzcorn = self._ncol * self._nrow * (self._nlay + 1) * 4

    logger.info('Reading...')

    ptr_num_act = _cxtgeo.new_intpointer()
    self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
    self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
    self._p_actnum_v = _cxtgeo.new_intarray(ntot)

    _cxtgeo.grd3d_import_grdecl(self._ncol,
                                self._nrow,
                                self._nlay,
                                self._p_coord_v,
                                self._p_zcorn_v,
                                self._p_actnum_v,
                                ptr_num_act,
                                tmpfile,
                                xtg_verbose_level)

    # remove tmpfile
    os.close(fds)
    os.remove(tmpfile)

    nact = _cxtgeo.intpointer_value(ptr_num_act)

    logger.info('Number of active cells: {}'.format(nact))
    self._subgrids = None
