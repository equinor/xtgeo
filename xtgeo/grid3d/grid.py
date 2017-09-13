# -*- coding: utf-8 -*-
"""Module/class for 3D grids with XTGeo."""

from __future__ import print_function

import sys
import inspect
import numpy as np

import errno
import os
import os.path
import logging

import cxtgeo.cxtgeo as _cxtgeo

import re
from tempfile import mkstemp
from xtgeo.common import XTGeoDialog

from .grid3d import Grid3D
from .grid_property import GridProperty
from .grid_properties import GridProperties

from xtgeo.grid3d import _hybridgrid

# =============================================================================
# Class constructor
# =============================================================================


class Grid(Grid3D):
    """
    Class for a 3D grid geometry (corner point) with optionally props,
    i.e. the grid cells and active cell indicator.

    The grid geometry class instances are normally created when
    importing a grid from file, as it is (currently) too complex to create from
    scratch.

    See also the GridProperty() and the GridProperties() classes.

    Example:
        geo = Grid()
        geo.from_file('myfile.roff')
        # alternative:
        geo = Grid('myfile.roff')

    """

    def __init__(self, *args, **kwargs):
        """
        The __init__ (constructor) method of the XTGeo Grid class.
        """

        # logging settings
        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._nx = 4
        self._ny = 3
        self._ny = 5
        self._nsubs = 0
        self._p_coord_v = None  # carray swig pointer to coordinates vector
        self._p_zcorn_v = None  # carray swig pointer to zcorns vector
        self._p_actnum_v = None  # carray swig pointer to actnum vector
        self._nactive = -999  # Number of active cells

        self._props = []  # List of 'attached' property objects

        # perhaps undef should be a class variable, not an instance variables?
        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

        self._xtg = XTGeoDialog()

        if len(args) == 1:
            # make an instance directly through import of a file
            fformat = kwargs.get('fformat', 'roff')
            initprops = kwargs.get('initprops', None)
            restartprops = kwargs.get('restartprops', None)
            restartdates = kwargs.get('restartdates', None)
            self.from_file(args[0], fformat=fformat, initprops=initprops,
                           restartprops=restartprops,
                           restartdates=restartdates)

    # =========================================================================
    # Properties:
    # =========================================================================

    @property
    def nx(self):
        """Returns the NX (row) number of cells."""
        return self._nx

    @nx.setter
    def nx(self, value):
        self.logger.warning('Cannot change the nx property')

    @property
    def ny(self):
        """Returns the NY (column) number of cells."""
        return self._ny

    @ny.setter
    def ny(self, value):
        self.logger.warning('Cannot change the ny property')

    @property
    def nz(self):
        """Returns the NZ (layers) number of cells."""
        return self._nz

    @nz.setter
    def nz(self, value):
        self.logger.warning('Cannot change the nz property')

    @property
    def nactive(self):
        """Returns the number of active cells."""
        return self._nactive

    @property
    def ntotal(self):
        """Returns the total number of cells."""
        return self._nx * self._ny * self._nz

    @property
    def props(self):
        """
        Returns or sets a list of property objects.

        When setting, the dimension of the property object is checked,
        and will raise an IndexError if it does not match the grid.

        """
        return self._props

    @props.setter
    def props(self, list):
        for l in list:
            if l.nx != self._nx or l.ny != self._ny or l.nz != self._nz:
                raise IndexError('Property NX NY NZ <{}> does not match grid!'
                                 .format(l.name))

        self._props = list

    @property
    def propnames(self):
        """
        Returns a list of property names to are hooked to a grid object.
        """

        plist = []
        for obj in self._props:
            plist.append(obj.name)

        return plist

    @property
    def undef(self):
        """
        Get the undef value for floats or ints numpy arrays.
        """
        return self._undef

    @property
    def undef_limit(self):
        """
        Returns the undef limit number, which is slightly less than the
        undef value.

        Hence for numerical precision, one can force undef values
        to a given number, e.g.::

           x[x<x.undef_limit]=999

        Undef limit values cannot be changed.

        """
        return self._undef_limit

    # =========================================================================
    # Other setters and getters as _functions_
    # =========================================================================

    def get_prop_by_name(self, name):
        """
        Gets a propert object by looking for name, return None if not present.
        """
        for obj in self.props:
            if obj.name == name:
                return obj

        return None

# =========================================================================
# Import and export
# =========================================================================

    def from_file(self, gfile,
                  fformat='guess',
                  initprops=None,
                  restartprops=None,
                  restartdates=None):
        """
        Import grid geometry from file, and makes an instance of this class.

        If file extension is missing, then the extension is guessed by fformat
        key, e.g. fformat egrid will guess to '.EGRID'. The 'eclipserun' will
        try to input INIT and UNRST file in addition the grid in 'one go'.

        Arguments:
            gfile (str): File name to be imported
            fformat (str): File format egrid/grid/roff/grdecl/eclipse_run
                (roff is default)
            initprops (str list): Optional, if given, and file format
                is 'eclipse_run', then list the names of the properties here.
            restartprops (str list): Optional, see initprops
            restartdates (int list): Optional, required if restartprops

        Example::

            >>> myfile = ../../testdata/Zone/gullfaks.roff
            >>> xg = Grid()
            >>> xg.from_file(myfile, fformat='roff')

        Raises:
            OSError: if file is not found etc
        """

        fflist = ['egrid', 'grid', 'grdecl', 'roff', 'eclipserun', 'guess']
        if fformat not in fflist:
            raise ValueError('Invalid fformat: <{}>, options are {}'.
                             format(fformat, fflist))

        # work on file extension
        froot, fext = os.path.splitext(gfile)
        fext = fext.replace('.', '')
        fext = fext.lower()

        if fformat == 'guess':
            self.logger.info('Format is <guess>')
            fflist = ['egrid', 'grid', 'grdecl', 'roff', 'eclipserun']
            if fext and fext in fflist:
                fformat = fext

        if not fext:
            # file extension is missing, guess from format
            self.logger.info('File extension missing; guessing...')
            useext = ''
            if fformat == 'egrid':
                useext = '.EGRID'
            elif fformat == 'grid':
                useext = '.GRID'
            elif fformat == 'grdecl':
                useext = '.grdecl'
            elif fformat == 'roff':
                useext = '.roff'
            elif fformat == 'guess':
                raise ValueError('Cannot guess format without file extension')

            gfile = froot + useext

        self.logger.info('File name to be used is {}'.format(gfile))

        test_gfile = gfile
        if fformat == 'eclipserun':
            test_gfile = gfile + '.EGRID'

        if os.path.isfile(test_gfile):
            self.logger.info('File {} exists OK'.format(test_gfile))
        else:
            self.logger.critical('No such file: {}'.format(test_gfile))
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), gfile)

        if (fformat == 'roff'):
            self._import_roff(gfile)
        elif (fformat == 'grid'):
            self._import_ecl_output(gfile, 0)
        elif (fformat == 'egrid'):
            self._import_ecl_output(gfile, 2)
        elif (fformat == 'eclipserun'):
            self._import_ecl_run(gfile, initprops=initprops,
                                 restartprops=restartprops,
                                 restartdates=restartdates,
                                 )
        elif (fformat == 'grdecl'):
            self._import_ecl_grdecl(gfile)
        else:
            self.logger.warning('Invalid file format')
            sys.exit(1)

        return self

    def to_file(self, gfile, fformat='roff'):
        """
        Export grid geometry to file (roff supported).

        Example:
            g.to_file('myfile.roff')
        """

        if fformat == 'roff' or fformat == 'roff_binary':
            self._export_roff(gfile, 0)
        elif fformat == 'roff_ascii':
            self._export_roff(gfile, 1)

# =========================================================================
# Get some grid basics
# =========================================================================
    def get_cactnum(self):
        """
        Returns the C pointer to the ACTNUM array, to be used as input for
        reading INIT and RESTART.
        """
        return self._p_actnum_v  # the SWIG pointer to the C structure

    def get_actnum(self, name='ACTNUM'):
        """
        Return an ACTNUM GridProperty object

        Arguments:
            name: name of property
        """

        ntot = self._nx * self._ny * self._nz
        act = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                           values=np.zeros(ntot, dtype=np.int32),
                           name=name, discrete=True)

        act._cvalues = self._p_actnum_v  # the SWIG pointer to the C structure
        act._update_values()
        act._codes = {0: '0', 1: '1'}
        act._ncodes = 2

        # return the object
        return act

    def get_dz(self, name='dZ', flip=True, mask=True):
        """
        Return the dZ as GridProperty object.

        The dZ is computed as an average height of the vertical pillars in
        each cell, projected to vertical dimension.

        Args:
            name (str): name of property
            flip (bool): Use False for Petrel grids (experimental)
            mask (bool): True if only for active cells, False for all cells

        Returns:
            A xtgeo GridProperty object
        """

        ntot = self._nx * self._ny * self._nz
        dz = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                          values=np.zeros(ntot, dtype=np.float64),
                          name=name, discrete=False)

        ptr_dz_v = _cxtgeo.new_doublearray(self.ntotal)

        nflip = 1
        if not flip:
            nflip = -1

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        option = 0
        if mask:
            option = 1

        _cxtgeo.grd3d_calc_dz(
            self._nx, self._ny, self._nz, self._p_zcorn_v,
            self._p_actnum_v, ptr_dz_v, nflip, option,
            xtg_verbose_level)

        dz._cvalues = ptr_dz_v
        dz._update_values()

        # return the property object
        return dz

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get X Y Z as properties
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_xyz(self, names=['X', 'Y', 'Z'], mask=True):
        """
        Return 3 GridProperty objects: x coordinate, ycoordinate, z coordinate.

        The values are mid cell values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates). However, the option mask=True will mask the numpies
        for undef cells.

        Arguments:
            names: a list of names per property
            mask:
        """

        ntot = self.ntotal

        x = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                         values=np.zeros(ntot, dtype=np.float64),
                         name=names[0], discrete=False)

        y = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                         values=np.zeros(ntot, dtype=np.float64),
                         name=names[1], discrete=False)

        z = GridProperty(nx=self._nx, ny=self._ny, nz=self._nz,
                         values=np.zeros(ntot, dtype=np.float64),
                         name=names[2], discrete=False)

        ptr_x_v = _cxtgeo.new_doublearray(self.ntotal)
        ptr_y_v = _cxtgeo.new_doublearray(self.ntotal)
        ptr_z_v = _cxtgeo.new_doublearray(self.ntotal)

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        option = 0
        if mask:
            option = 1

        _cxtgeo.grd3d_calc_xyz(self._nx, self._ny, self._nz, self._p_coord_v,
                               self._p_zcorn_v, self._p_actnum_v,
                               ptr_x_v, ptr_y_v, ptr_z_v,
                               option, xtg_verbose_level)

        x._cvalues = ptr_x_v
        y._cvalues = ptr_y_v
        z._cvalues = ptr_z_v

        x._update_values()
        y._update_values()
        z._update_values()

        # return the objects
        return x, y, z

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get grid geometrics
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_geometrics(self, allcells=False, cellcenter=True):
        """
        Get a list of grid geometrics such as origin, minimum, maximum, average
        rotation, etc.

        This return list is (xori, yori, zori, xmin, xmax, ymin, ymax, zmin,
        zmax, avg_rotation, avg_dx, avg_dy, avg_dz, grid_regularity_flag)

        Input:
        allcells=True: Use all cells (also inactive)
        cellcenter=True: Use cell center coordinates, not corners
        """

        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        ptr_x = []
        for i in range(13):
            ptr_x.append(_cxtgeo.new_doublepointer())

        option1 = 1
        if allcells:
            option1 = 0

        option2 = 1
        if cellcenter:
            option2 = 0

        quality = _cxtgeo.grd3d_geometrics(self._nx, self._ny, self._nz,
                                           self._p_coord_v, self._p_zcorn_v,
                                           self._p_actnum_v, ptr_x[0],
                                           ptr_x[1], ptr_x[2], ptr_x[3],
                                           ptr_x[4], ptr_x[5], ptr_x[6],
                                           ptr_x[7], ptr_x[8], ptr_x[9],
                                           ptr_x[10], ptr_x[11], ptr_x[12],
                                           option1, option2,
                                           xtg_verbose_level)

        list = []
        for i in range(13):
            list.append(_cxtgeo.doublepointer_value(ptr_x[i]))

        list.append(quality)

        self.logger.info('Cell geometrics done')
        return list

# =============================================================================
# Some more special operations that changes the grid
# =============================================================================

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reduce grid to one single layer (for special purpose)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reduce_to_one_layer(self):
        """
        Reduce the grid to one single single layer.
        """
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        # need new pointers in C (not for coord)

        ptr_new_num_act = _cxtgeo.new_intpointer()
        ptr_new_zcorn_v = _cxtgeo.new_doublearray(
            self._nx * self._ny * (1 + 1) * 4)
        ptr_new_actnum_v = _cxtgeo.new_intarray(self._nx * self._ny * 1)

        _cxtgeo.grd3d_reduce_onelayer(self._nx, self._ny, self._nz,
                                      self._p_zcorn_v,
                                      ptr_new_zcorn_v,
                                      self._p_actnum_v,
                                      ptr_new_actnum_v,
                                      ptr_new_num_act,
                                      0,
                                      xtg_verbose_level)

        self._nz = 1
        self._p_zcorn_v = ptr_new_zcorn_v
        self._p_actnum_v = ptr_new_actnum_v
        self._nactive = _cxtgeo.intpointer_value(ptr_new_num_act)
        self._nsubs = 0
        self._props = []

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Translate coordinates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):
        """
        Translate (move) and/or flip grid coordinates in 3D.

        Inputs are tuples for (X Y Z). The flip must be 1 or -1.
        """

        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        tx, ty, tz = translate
        fx, fy, fz = flip

        ier = _cxtgeo.grd3d_translate(self._nx, self._ny, self._nz,
                                      fx, fy, fz, tx, ty, tz,
                                      self._p_coord_v, self._p_zcorn_v,
                                      xtg_verbose_level)
        if ier != 0:
            raise Exception('Something went wrong in translate')

        self.logger.info('Translation of coords done')

# =============================================================================
# Various methods
# =============================================================================

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Convert to hybrid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def convert_to_hybrid(self, nhdiv=10, toplevel=1000, bottomlevel=1100,
                          region=None, region_number=None):

        res = _hybridgrid.make_hybridgrid(self, nhdiv=nhdiv, toplevel=toplevel,
                                          bottomlevel=bottomlevel,
                                          region=region,
                                          region_number=region_number)

        self = res

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Report well to zone mismatch
    # This works together with a Well object
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def report_zone_mismatch(self, well=None, zonelogname='ZONELOG',
                             mode=0, zoneprop=None, onelayergrid=None,
                             zonelogrange=[0, 9999], zonelogshift=0,
                             depthrange=None, option=0, perflogname=None):
        """
        Reports mismatch between wells and a zone
        """
        this = inspect.currentframe().f_code.co_name

        # first do some trimming of the well dataframe
        if not well:
            self.logger.info('No well object in <{}>; return no result'.
                             format(this))
            return None

        # qperf = True
        if perflogname == 'None' or perflogname is None:
            # qperf = False
            pass
        else:
            if perflogname not in well.lognames:
                self.logger.info(
                    'Ask for perf log <{}> but no such in <{}> for well'
                    ' {}; return'.format(perflogname, this, well.wellname))
                return None

        self.logger.info('Process well object for {}...'.format(well.wellname))
        df = well.dataframe.copy()

        if depthrange:
            self.logger.info('Filter depth...')
            df = df[df.Z_TVDSS > depthrange[0]]
            df = df[df.Z_TVDSS < depthrange[1]]
            df = df.copy()
            self.logger.debug(df)

        self.logger.info('Adding zoneshift {}'.format(zonelogshift))
        if zonelogshift != 0:
            df[zonelogname] += zonelogshift

        self.logger.info('Filter ZONELOG...')
        df = df[df[zonelogname] > zonelogrange[0]]
        df = df[df[zonelogname] < zonelogrange[1]]
        df = df.copy()

        if perflogname:
            self.logger.info('Filter PERF...')
            df[perflogname].fillna(-999, inplace=True)
            df = df[df[perflogname] > 0]
            df = df.copy()

        df.reset_index(drop=True, inplace=True)
        well.dataframe = df

        self.logger.debug(df)

        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        # get the relevant well log C arrays...
        ptr_xc = well.get_carray('X_UTME')
        ptr_yc = well.get_carray('Y_UTMN')
        ptr_zc = well.get_carray('Z_TVDSS')
        ptr_zo = well.get_carray(zonelogname)

        nval = well.nrows

        ptr_results = _cxtgeo.new_doublearray(10)

        ptr_zprop = zoneprop.cvalues

        cstatus = _cxtgeo.grd3d_rpt_zlog_vs_zon(self._nx, self._ny, self._nz,
                                                self._p_coord_v,
                                                self._p_zcorn_v,
                                                self._p_actnum_v, ptr_zprop,
                                                nval, ptr_xc, ptr_yc, ptr_zc,
                                                ptr_zo, zonelogrange[0],
                                                zonelogrange[1],
                                                onelayergrid._p_zcorn_v,
                                                onelayergrid._p_actnum_v,
                                                ptr_results, option,
                                                xtg_verbose_level)

        if cstatus == 0:
            self.logger.debug('OK well')
        elif cstatus == 2:
            self.logger.warn('Well {} have no zonation?'.format(well.wellname))
        else:
            self.logger.critical('Somthing si rotten with {}'.
                                 format(well.wellname))

        # extract the report
        perc = _cxtgeo.doublearray_getitem(ptr_results, 0)
        tpoi = _cxtgeo.doublearray_getitem(ptr_results, 1)
        mpoi = _cxtgeo.doublearray_getitem(ptr_results, 2)

        return [perc, tpoi, mpoi]

# =============================================================================
# PRIVATE METHODS
# should not be applied outside the class!
# =============================================================================

# -----------------------------------------------------------------------------
# Import methods for various formats
# -----------------------------------------------------------------------------

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # import roff binary
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _import_roff(self, gfile):

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.syslevel

        self.logger.info('Working with file {}'.format(gfile))

        self.logger.info('Scanning...')
        ptr_nx = _cxtgeo.new_intpointer()
        ptr_ny = _cxtgeo.new_intpointer()
        ptr_nz = _cxtgeo.new_intpointer()
        ptr_nsubs = _cxtgeo.new_intpointer()

        _cxtgeo.grd3d_scan_roff_bingrid(ptr_nx, ptr_ny, ptr_nz, ptr_nsubs,
                                        gfile, xtg_verbose_level)

        self._nx = _cxtgeo.intpointer_value(ptr_nx)
        self._ny = _cxtgeo.intpointer_value(ptr_ny)
        self._nz = _cxtgeo.intpointer_value(ptr_nz)
        self._nsubs = _cxtgeo.intpointer_value(ptr_nsubs)

        ntot = self._nx * self._ny * self._nz
        ncoord = (self._nx + 1) * (self._ny + 1) * 2 * 3
        nzcorn = self._nx * self._ny * (self._nz + 1) * 4

        self.logger.info('NCOORD {}'.format(ncoord))
        self.logger.info('NZCORN {}'.format(nzcorn))
        self.logger.info('Reading...')

        ptr_num_act = _cxtgeo.new_intpointer()
        self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
        self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
        self._p_actnum_v = _cxtgeo.new_intarray(ntot)
        self._p_subgrd_v = _cxtgeo.new_intarray(self._nsubs)

        _cxtgeo.grd3d_import_roff_grid(ptr_num_act, ptr_nsubs, self._p_coord_v,
                                       self._p_zcorn_v, self._p_actnum_v,
                                       self._p_subgrd_v, self._nsubs, gfile,
                                       xtg_verbose_level)

        self._nactive = _cxtgeo.intpointer_value(ptr_num_act)

        self.logger.info('Number of active cells: {}'.format(self.nactive))
        self.logger.info('Number of subgrids: {}'.format(self._nsubs))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # import eclipse output .GRID
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _import_ecl_output(self, gfile, gtype):

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.syslevel

        # gtype=0 GRID, gtype=1 FGRID, 2=EGRID, 3=FEGRID ...not all supported
        if gtype == 1 or gtype == 3:
            self.logger.error(
                'Other than GRID and EGRID format not supported'
                ' yet. Return')
            return

        self.logger.info('Working with file {}'.format(gfile))

        self.logger.info('Scanning...')
        ptr_nx = _cxtgeo.new_intpointer()
        ptr_ny = _cxtgeo.new_intpointer()
        ptr_nz = _cxtgeo.new_intpointer()

        if gtype == 0:
            _cxtgeo.grd3d_scan_ecl_grid_hd(gtype, ptr_nx, ptr_ny, ptr_nz,
                                           gfile, xtg_verbose_level)
        elif gtype == 2:
            _cxtgeo.grd3d_scan_ecl_egrid_hd(gtype, ptr_nx, ptr_ny, ptr_nz,
                                            gfile, xtg_verbose_level)

        self._nx = _cxtgeo.intpointer_value(ptr_nx)
        self._ny = _cxtgeo.intpointer_value(ptr_ny)
        self._nz = _cxtgeo.intpointer_value(ptr_nz)

        self.logger.info('NX NY NZ {} {} {}'.format(self._nx, self._ny,
                                                    self._nz))

        ntot = self._nx * self._ny * self._nz
        ncoord = (self._nx + 1) * (self._ny + 1) * 2 * 3
        nzcorn = self._nx * self._ny * (self._nz + 1) * 4

        self.logger.info('NTOT NCCORD NZCORN {} {} {}'.format(ntot, ncoord,
                                                              nzcorn))

        self.logger.info('Reading... ncoord is {}'.format(ncoord))

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
            _cxtgeo.grd3d_import_ecl_egrid(0, self._nx, self._ny, self._nz,
                                           ptr_num_act,
                                           self._p_coord_v, self._p_zcorn_v,
                                           self._p_actnum_v, gfile,
                                           xtg_verbose_level)

        nact = _cxtgeo.intpointer_value(ptr_num_act)
        self._nactive = nact

        self.logger.info('Number of active cells: {}'.format(nact))
        self._nsubs = 0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Import eclipse run suite: EGRID + properties from INIT and UNRST
    # For the INIT and UNRST, props dates shall be selected
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _import_ecl_run(self, groot, initprops=None,
                        restartprops=None, restartdates=None):

        ecl_grid = groot + '.EGRID'
        ecl_init = groot + '.INIT'
        ecl_rsta = groot + '.UNRST'

        # import the grid
        self._import_ecl_output(ecl_grid, 2)

        # import the init properties unless list is empty
        if initprops:
            initprops = GridProperties()
            initprops.from_file(ecl_init, name=name, fformat='init', date=None,
                                grid=self)
            for p in initprops.props:
                self._props.append(p)

        # import the restart properties for dates unless lists are empty
        if restartprops and restartdates:
            restprops = GridProperties()
            restprops.from_file(ecl_rsta, names=restartprops,
                                fformat='unrst', dates=restartdates,
                                grid=self)
            for p in restprops.props:
                self._props.append(p)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # import eclipse input .GRDECL
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _import_ecl_grdecl(self, gfile):

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')

        xtg_verbose_level = self._xtg.syslevel

        # make a temporary file
        fd, tmpfile = mkstemp()
        # make a temporary

        with open(gfile) as oldfile, open(tmpfile, 'w') as newfile:
            for line in oldfile:
                if not (re.search(r'^--', line) or re.search(r'^\s+$', line)):
                    newfile.write(line)

        newfile.close()
        oldfile.close()

        # find nx ny nz
        mylist = []
        found = False
        with open(tmpfile) as xfile:
            for line in xfile:
                if (found):
                    self.logger.info(line)
                    mylist = line.split()
                    break
                if re.search(r'^SPECGRID', line):
                    found = True

        if not found:
            self.logger.error('SPECGRID not found. Nothing imported!')
            return
        xfile.close()

        self._nx, self._ny, self._nz = \
            int(mylist[0]), int(mylist[1]), int(mylist[2])

        self.logger.info('NX NY NZ in grdecl file: {} {} {}'.format(self._nx,
                                                                    self._ny,
                                                                    self._nz))

        ntot = self._nx * self._ny * self._nz
        ncoord = (self._nx + 1) * (self._ny + 1) * 2 * 3
        nzcorn = self._nx * self._ny * (self._nz + 1) * 4

        self.logger.info('Reading...')

        ptr_num_act = _cxtgeo.new_intpointer()
        self._p_coord_v = _cxtgeo.new_doublearray(ncoord)
        self._p_zcorn_v = _cxtgeo.new_doublearray(nzcorn)
        self._p_actnum_v = _cxtgeo.new_intarray(ntot)

        _cxtgeo.grd3d_import_grdecl(self._nx,
                                    self._ny,
                                    self._nz,
                                    self._p_coord_v,
                                    self._p_zcorn_v,
                                    self._p_actnum_v,
                                    ptr_num_act,
                                    tmpfile,
                                    xtg_verbose_level,
                                    )

        # remove tmpfile
        os.close(fd)
        os.remove(tmpfile)

        nact = _cxtgeo.intpointer_value(ptr_num_act)
        self._nactive = nact

        self.logger.info('Number of active cells: {}'.format(nact))
        self._nsubs = 0

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # export ROFF
    # option = 0 binary; option = 1 ascii
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _export_roff(self, gfile, option):

        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        if self._nsubs == 0 and not hasattr(self, '_p_subgrd_v'):
            self.logger.debug('Create a pointer for _p_subgrd_v ...')
            self._p_subgrd_v = _cxtgeo.new_intpointer()

        # get the geometrics list to find the xshift, etc
        gx = self.get_geometrics()

        _cxtgeo.grd3d_export_roff_grid(option, self._nx, self._ny, self._nz,
                                       self._nsubs, 0, gx[3], gx[5], gx[7],
                                       self._p_coord_v, self._p_zcorn_v,
                                       self._p_actnum_v, self._p_subgrd_v,
                                       gfile, xtg_verbose_level)

        # skip parameters for now (cf Perl code)

        # end tag
        _cxtgeo.grd3d_export_roff_end(option, gfile, xtg_verbose_level)


# =============================================================================
# Some private helper methods
# =============================================================================

    # copy (update) values from SWIG carray to numpy, 1D array
    # CHECK THIS!
    def _update_values(self):
        n = self._nx * self._ny * self._nz
        if not self._isdiscrete:
            # x = _cxtgeo.swig_carr_to_numpy_1d(n, self._cvalues)

            self._undef = _cxtgeo.UNDEF
            self._undef_limit = _cxtgeo.UNDEF_INT_LIMIT

            self.mask_undef()

        else:
            self._values = _cxtgeo.swig_carr_to_numpy_i1d(n, self._cvalues)

            self._undef = _cxtgeo.UNDEF_INT
            self._undef_limit = _cxtgeo.UNDEF_INT_LIMIT

            # make it int32 (not as RMS?) and mask it
            self._values = self._values.astype(np.int32)
            self.mask_undef()

    # copy (update) values from numpy to SWIG, 1D array
    def _update_cvalues(self):
        if self._ptype == 1:
            _cxtgeo.swig_numpy_to_carr_1d(self._values, self._cvalues)
        else:
            _cxtgeo.swig_numpy_to_carr_i1d(self._values, self._cvalues)


# =============================================================================
# MAIN, for initial testing. Run from current directory
# =============================================================================
def main():

    gfile = '../../../../testdata/Zone/emerald_hetero_grid.roff'

    gg = Grid()
    gg.from_file(gfile, fformat='roff')

    xtg = XTGeoDialog()

    xtg.info('nx ny nz {} {} {}'.format(gg.nx, gg.ny, gg.nz))


if __name__ == '__main__':
    main()
