# -*- coding: utf-8 -*-
"""Module/class for 3D grids (corner point geometry) with XTGeo."""

from __future__ import print_function, absolute_import

import errno
import os
import os.path

import numpy as np

import cxtgeo.cxtgeo as _cxtgeo

import xtgeo
from xtgeo.grid3d import Grid3D

from xtgeo.grid3d import _grid_hybrid
from xtgeo.grid3d import _grid_import
from xtgeo.grid3d import _grid_export
from xtgeo.grid3d import _grid_refine
from xtgeo.grid3d import _grid_etc1
from xtgeo.grid3d import _grid_roxapi


class Grid(Grid3D):
    """Class for a 3D grid geometry (corner point) with optionally props.

    I.e. the geometric grid cells and active cell indicator.

    The grid geometry class instances are normally created when
    importing a grid from file, as it is (currently) too complex to create from
    scratch.

    See also the :class:`xtgeo.grid3d.GridProperty` and the
    :class:`.GridProperties` classes.

    Example::

        geo = Grid()
        geo.from_file('myfile.roff')
        #
        # alternative (make instance directly from file):
        geo = Grid('myfile.roff')

    """

    def __init__(self, *args, **kwargs):

        super(Grid, self).__init__(*args, **kwargs)

        self._nsubs = 0
        self._p_coord_v = None       # carray swig pointer to coords vector
        self._p_zcorn_v = None       # carray swig pointer to zcorns vector
        self._p_actnum_v = None      # carray swig pointer to actnum vector
        self._nactive = -999         # Number of active cells
        self._actnum_indices = None  # Index numpy array for active cells

        self._props = []  # List of 'attached' property objects

        # perhaps undef should be a class variable, not an instance variables?
        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

        # Roxar api spesific:
        self._roxgrid = None
        self._roxindexer = None

        if len(args) == 1:
            # make an instance directly through import of a file
            fformat = kwargs.get('fformat', 'guess')
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
    def ncol(self):
        """Cf :py:attr:`.Grid3D.ncol`"""
        return super(Grid, self).ncol

    @property
    def nrow(self):
        """Cf :py:attr:`.Grid3D.nrow`"""
        return super(Grid, self).nrow

    @property
    def nlay(self):
        """Cf :py:attr:`.Grid3D.nlay`"""
        return super(Grid, self).nlay

    @property
    def nactive(self):
        """Returns the number of active cells."""
        return self._nactive

    @property
    def actnum_indices(self):
        """Returns the ndarray which holds the indices for active cells"""
        if self._actnum_indices is None:
            actnum = self.get_actnum()
            self._actnum_indices = np.flatnonzero(actnum.values)

        return self._actnum_indices

    @property
    def ntotal(self):
        """Returns the total number of cells."""
        return self._ncol * self._nrow * self._nlay

    @property
    def props(self):
        """Returns or sets a list of property objects.

        When setting, the dimension of the property object is checked,
        and will raise an IndexError if it does not match the grid.

        """
        return self._props

    @props.setter
    def props(self, plist):
        for litem in plist:
            if litem.ncol != self._ncol or litem.nrow != self._nrow or\
               litem.nlay != self._nlay:
                raise IndexError('Property NX NY NZ <{}> does not match grid!'
                                 .format(litem.name))

        self._props = plist

    @property
    def propnames(self):
        """Returns a list of property names that are hooked to a grid."""

        plist = []
        for obj in self._props:
            plist.append(obj.name)

        return plist

    @property
    def undef(self):
        """Get the undef value for floats or ints numpy arrays."""
        return self._undef

    @property
    def undef_limit(self):
        """Returns the undef limit number - slightly less than the undef value.

        Hence for numerical precision, one can force undef values
        to a given number, e.g.::

           x[x<x.undef_limit]=999

        Undef limit values cannot be changed.
        """

        return self._undef_limit

    @property
    def roxgrid(self):
        """Get the Roxar native proj.grid_models[gname].get_grid() object"""
        return self._roxgrid

    @property
    def roxindexer(self):
        """Get the Roxar native proj.grid_models[gname].get_grid().grid_indexer
        object"""

        return self._roxindexer

    def get_prop_by_name(self, name):
        """Gets a property object by name lookup, return None if not present.
        """
        for obj in self.props:
            if obj.name == name:
                return obj

        return None

    def from_file(self, gfile, fformat='guess', initprops=None,
                  restartprops=None, restartdates=None):

        """Import grid geometry from file, and makes an instance of this class.

        If file extension is missing, then the extension is guessed by fformat
        key, e.g. fformat egrid will be guessed if '.EGRID'. The 'eclipserun'
        will try to input INIT and UNRST file in addition the grid in 'one go'.

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
            >>> # or shorter:
            >>> xg = Grid(myfile)  # will guess the file format

        Raises:
            OSError: if file is not found etc
        """

        fflist = set(['egrid', 'grid', 'grdecl', 'roff', 'eclipserun',
                      'guess'])
        if fformat not in fflist:
            raise ValueError('Invalid fformat: <{}>, options are {}'.
                             format(fformat, fflist))

        # work on file extension
        froot, fext = os.path.splitext(gfile)
        fext = fext.replace('.', '')
        fext = fext.lower()

        self.logger.info('Format is {}'.format(fformat))
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
            _grid_import.import_roff(self, gfile)
        elif (fformat == 'grid'):
            _grid_import.import_ecl_output(self, gfile, 0)
        elif (fformat == 'egrid'):
            _grid_import.import_ecl_output(self, gfile, 2)
        elif (fformat == 'eclipserun'):
            _grid_import.import_ecl_run(self, gfile, initprops=initprops,
                                        restartprops=restartprops,
                                        restartdates=restartdates)
        elif (fformat == 'grdecl'):
            _grid_import.import_ecl_grdecl(self, gfile)
        else:
            raise SystemExit('Invalid file format')

        return self

    def from_roxar(self, projectname, gname):

        """Import grid model geometry from RMS project, and makes an instance.

        Arguments:
            projectname (str): Name of RMS project
            gfile (str): Name of grid model

        """

        _grid_roxapi.import_grid_roxapi(self, projectname, gname)

    def to_file(self, gfile, fformat='roff'):
        """
        Export grid geometry to file (roff binary supported).

        Example::

            g.to_file('myfile.roff')
        """

        if fformat == 'roff' or fformat == 'roff_binary':
            _grid_export.export_roff(self, gfile, 0)
        elif fformat == 'roff_ascii':
            _grid_export.export_roff(self, gfile, 1)
        elif fformat == 'grdecl':
            _grid_export.export_grdecl(self, gfile)
        else:
            raise SystemExit('Invalid file format')

    def get_cactnum(self):
        """Returns the C pointer object reference to the ACTNUM array."""
        return self._p_actnum_v  # the SWIG pointer to the C structure

    def get_indices(self, names=('I', 'J', 'K')):
        """Return 3 GridProperty objects for column, row, and layer index,

        Note that the indexes starts with 1, not zero (i.e. upper
        cell layer is K=1)

        Args:
            names (tuple): Names of the columns (as property names)

        Examples::

            i_index, j_index, k_index = grd.get_indices()

        """

        grd = np.indices((self.ncol, self.nrow, self.nlay))

        ilist = []
        for axis in range(3):
            index = grd[axis]
            index = index.flatten(order='F')
            index = index + 1
            index = index.astype(np.int32)

            idx = xtgeo.grid3d.GridProperty(ncol=self._ncol, nrow=self._nrow,
                                            nlay=self._nlay, values=index,
                                            name=names[axis], discrete=True)
            codes = {}
            ncodes = 0
            for i in range(index.min(), index.max() + 1):
                codes[i] = str(i)
                ncodes = ncodes + 1

            idx._codes = codes
            idx._ncodes = ncodes
            idx._grid = self
            ilist.append(idx)

        return ilist

    def get_actnum(self, name='ACTNUM'):
        """Return an ACTNUM GridProperty object.

        Arguments:
            name (str): name of property in the XTGeo GridProperty object.

        Example::

            act = mygrid.get_actnum()
            print('{}% cells are active'.format(act.values.mean() * 100))
        """

        ntot = self._ncol * self._nrow * self._nlay
        act = xtgeo.grid3d.GridProperty(ncol=self._ncol, nrow=self._nrow,
                                        nlay=self._nlay,
                                        values=np.zeros(ntot, dtype=np.int32),
                                        name=name, discrete=True)

        act._cvalues = self._p_actnum_v  # the SWIG pointer to the C structure
        act._update_values()
        act._codes = {0: '0', 1: '1'}
        act._ncodes = 2
        act._grid = self

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
            A XTGeo GridProperty object
        """

        deltaz = _grid_etc1.get_dz(self, name=name, flip=flip, mask=mask)

        return deltaz

    def get_dxdy(self, names=('dX', 'dY')):
        """
        Return the dX and dY as GridProperty object.

        The values lengths are projected to a constant Z

        Args:
            name (tuple): names of properties

        Returns:
            Two XTGeo GridProperty objects (dx, dy)
        """

        deltax, deltay = _grid_etc1.get_dxdy(self, names=names)

        # return the property objects
        return deltax, deltay

    def get_xyz(self, names=('X_UTME', 'Y_UTMN', 'Z_TVDSS'), mask=True):
        """Returns 3 xtgeo.grid3d.GridProperty objects: x coordinate,
        ycoordinate, zcoordinate.

        The values are mid cell values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates). However, the option mask=True will mask the numpies
        for undef cells.

        Args:
            names: a 3 x tuple of names per property (default is X_UTME,
            Y_UTMN, Z_TVDSS).
            mask: If True, then only active cells.
        """

        xcoord, ycoord, zcoord = _grid_etc1.get_xyz(self, names=names,
                                                    mask=mask)

        # return the objects
        return xcoord, ycoord, zcoord

    def get_xyz_cell_corners(self, ijk=(1, 1, 1), mask=True, zerobased=False):
        """Return a 8 * 3 tuple x, y, z for each corner.

        .. code-block:: none

           2       3
           !~~~~~~~!
           !  top  !
           !~~~~~~~!    Listing corners with Python index (0 base)
           0       1

           6       7
           !~~~~~~~!
           !  base !
           !~~~~~~~!
           4       5

        Args:
            ijk (tuple): A tuple of I J K (NB! cell counting starts from 1
                unless zerobased is True)
            mask (bool): Skip undef cells if set to True.

        Returns:
            A tuple with 24 elements (x1, y1, z1, ... x8, y8, z8)
                for 8 corners. None if cell is inactive and mask=True.

        Example::

            >>> grid = Grid()
            >>> grid.from_file('gullfaks2.roff')
            >>> xyzlist = grid.get_xyz_corners_cell(ijk=(45,13,2))

        Raises:
            RuntimeWarning if spesification is invalid.
        """

        clist = _grid_etc1.get_xyz_cell_corners(self, ijk=ijk, mask=mask,
                                                zerobased=zerobased)

        return clist

    def get_xyz_corners(self, names=('X_UTME', 'Y_UTMN', 'Z_TVDSS')):
        """Returns 8*3 (24) xtgeo.grid3d.GridProperty objects, x, y, z for
        each corner.

        The values are cell corner values. Note that ACTNUM is
        ignored, so these is also extracted for UNDEF cells (which may have
        weird coordinates).

        .. code-block:: none

           2       3
           !~~~~~~~!
           !  top  !
           !~~~~~~~!    Listing corners with Python index (0 base)
           0       1

           6       7
           !~~~~~~~!
           !  base !
           !~~~~~~~!
           4       5

        Args:
            names (list): Generic name of the properties, will have a
                number added, e.g. X0, X1, etc.

        Example::

            >>> grid = Grid()
            >>> grid.from_file('gullfaks2.roff')
            >>> clist = grid.get_xyz_corners()


        Raises:
            RunetimeError if corners has wrong spesification
        """

        grid_props = _grid_etc1.get_xyz_corners(self, names=names)

        # return the 24 objects in a long tuple (x1, y1, z1, ... x8, y8, z8)
        return grid_props

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Get grid geometrics
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_geometrics(self, allcells=False, cellcenter=True,
                       return_dict=False):
        """Get a list of grid geometrics such as origin, min, max, etc.

        This returns a tuple: (xori, yori, zori, xmin, xmax, ymin, ymax, zmin,
        zmax, avg_rotation, avg_dx, avg_dy, avg_dz, grid_regularity_flag)

        If a dictionary is returned, the keys are as in the list above.

        Args:
            allcells (bool): If True, return also for inactive cells
            cellcenter (bool): If True, use cell center, otherwise corner
                coords
            return_dict (bool): If True, return a dictionary instead of a
                list, which is usually more convinient.

        Raises: Nothing

        Example::

            mygrid = Grid('gullfaks.roff')
            gstuff = mygrid.get_geometrics(return_dict=True)
            print('X min/max is {} {}'.format(gstuff['xmin', gstuff['xmax']))

        """

        gresult = _grid_etc1.get_geometrics(self, allcells=allcells,
                                            cellcenter=cellcenter,
                                            return_dict=return_dict)

        return gresult

    # =========================================================================
    # Some more special operations that changes the grid or actnum
    # =========================================================================
    def inactivate_by_dz(self, threshold):
        """Inactivate cells thinner than a given threshold."""

        self = _grid_etc1.inactivate_by_dz(self, threshold)

    def inactivate_inside(self, poly, layer_range=None, inside=True,
                          force_close=False):
        """Inacativate grid inside a polygon.

        The Polygons instance may consist of several polygons. If a polygon
        is open, then the flag force_close will close any that are not open
        when doing the operations in the grid.

        Args:
            poly(Polygons): A polygons object
            layer_range (tuple): A tuple of two ints, upper layer = 1, e.g.
                (1, 14). Note that base layer count is 1 (not zero)
            inside (bool): True if remove inside polygon
            force_close (bool): If True then force polygons to be closed.

        Raises:
            RuntimeError: If a problems with one or more polygons.
            ValueError: If Polygon is not a XTGeo object
        """

        self = _grid_etc1.inactivate_inside(self, poly,
                                            layer_range=layer_range,
                                            inside=inside,
                                            force_close=force_close)

    def inactivate_outside(self, poly, layer_range=None, force_close=False):
        """Inacativate grid outside a polygon. (cf inactivate_inside)"""

        self.inactivate_inside(poly, layer_range=layer_range, inside=False,
                               force_close=force_close)

    def collapse_inactive_cells(self):
        """ Collapse inactive layers where, for I J with other active cells."""

        self = _grid_etc1.collapse_inactive_cells(self)

    def reduce_to_one_layer(self):
        """Reduce the grid to one single single layer.

        Example::

            >>> from xtgeo.grid3d import Grid
            >>> gf = Grid('gullfaks2.roff')
            >>> gf.nlay
            47
            >>> gf.reduce_to_one_layer()
            >>> gf.nlay
            1

        """

        self = _grid_etc1.reduce_to_one_layer(self)

    def translate_coordinates(self, translate=(0, 0, 0), flip=(1, 1, 1)):
        """Translate (move) and/or flip grid coordinates in 3D.

        Args:
            translate (tuple): Tranlattion distance in X, Y, Z coordinates
            flip (tuple): Flip array. The flip values must be 1 or -1.

        Raises:
            RuntimeError: If translation goes wrong for unknown reasons
        """

        self = _grid_etc1.translate_coordinates(self, translate=translate,
                                                flip=flip)

    def convert_to_hybrid(self, nhdiv=10, toplevel=1000.0, bottomlevel=1100.0,
                          region=None, region_number=None):
        """Convert to hybrid grid, either globally or in a selected region..

        Args:
            nhdiv (int): Number of hybrid layers.
            toplevel (float): Top of hybrid grid.
            bottomlevel (float): Base of hybrid grid.
            region (GridProperty): Region property (if needed).
            region_number (int): Which region to apply hybrid grid in.
        """

        self = _grid_hybrid.make_hybridgrid(self, nhdiv=nhdiv,
                                            toplevel=toplevel,
                                            bottomlevel=bottomlevel,
                                            region=region,
                                            region_number=region_number)

    def refine_vertically(self, rfactor):
        """Refine the grid vertically by rfactor (limited to constant for
        all layers)
        """

        self = _grid_refine.refine_vertically(self, rfactor)

    def report_zone_mismatch(self, well=None, zonelogname='ZONELOG',
                             mode=0, zoneprop=None, onelayergrid=None,
                             zonelogrange=(0, 9999), zonelogshift=0,
                             depthrange=None, option=0, perflogname=None):
        """Reports mismatch between wells and a zone.

        Args:
            well (xtgeo.well.Well): a XTGeo well object
            zonelogname (str): Name of the zone logger
            mode (int): Means...
            zoneprop (xtgeo.grid3d.GridProperty): Grid property to use for
                zonation
            zonelogrange (tuple): zone log range, from - to (inclusive)
            onelayergrid (xtgeo.grid3d.Grid): Object as one layer grid
            zonelogshift (int): Deviation (shift) between grid and zonelog
            depthrange (tuple): Interval for search in TVD depth, to speed up
            option (int): Some option)
            perflogname (str): Name of perforation log

        Example::

            g1 = Grid('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')
            g2 = Grid('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')
            g2.reduce_to_one_layer()

            z = GridProperty()
            z.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2_zone.roff',
                        name='Zone')

            w2 = Well('../xtgeo-testdata/wells/gfb/1/34_10-1.w')

            w3 = Well('../xtgeo-testdata/wells/gfb/1/34_10-B-21_B.w')

            wells = [w2, w3]

            for w in wells:
                response = g1.report_zone_mismatch(
                well=w, zonelogname='ZONELOG', mode=0, zoneprop=z,
                onelayergrid=g2, zonelogrange=[0, 19], option=0,
                depthrange=[1700, 9999])
        """

        reports = _grid_etc1.report_zone_mismatch(self,
                                                  well=well,
                                                  zonelogname=zonelogname,
                                                  mode=mode,
                                                  zoneprop=zoneprop,
                                                  onelayergrid=onelayergrid,
                                                  zonelogrange=zonelogrange,
                                                  zonelogshift=zonelogshift,
                                                  depthrange=depthrange,
                                                  option=option,
                                                  perflogname=perflogname)

        return reports
