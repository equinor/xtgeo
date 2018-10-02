# -*- coding: utf-8 -*-

"""GridProperties class"""
from __future__ import division, absolute_import
from __future__ import print_function

import os.path
import sys

import xtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid3D

from xtgeo.grid3d import _gridprops_io
from xtgeo.grid3d import _gridprops_etc

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class GridProperties(Grid3D):
    """Class for a collection of 3D grid props, belonging to the same grid.

    See also the :class:`xtgeo.grid3d.GridProperty` class.
    """

    def __init__(self):

        self._ncol = 10
        self._nrow = 12
        self._nlay = 14

        self._props = []            # list of GridProperty objects
        self._names = []            # list of GridProperty names
        self._dates = []            # list of dates (_after_ import) YYYYDDMM

    @property
    def names(self):
        """Returns a list of property names

        Example::

            namelist = props.names
            for prop in namelist:
                print ('Property name is {}'.format(name))

        """

        return self._names

    @property
    def props(self):
        """Returns a list of XTGeo GridProperty objects, None if empty.

        Example::

            gg = xtgeo.grid3d.Grid('TEST1.EGRID')
            myprops = XTGeo.grid3d.GridProperties()
            from_file('TEST1.INIT', fformat='init', names=['PORO']         )
            proplist = myprops.props
            for prop in proplist:
                print ('Property object ID is {}'.format(prop))

            # adding a property, e.g. get ACTNUM as a property from the grid
            actn = gg.get_actnum()  # this will get actn as a GridProperty
            myprops.add_props([actn])
        """
        if len(self._props) == 0:
            return None

        return self._props

    @props.setter
    def props(self, propslist):
        self._props = propslist

    @property
    def dates(self):
        """Returns a list of valid (found) dates after import.

        Returns None if no dates present

        Example::

            datelist = props.dates
            for date in datelist:
                print ('Date applied is {}'.format(date))

        """
        if len(self._dates) == 0:
            return None

        return self._dates

    # =========================================================================
    # Copy, and etc aka setters and getters
    # =========================================================================

    def copy(self):
        """Copy a GridProperties instance to a new unique instance.

        Note that the GridProperty instances will also be unique.
        """

        new = GridProperties()
        new._ncol = self._ncol
        new._nrow = self._nrow
        new._nlay = self._nlay

        for prp in self._props:
            newprp = prp.copy()
            new._props.append(newprp)

        for name in self._names:
            new._names.append(name)

        for date in self._dates:
            new._dates.append(date)

        return new

    def get_prop_by_name(self, name):
        """Find and return a property object (GridProperty) by name."""

        for prop in self._props:
            logger.debug("Look for {}, actual is {}"
                         .format(name, prop.name))
            if prop.name == name:
                logger.debug(repr(prop))
                return prop

        raise ValueError('Cannot find property with name <{}>'.format(name))

    def append_props(self, proplist):
        """Adds a list of GridProperty objects to the current
        GridProperties instance."""

        for prop in proplist:
            if isinstance(prop, xtgeo.grid3d.GridProperty):

                self._props.append(prop)
                self._names.append(prop.name)
                self._dates.append(prop.date)
            else:
                raise ValueError('Input property is not a valid GridProperty '
                                 'object')

    # =========================================================================
    # Import and export
    # This class can importies several properties in one go, which is efficient
    # for some file types such as Eclipse INIT and UNRST, and Roff
    # =========================================================================

    def from_file(self, pfile, fformat='roff', names=None,
                  dates=None, grid=None, namestyle=0):
        """Import grid properties from file in one go.

        This class is particulary useful for Eclipse INIT and RESTART files.

        Args:
            pfile (str): Name of file with properties
            fformat (str): roff/init/unrst
            names: list of property names, e.g. ['PORO', 'PERMX']
            dates: list of dates on YYYYMMDD format, for restart files
            grid (obj): The grid geometry object (optional if ROFF)
            namestyle (int): 0 (default) for style SWAT_20110223,
                1 for SWAT--2011_02_23 (applies to restart only)

        Example::
            >>> props = GridProperties()
            >>> props.from_file('ECL.UNRST', fformat='unrst',
                dates=[20110101, 20141212], names=['PORO', 'DZ']

        Raises:
            FileNotFoundError: if input file is not found
            ValueError: if a property is not found
            RuntimeWarning: if some dates are not found

        """

        # work on file extension
        froot, fext = os.path.splitext(pfile)
        if not fext:
            # file extension is missing, guess from format
            logger.info('File extension missing; guessing...')

            useext = ''
            if fformat == 'init':
                useext = '.INIT'
            elif fformat == 'unrst':
                useext = '.UNRST'
            elif fformat == 'roff':
                useext = '.roff'

            pfile = froot + useext

        logger.info('File name to be used is {}'.format(pfile))

        if os.path.isfile(pfile):
            logger.info('File {} exists OK'.format(pfile))
        else:
            logger.warning('No such file: {}'.format(pfile))
            sys.exit(1)

        if fformat.lower() == 'roff':
            self._import_roff(pfile, names)
        elif fformat.lower() in ('init', 'unrst'):
            _gridprops_io.import_ecl_output(self, pfile,
                                            dates=dates, grid=grid,
                                            names=names,
                                            namestyle=namestyle)
        else:
            logger.warning('Invalid file format')
            sys.exit(1)

    def to_file(self, pfile, fformat='roff'):
        """Export grid property to file. NB not working!

        Args:
            pfile (str): file name
            fformat (str): file format to be used (roff is the only supported)
            mode (int): 0 for binary ROFF, 1 for ASCII
        """
        raise NotImplementedError('Not implented yet!')

    def dataframe(self, activeonly=True, ijk=False, xyz=False,
                  doubleformat=False, grid=None):
        """Returns a Pandas dataframe table for the properties

        Args:
            activeonly (bool): If True, return only active cells
            ijk (bool): If True, show cell indices, IX JY KZ columns
            xyz (bool): If True, show cell center coordinates (needs grid).
            doubleformat (bool): If True, floats are 64 bit, otherwise 32 bit.
                Note that coordinates (if xyz=True) is always 64 bit floats.
            grid (Grid): The grid geometry object. This is required for the
                xyz option.

        Returns:
            Pandas dataframe object

        Examples::

            grd = Grid(gfile1, fformat='egrid')
            xpr = GridProperties()

            names = ['SOIL', 'SWAT', 'PRESSURE']
            dates = [19991201]
            xpr.from_file(rfile1, fformat='unrst', names=names, dates=dates,
                        grid=grd)
            df = x.dataframe(activeonly=False, ijk=True, xyz=True)

        """

        df = _gridprops_etc.dataframe(self, activeonly=activeonly, ijk=ijk,
                                      xyz=xyz, doubleformat=doubleformat,
                                      grid=grid)

        return df

    # =========================================================================
    # Static methods (scans etc)
    # =========================================================================

    @staticmethod
    def scan_keywords(pfile, fformat='xecl', maxkeys=100000, dataframe=False,
                      dates=False):
        """Quick scan of keywords in Eclipse binary restart/init/... file.

        For Eclipse files:
        Returns a list of tuples (or dataframe), e.g. ('PRESSURE',
        'REAL', 355299, 3582700), where (keyword, type, no_of_values,
        byteposition_in_file)

        For ROFF files
        Returns a list of tuples (or dataframe), e.g.
        ('translate!xoffset', 'float', 1, 3582700),
        where (keyword, type, no_of_values, byteposition_in_file).

        For Eclipse, the byteposition is to the KEYWORD, while for ROFF
        the byte position is to the beginning of the actual data.

        Args:
            pfile (str): Name or a filehandle to file with properties
            fformat (str): xecl (Eclipse INIT, RESTART, ...) or roff for
                ROFF binary,
            maxkeys (int): Maximum number of keys
            dataframe (bool): If True, return a Pandas dataframe instead
            dates (bool): if True, the date is the last column (only
                menaingful for restart files). Default is False.

        Return:
            A list of tuples or dataframe with keyword info

        Example::
            >>> props = GridProperties()
            >>> dlist = props.scan_keywords('ECL.UNRST')

        """

        dlist = _gridprops_io.scan_keywords(pfile, fformat=fformat,
                                            maxkeys=maxkeys,
                                            dataframe=dataframe,
                                            dates=dates)

        return dlist

    @staticmethod
    def scan_dates(pfile, fformat='unrst', maxdates=1000, dataframe=False):
        """Quick scan dates in a simulation restart file.

        Args:
            pfile (str): Name of file or file handle with properties
            fformat (str): unrst (so far)
            maxdates (int): Maximum number of dates to collect
            dataframe (bool): If True, return a Pandas dataframe instead

        Return:
            A list of tuples or a dataframe with (seqno, date),
            date is on YYYYMMDD form.

        Example::
            >>> props = GridProperties()
            >>> dlist = props.scan_dates('ECL.UNRST')

        """

        dlist = _gridprops_io.scan_dates(pfile, fformat=fformat,
                                         maxdates=maxdates,
                                         dataframe=dataframe)

        return dlist
