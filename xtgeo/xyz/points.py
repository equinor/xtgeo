# -*- coding: utf-8 -*-
"""XTGeo xyz.points module"""

from __future__ import print_function, absolute_import

import numpy as np
import pandas as pd
import os.path
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog


class Points(object):
    """Points: Class for a points set in the XTGeo framework.

    The point set is a Pandas dataframe object.

    The instance can be made either from file or by a spesification::

        >>> xp = Points()
        >>> xp.from_file('somefilename', fformat='xyz')

    """

    def __init__(self, *args, **kwargs):

        """The Points constructor method.

        Args:
            xxx (nn): to come

        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()
        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT
        self._df = None

        if args:
            # make instance from file import
            pfile = args[0]
            fformat = kwargs.get('fformat', 'xyz')
            self.from_file(pfile, fformat=fformat)

        else:
            # make instance by kw spesification
            self._xx = kwargs.get('xx', 0.0)

        self.logger.debug('Ran __init__ method for RegularSurface object')

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, pfile, fformat="xyz"):
        """Import Points from a file.

        Args:
            pfile (str): Name of file
            fformat (str): File format, simple XYZ is currently supported

        Returns:
            Object instance (needed optionally)

        """
        if (os.path.isfile(pfile)):
            pass
        else:
            self.logger.critical("Not OK file")
            raise os.error

        if (fformat is None or fformat == "xyz"):
            self._import_xyz(pfile)
        else:
            self.logger.error("Invalid file format")

        return self

    def to_file(self, pfile, fformat="xyz", attributes=None, filter=None,
                wcolumn=None, hcolumn=None, mdcolumn=None):
        """Export well to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz / rms_attr /rms_wellpicks
            attributes (list): List of extra columns to export (some formats)
            filter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            wcolumn (str): Name of well column (rms_wellpicks format only)
            hcolumn (str): Name of horizons column (rms_wellpicks format only)
            mdcolumn (str): Name of MD column (rms_wellpicks format only)

        Returns:
            Number of points exported

        Note that the rms_wellpicks will try to output to::

            HorizonName, WellName, MD  if a MD (mdcolumn) is present,
            HorizonName, WellName, X, Y, Z  otherwise

        Raises:
            KeyError if filter is set and key(s) are invalid

        """
        if self.dataframe is None:
            ncount = 0
            self.logger.warning('Nothing to export!')
            return ncount

        if fformat is None or fformat == "xyz":
            # same as rms_attr, but no attributes are possible
            ncount = self._export_rms_attr(pfile, attributes=None,
                                           filter=filter)

        elif fformat == "rms_attr":
            ncount = self._export_rms_attr(pfile, attributes=attributes,
                                           filter=filter)
        elif fformat == "rms_wellpicks":
            ncount = self._export_rms_wpicks(pfile, hcolumn, wcolumn,
                                             mdcolumn=mdcolumn)

        if ncount == 0:
            self.logger.warning('Nothing to export!')

        return ncount

    # =========================================================================
    # Get and Set properties
    # =========================================================================

    @property
    def nrows(self):
        """ Returns the Pandas dataframe object number of rows"""
        if self._df is None:
            return 0
        else:
            return len(self._df.index)

    @property
    def dataframe(self):
        """ Returns or set the Pandas dataframe object"""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    def get_carray(self, lname):
        """ WRONG PLACE?
        Returns the C array pointer (via SWIG) for a given log.

        Type conversion is double if float64, int32 if DISC log.
        Returns None of log does not exist.
        """
        try:
            np_array = self._df[lname].values
        except:
            return None

        if self.get_logtype(lname) == "DISC":
            carr = self._convert_np_carr_int(np_array)
        else:
            carr = self._convert_np_carr_double(np_array)

        return carr

    # =========================================================================
    # Get tops and zones from well data
    # =========================================================================

    def from_wells(self, wells, tops=True, incl_limit=None, top_prefix='Top',
                   zonelist=None, use_undef=False):

        """Get tops or zone points data from a list of wells.

        Args:
            wells (list): List of XTGeo well objects
            tops (bool): Get the tops if True (default), otherwise zone
            incl_limit (float): Inclination limit for zones (thickness points)
            top_prefix (str): Prefix used for Tops
            zonelist (list-like): Which zone numbers to apply.
            use_undef (bool): If True, then transition from UNDEF is also
                used.

        Returns:
            None if well list is empty; otherwise the number of wells

        Raises:
            Todo
        """

        if len(wells) == 0:
            return None

        dflist = []
        for well in wells:
            wp = well.get_zonation_points(tops=tops, incl_limit=incl_limit,
                                          top_prefix=top_prefix,
                                          zonelist=zonelist,
                                          use_undef=use_undef)
            if wp is not None:
                dflist.append(wp)

        if len(dflist) > 0:
            self._df = pd.concat(dflist, ignore_index=True)
        else:
            return None

        return len(dflist)

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import/Export methods for various formats
    # -------------------------------------------------------------------------

    def _import_xyz(self, pfile):

        # now import all points as Pandas framework

        self._df = pd.read_csv(pfile, delim_whitespace=True, skiprows=0,
                               header=None, names=['X', 'Y', 'Z'],
                               dtype=np.float64, na_values=999.00)

        self.logger.debug(self._df.head())

    def _export_rms_attr(self, pfile, attributes=None, filter=None):
        """Export til RMS attribute, also called RMS extended set

        Filter is on the form {TopName: ['Name1', 'Name2']}

        Returns:
            The number of values exported. If value is 0; then no file
            is made.
        """

        df = self.dataframe.copy()
        columns = ['X', 'Y', 'Z']
        df.fillna(value=999.0, inplace=True)

        mode = 'w'

        # apply filter if any
        if filter:
            for key, val in filter.items():
                if key in df.columns:
                    df = df.loc[df[key].isin(val)]
                else:
                    raise KeyError('The requested filter key {} was not '
                                   'found in dataframe. Valied keys are '
                                   '{}'.format(key, df.columns))

        if len(df.index) < 1:
            self.logger.warning('Nothing to export')
            return 0

        if attributes is not None:
            mode = 'a'
            columns += attributes
            with open(pfile, 'w') as fout:
                for col in attributes:
                    if col in df.columns:
                        fout.write('String ' + col + '\n')

        with open(pfile, mode) as f:
            df.to_csv(f, sep=' ', header=None,
                      columns=columns, index=False, float_format='%.3f')

        return len(df.index)

    def _export_rms_wpicks(self, pfile, hcolumn, wcolumn, mdcolumn=None):
        """Export til RMS wellpicks

        If a MD column (mdcolumn) exists, it will use the MD

        Args:
            pfile (str): File to export to
            hcolumn (str): Name of horizon/zone column in the point set
            wcolumn (str): Name of well column in the point set
            mdcolumn (str): Name of measured depht column (if any)
        Returns:
            The number of values exported. If value is 0; then no file
            is made.

        """

        df = self.dataframe.copy()

        print(df)

        columns = []

        if hcolumn in df.columns:
            columns.append(hcolumn)
        else:
            raise ValueError('Column for horizons/zones <{}> '
                             'not present'.format(hcolumn))

        if wcolumn in df.columns:
            columns.append(wcolumn)
        else:
            raise ValueError('Column for wells <{}> '
                             'not present'.format(wcolumn))

        if mdcolumn in df.columns:
            columns.append(mdcolumn)
        else:
            columns += ['X', 'Y', 'Z']

        print(df)
        print(columns)

        if len(df.index) < 1:
            self.logger.warning('Nothing to export')
            return 0

        with open(pfile, 'w') as f:
            df.to_csv(f, sep=' ', header=None,
                      columns=columns, index=False)

        return len(df.index)

    # -------------------------------------------------------------------------
    # Special methods for nerds
    # -------------------------------------------------------------------------

    def _convert_np_carr_int(self, np_array):
        """
        Convert numpy 1D array to C array, assuming int type. The numpy
        is always a double (float64), so need to convert first
        """
        carr = _cxtgeo.new_intarray(self.nrows)

        np_array = np_array.astype(np.int32)

        _cxtgeo.swig_numpy_to_carr_i1d(np_array, carr)

        return carr

    def _convert_np_carr_double(self, np_array):
        """
        Convert numpy 1D array to C array, assuming double type
        """
        carr = _cxtgeo.new_doublearray(self.nrows)

        _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

        return carr

    def _convert_carr_double_np(self, carray, nlen=None):
        """
        Convert a C array to numpy, assuming double type.
        """
        if nlen is None:
            nlen = len(self._df.index)

        nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

        return nparray
