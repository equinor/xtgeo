"""XTGeo Points class"""

# #############################################################################
#
# NAME:
#    points.py
#
# AUTHOR(S):
#    Jan C. Rivenaes
#
# DESCRIPTION:
#    Class for XYZ points and also a base class for polygons (connected points)
#    These points are stored as Pandas in Python.
# TODO/ISSUES/BUGS:
#
# LICENCE:
#    Statoil property
# #############################################################################

from __future__ import print_function

import numpy as np
import pandas as pd
import os.path
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog


class Points(object):
    """Class for a points set in the XTGeo framework.

    The point set is a Pandas dataframe object.
    """

    def __init__(self, *args, **kwargs):

        """The __init__ (constructor) method.

        The instance can be made either from file or by a spesification::

        >>> xp = Points()
        >>> xp.from_file('somefilename', fformat='xyz')

        Args:
            xxx (nn): to come


        """

        clsname = "{}.{}".format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()
        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT

        if args:
            # make instance from file import
            pfile = args[0]
            fformat = kwargs.get('fformat', 'xyz')
            self.from_file(pfile, fformat=fformat)

        else:
            # make instance by kw spesification
            self._xx = kwargs.get('xx', 0.0)

            values = kwargs.get('values', None)

        self.logger.debug('Ran __init__ method for RegularSurface object')

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, pfile, fformat="xyz"):
        """
        Import Points from a file.

        Args:
            pfile (str): Name of file
            fformat (str): File format, simple XYZ is currently supported

        Returns:
            Object instance, optionally

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

    def to_file(self, pfile, fformat="xyz", attributes=None):
        """
        Export well to file

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz / rms_attr
            attributes (list): List of extra columns to export (some formats)

        Example::

            >>> x = Well()

        """
        if fformat is None or fformat == "xyz":
            self._export_xyz(pfile)

        elif fformat == "rms_attr":
            self._export_rms_attr(pfile, attributes=attributes)

    # =========================================================================
    # Get and Set properties
    # =========================================================================

    @property
    def nrows(self):
        """ Returns the Pandas dataframe object number of rows"""
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

    def from_wells(self, wells, zonelogname='ZONELOG', tops=True,
                   incl_limit=None, top_prefix='Top', zonelist=None):
        """Get tops or zone points data from a list of wells.

        Args:
            wells (list): List of XTGeo well objects
            zonelogname (str): Name of zonelog, default is 'ZONELOG'
            tops (bool): Get the tops if True (default), otherwise zone
            incl_limit (float): Inclination limit for zones (thickness points)
            top_prefix (str): Prefix used for Tops

        Returns:
            None if well list is empty; otherwise the number of wells

        Raises:
            Todo
        """

        if len(wells) == 0:
            return None

        dflist = []
        for well in wells:
            wp = well.get_zonation_points(zonelogname=zonelogname,
                                          tops=tops, incl_limit=incl_limit,
                                          top_prefix=top_prefix,
                                          zonelist=zonelist)
            dflist.append(wp)

        if len(dflist) > 0:
            self._df = pd.concat(dflist)
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

    # Import XYZ
    # -------------------------------------------------------------------------
    def _import_xyz(self, pfile):

        # now import all points as Pandas framework

        self._df = pd.read_csv(pfile, delim_whitespace=True, skiprows=0,
                               header=None, names=['X', 'Y', 'Z'],
                               dtype=np.float64, na_values=999.00)

        self.logger.debug(self._df.head())

    # Export RMS ascii
    # -------------------------------------------------------------------------
    def _export_rms_attr(self, pfile, attributes=None):
        """Export til RMS attribute, also called RMS extended set"""

        df = self.dataframe
        columns = ['X', 'Y', 'Z']
        mode = 'r'
        if attributes is not None:
            mode = 'a'
            columns += attributes
            with open(pfile, 'w') as fout:
                for col in attributes:
                    if col in df.columns:
                        fout.write('String ' + col + '\n')

        with open(pfile, mode) as f:
            df.to_csv(f, sep=' ', header=None,
                      columns=columns, index=False)



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
