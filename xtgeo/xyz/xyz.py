# -*- coding: utf-8 -*-
"""XTGeo xyz module (abstract)"""

from __future__ import print_function, absolute_import

import abc
import six
import os.path

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from xtgeo.xyz import _xyz_io


@six.add_metaclass(abc.ABCMeta)
class XYZ(object):
    """Abstract Base class for Points and Polygons in XTGeo, but with
    concrete methods."""

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self._xtg = XTGeoDialog()
        self.logger = self._xtg.functionlogger(clsname)

        self._undef = _cxtgeo.UNDEF
        self._undef_limit = _cxtgeo.UNDEF_LIMIT
        self._df = None
        self._ispolygons = False

        if len(args) >= 1:
            # make instance from file import
            self.logger.info('Instance from file')
            pfile = args[0]
            fformat = kwargs.get('fformat', 'guess')
            self.from_file(pfile, fformat=fformat)
        else:
            self.logger.info('Instance initiated')
            # make instance by kw spesification
            self._xx = kwargs.get('xx', 0.0)

        self.logger.debug('Ran __init__ method for {} object'.format(clsname))

    # =========================================================================
    # Import and export
    # =========================================================================

    @abc.abstractmethod
    def from_file(self, pfile, fformat='guess'):
        """Import Points or Polygons from a file.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'guess': Try to choose file format based on extension

        Args:
            pfile (str): Name of file
            fformat (str): File format, see list above

        Returns:
            Object instance (needed optionally)

        Raises:
            OSError: if file is not present or wrong permissions.

        """

        if (os.path.isfile(pfile)):
            pass
        else:
            self.logger.critical('Not OK file')
            raise os.error

        froot, fext = os.path.splitext(pfile)
        if fformat == 'guess':
            if len(fext) == 0:
                self.logger.critical('File extension missing. STOP')
                raise SystemExit
            else:
                fformat = fext.lower().replace('.', '')

        if fformat in ['xyz', 'poi', 'pol']:
            _xyz_io.import_xyz(self, pfile)
        elif (fformat == 'zmap'):
            _xyz_io.import_zmap(self, pfile)
        else:
            self.logger.error('Invalid file format (not supported): {}'
                              .format(fformat))
            raise SystemExit

        return self

    @abc.abstractmethod
    def to_file(self, pfile, fformat='xyz', attributes=None, filter=None,
                wcolumn=None, hcolumn=None, mdcolumn=None):
        """Export XYZ (Points/Polygons) to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
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

        if fformat is None or fformat in ['xyz', 'poi', 'pol']:
            # NB! reuse export_rms_attr function, but no attributes
            # are possible
            ncount = _xyz_io.export_rms_attr(self, pfile, attributes=None,
                                             filter=filter)

        elif fformat == 'rms_attr':
            ncount = _xyz_io.export_rms_attr(self, pfile,
                                             attributes=attributes,
                                             filter=filter)
        elif fformat == 'rms_wellpicks':
            ncount = _xyz_io.export_rms_wpicks(self, pfile, hcolumn, wcolumn,
                                               mdcolumn=mdcolumn)

        if ncount == 0:
            self.logger.warning('Nothing to export!')

        return ncount

    # =========================================================================
    # Get and Set properties
    # =========================================================================

    @abc.abstractproperty
    def nrow(self):
        """ Returns the Pandas dataframe object number of rows"""
        if self._df is None:
            return 0
        else:
            return len(self._df.index)

    @abc.abstractproperty
    def dataframe(self):
        """ Returns or set the Pandas dataframe object"""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    # @abc.abstractmethod
    # def get_carray(self, lname):
    #     """Returns the C array pointer (via SWIG) for a given log.

    #     Type conversion is double if float64, int32 if DISC log.
    #     Returns None of log does not exist.
    #     """
    #     try:
    #         np_array = self._df[lname].values
    #     except Exception:
    #         return None

    #     if self.get_logtype(lname) == 'DISC':
    #         carr = _xyz_lowlevel.convert_np_carr_int(self, np_array)
    #     else:
    #         carr = _xyz_lowlevel.convert_np_carr_double(self, np_array)

    #     return carr
