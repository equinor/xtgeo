# -*- coding: utf-8 -*-
"""XTGeo well module"""

from __future__ import print_function, absolute_import

import sys
import numpy as np
import pandas as pd
import os.path
import logging

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog
from . import _wellmarkers
from . import _well_io

xtg = XTGeoDialog()


class Well(object):
    """Class for a well in the xtgeo framework.

    The well logs are stored as Pandas dataframe, which make manipulation
    easy and fast.

    The well trajectory are here represented as logs, and XYZ have magic names:
    X_UTME, Y_UTMN, Z_TVDSS.
    """

    UNDEF = _cxtgeo.UNDEF
    UNDEF_LIMIT = _cxtgeo.UNDEF_LIMIT
    UNDEF_INT = _cxtgeo.UNDEF_INT
    UNDEF_INT_LIMIT = _cxtgeo.UNDEF_INT_LIMIT

    def __init__(self, *args, **kwargs):

        """The __init__ (constructor) method.

        The instance can be made either from file or (todo!) by spesification::

        >>> x1 = Well('somefilename')  # assume RMS ascii well
        >>> x2 = Well('somefilename', fformat='rms_ascii')

        Args:
            xxx (nn): to come


        """

        clsname = '{}.{}'.format(type(self).__module__, type(self).__name__)
        self.logger = logging.getLogger(clsname)
        self.logger.addHandler(logging.NullHandler())

        self._xtg = XTGeoDialog()

        # MD and ZONELOG are two essential logs; keep track of these names
        self._mdlogname = None
        self._zonelogname = None

        if args:
            # make instance from file import
            wfile = args[0]
            fformat = kwargs.get('fformat', 'rms_ascii')
            mdlogname = kwargs.get('mdlogname', None)
            zonelogname = kwargs.get('zonelogname', None)
            self.from_file(wfile, fformat=fformat, mdlogname=mdlogname,
                           zonelogname=zonelogname)

        else:
            # dummy
            self._xx = kwargs.get('xx', 0.0)

            # # make instance by kw spesification ... todo
            # raise RuntimeWarning('Cannot initialize a Well object without '
            #                      'import at the current stage.')

        self.logger.debug('Ran __init__ method for RegularSurface object')

    # =========================================================================
    # Import and export
    # =========================================================================

    def from_file(self, wfile, fformat='rms_ascii',
                  mdlogname=None, zonelogname=None, strict=True):
        """Import well from file.

        Args:
            wfile (str): Name of file
            fformat (str): File format, rms_ascii (rms well) is
                currently supported and default format.
            mdlogname (str): Name of measured depth log, if any
            zonelogname (str): Name of zonation log, if any
            strict (bool): If True, then import will fail if
                zonelogname or mdlogname are asked for but not present
                in wells.

        Returns:
            Object instance (optionally)

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mywell = Well('31_2-6.w')
        """

        if (os.path.isfile(wfile)):
            pass
        else:
            self.logger.critical('Not OK file')
            raise os.error

        if (fformat is None or fformat == 'rms_ascii'):
            attr = _well_io.import_rms_ascii(wfile, mdlogname=mdlogname,
                                             zonelogname=zonelogname,
                                             strict=strict)
        else:
            self.logger.error('Invalid file format')

        # set the attributes
        self._wlogtype = attr['wlogtype']
        self._wlogrecord = attr['wlogrecord']
        self._lognames_all = attr['lognames_all']
        self._lognames = attr['lognames']
        self._ffver = attr['ffver']
        self._wtype = attr['wtype']
        self._rkb = attr['rkb']
        self._xpos = attr['xpos']
        self._ypos = attr['ypos']
        self._wname = attr['wname']
        self._nlogs = attr['nlogs']
        self._df = attr['df']
        self._mdlogname = attr['mdlogname']
        self._zonelogname = attr['zonelogname']

        return self

    def to_file(self, wfile, fformat='rms_ascii'):
        """
        Export well to file

        Args:
            wfile (str): Name of file
            fformat (str): File format

        Example::

            >>> x = Well()

        """
        if (fformat is None or fformat == 'rms_ascii'):
            _well_io.export_rms_ascii(self, wfile)

    # =========================================================================
    # Get and Set properties (tend to pythonic properties; not javaic get
    # & set syntax, if possible)
    # =========================================================================

    @property
    def rkb(self):
        """ Returns RKB height for the well."""
        return self._rkb

    @property
    def xpos(self):
        """ Returns well header X position."""
        return self._xpos

    @property
    def ypos(self):
        """ Returns well header Y position."""
        return self._ypos

    @property
    def wellname(self):
        """ Returns well name."""
        return self._wname

    @property
    def mdlogname(self):
        """ Returns name of MD log, if any (Null if not)."""
        return self._mdlogname

    @property
    def zonelogname(self):
        """ Returns name of zone log, if any (Null if not)."""
        return self._zonelogname

    @property
    def xwellname(self):
        """
        Returns well name on a file syntax safe form (/ and space replaced
        with _).
        """
        x = self._wname
        x = x.replace('/', '_')
        x = x.replace(' ', '_')
        return x

    @property
    def truewellname(self):
        """
        Returns well name on the assummed form aka '31/2-E-4 AH2'.
        """
        x = self.xwellname
        x = x.replace('_', '/', 1)
        x = x.replace('_', ' ')
        return x

    @property
    def dataframe(self):
        """ Returns or set the Pandas dataframe object for all logs."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    @property
    def nrow(self):
        """ Returns the Pandas dataframe object number of rows"""
        return len(self._df.index)

    @property
    def ncol(self):
        """ Returns the Pandas dataframe object number of columns"""
        return len(self._df.columns)

    @property
    def nlogs(self):
        """ Returns the Pandas dataframe object number of columns"""
        return len(self._df.columns) - 3

    @property
    def lognames_all(self):
        """ Returns the Pandas dataframe column names as list (incl X Y TVD)"""
        return list(self._df)

    @property
    def lognames(self):
        """ Returns the Pandas dataframe column as list (excl X Y TVD)"""
        return list(self._df)[3:]

    def get_logtype(self, lname):
        """ Returns the type of a give log (e.g. DISC). None if not exists."""
        if lname in self._wlogtype:
            return self._wlogtype[lname]
        else:
            return None

    def get_logrecord(self, lname):
        """ Returns the record (dict) of a give log. None if not exists"""

        if lname in self._wlogtype:
            return self._wlogrecord[lname]
        else:
            return None

    def get_logrecord_codename(self, lname, key):
        """ Returns the name entry of a log record, for a given key

        Example::

            # get the name for zonelog entry no 4:
            zname = well.get_logrecord_codename('ZONELOG', 4)
        """

        zlogdict = self.get_logrecord(lname)
        try:
            name = zlogdict[key]
        except Exception:
            return None
        else:
            return name

    def get_carray(self, lname):
        """Returns the C array pointer (via SWIG) for a given log.

        Type conversion is double if float64, int32 if DISC log.
        Returns None of log does not exist.
        """
        try:
            np_array = self._df[lname].values
        except Exception:
            return None

        if self.get_logtype(lname) == 'DISC':
            carr = self._convert_np_carr_int(np_array)
        else:
            carr = self._convert_np_carr_double(np_array)

        return carr

    def get_filled_dataframe(self):
        """Fill the Nan's in the dataframe with real UNDEF values.

        This module returns a copy of the dataframe in the object; it
        does not change the instance.

        Returns:
            A pandas dataframe where Nan er replaces with high values.

        """

        lnames = self.lognames

        newdf = self._df.copy()

        # make a dictionary of datatypes
        dtype = {'X_UTME': 'float64', 'Y_UTMN': 'float64',
                 'Z_TVDSS': 'float64'}

        dfill = {'X_UTME': Well.UNDEF, 'Y_UTMN': Well.UNDEF,
                 'Z_TVDSS': Well.UNDEF}

        for lname in lnames:
            if self.get_logtype(lname) == 'DISC':
                dtype[lname] = 'int32'
                dfill[lname] = Well.UNDEF_INT
            else:
                dtype[lname] = 'float64'
                dfill[lname] = Well.UNDEF

        # now first fill Nan's (because int cannot be converted if Nan)
        newdf.fillna(dfill, inplace=True)

        # now cast to dtype
        newdf.astype(dtype, inplace=True)

        return newdf

    def create_relative_hlen(self):
        """Make a relative length of a well, as a log.

        The first well og entry defines zero, then the horizontal length
        is computed relative to that by simple geometric methods.
        """

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        # extract numpies from XYZ trajetory logs
        ptr_xv = self.get_carray('X_UTME')
        ptr_yv = self.get_carray('Y_UTMN')
        ptr_zv = self.get_carray('Z_TVDSS')

        # get number of rows in pandas
        nlen = self.nrow

        ptr_hlen = _cxtgeo.new_doublearray(nlen)

        ier = _cxtgeo.pol_geometrics(nlen, ptr_xv, ptr_yv, ptr_zv, ptr_hlen,
                                     xtg_verbose_level)

        if ier != 0:
            sys.exit(-9)

        dnumpy = self._convert_carr_double_np(ptr_hlen)
        self._df['R_HLEN'] = pd.Series(dnumpy, index=self._df.index)

        # delete tmp pointers
        _cxtgeo.delete_doublearray(ptr_xv)
        _cxtgeo.delete_doublearray(ptr_yv)
        _cxtgeo.delete_doublearray(ptr_zv)
        _cxtgeo.delete_doublearray(ptr_hlen)

    def geometrics(self):
        """Compute some well geometrical arrays MD and INCL, as logs.

        These are kind of quasi measurements hence the logs will named
        with a Q in front as Q_MDEPTH and Q_INCL.

        These logs will be added to the dataframe
        """

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        # extract numpies from XYZ trajetory logs
        ptr_xv = self.get_carray('X_UTME')
        ptr_yv = self.get_carray('Y_UTMN')
        ptr_zv = self.get_carray('Z_TVDSS')

        # get number of rows in pandas
        nlen = self.nrow

        ptr_md = _cxtgeo.new_doublearray(nlen)
        ptr_incl = _cxtgeo.new_doublearray(nlen)

        ier = _cxtgeo.well_geometrics(nlen, ptr_xv, ptr_yv, ptr_zv, ptr_md,
                                      ptr_incl, 0, xtg_verbose_level)

        if ier != 0:
            sys.exit(-9)

        dnumpy = self._convert_carr_double_np(ptr_md)
        self._df['Q_MDEPTH'] = pd.Series(dnumpy, index=self._df.index)

        dnumpy = self._convert_carr_double_np(ptr_incl)
        self._df['Q_INCL'] = pd.Series(dnumpy, index=self._df.index)

        # delete tmp pointers
        _cxtgeo.delete_doublearray(ptr_xv)
        _cxtgeo.delete_doublearray(ptr_yv)
        _cxtgeo.delete_doublearray(ptr_zv)
        _cxtgeo.delete_doublearray(ptr_md)
        _cxtgeo.delete_doublearray(ptr_incl)

    def get_fence_polyline(self, sampling=20, extend=2, tvdmin=None):
        """
        Return a fence polyline as a numpy array.

        (Perhaps this should belong to a polygon class?)
        """

        # need to call the C function...
        _cxtgeo.xtg_verbose_file('NONE')
        xtg_verbose_level = self._xtg.syslevel

        df = self._df

        if tvdmin is not None:
            self._df = df[df['Z_TVDSS'] > tvdmin]

        ptr_xv = self.get_carray('X_UTME')
        ptr_yv = self.get_carray('Y_UTMN')
        ptr_zv = self.get_carray('Z_TVDSS')

        nbuf = 1000000
        ptr_xov = _cxtgeo.new_doublearray(nbuf)
        ptr_yov = _cxtgeo.new_doublearray(nbuf)
        ptr_zov = _cxtgeo.new_doublearray(nbuf)
        ptr_hlv = _cxtgeo.new_doublearray(nbuf)

        ptr_nlen = _cxtgeo.new_intpointer()

        ier = _cxtgeo.pol_resample(self.nrow, ptr_xv, ptr_yv, ptr_zv,
                                   sampling, extend, nbuf, ptr_nlen,
                                   ptr_xov, ptr_yov, ptr_zov, ptr_hlv,
                                   0, xtg_verbose_level)

        if ier != 0:
            sys.exit(-2)

        nlen = _cxtgeo.intpointer_value(ptr_nlen)

        npxarr = self._convert_carr_double_np(ptr_xov, nlen=nlen)
        npyarr = self._convert_carr_double_np(ptr_yov, nlen=nlen)
        npzarr = self._convert_carr_double_np(ptr_zov, nlen=nlen)
        npharr = self._convert_carr_double_np(ptr_hlv, nlen=nlen)
        npharr = npharr - sampling * extend

        x = np.concatenate((npxarr, npyarr, npzarr, npharr), axis=0)
        x = np.reshape(x, (nlen, 4), order='F')

        _cxtgeo.delete_doublearray(ptr_xov)
        _cxtgeo.delete_doublearray(ptr_yov)
        _cxtgeo.delete_doublearray(ptr_zov)
        _cxtgeo.delete_doublearray(ptr_hlv)

        return x

    def report_zonation_holes(self, zonelogname=None, mdlogname=None,
                              threshold=5):
        """Reports if well has holes in zonation, less or equal to N samples.

        Zonation may have holes due to various reasons, and
        usually a few undef samples indicates that something is wrong.
        This method reports well and start interval of the "holes"

        Args:
            zonelogname (str): name of Zonelog to be applied
            threshold (int): Number of samples (max.) that defines a hole, e.g.
                5 means that undef samples in the range [1, 5] (including 5) is
                applied

        Returns:
            A Pandas dataframe as report. None if no list is made.
        """

        if zonelogname is None:
            zonelogname = self._zonelogname

        if mdlogname is None:
            mdlogname = self._mdlogname

        print('MDLOGNAME is {}'.format(mdlogname))

        wellreport = []

        try:
            zlog = self._df[zonelogname].values.copy()
        except Exception:
            self.logger.warning('Cannot get zonelog')
            xtg.warn('Cannot get zonelog {} for {}'
                     .format(zonelogname, self.wellname))
            return None

        try:
            mdlog = self._df[mdlogname].values
        except Exception:
            self.logger.warning('Cannot get mdlog')
            xtg.warn('Cannot get mdlog {} for {}'
                     .format(mdlogname, self.wellname))
            return None

        x = self._df['X_UTME'].values
        y = self._df['Y_UTMN'].values
        z = self._df['Z_TVDSS'].values
        zlog[np.isnan(zlog)] = Well.UNDEF_INT

        nc = 0
        first = True
        hole = False
        for ind, zone in np.ndenumerate(zlog):
            i = ind[0]
            if zone > Well.UNDEF_INT_LIMIT and first:
                continue

            if zone < Well.UNDEF_INT_LIMIT and first:
                first = False
                continue

            if zone > Well.UNDEF_INT_LIMIT:
                nc += 1
                hole = True

            if zone > Well.UNDEF_INT_LIMIT and nc > threshold:
                self.logger.info('Restart first (bigger hole)')
                hole = False
                first = True
                nc = 0
                continue

            if hole and zone < Well.UNDEF_INT_LIMIT and nc <= threshold:
                # here we have a hole that fits criteria
                if mdlog is not None:
                    entry = (i, x[i], y[i], z[i], int(zone), self.xwellname,
                             mdlog[i])
                else:
                    entry = (i, x[i], y[i], z[i], int(zone), self.xwellname)

                wellreport.append(entry)

                # retstart count
                hole = False
                nc = 0

            if hole and zone < Well.UNDEF_INT_LIMIT and nc > threshold:
                hole = False
                nc = 0

        if len(wellreport) == 0:
            return None
        else:
            if mdlog is not None:
                clm = ['INDEX', 'X_UTME', 'Y_UTMN', 'Z_TVDSS',
                       'Zone', 'Well', 'MD']
            else:
                clm = ['INDEX', 'X_UTME', 'Y_UTMN', 'Z_TVDSS', 'Zone', 'Well']

            df = pd.DataFrame(wellreport, columns=clm)
            return df

    def get_zonation_points(self, tops=True, incl_limit=80, top_prefix='Top',
                            zonelist=None, use_undef=False):

        """Extract zonation points from Zonelog and make a marker list.

        Currently it is either 'Tops' or 'Zone' (thicknesses); default
        is tops (i.e. tops=True).

        Args:
            tops (bool): If True then compute tops, else (thickness) points.
            incl_limit (float): If given, and usezone is True, the max
                angle of inclination to be  used as input to zonation points.
            top_prefix (str): As well logs usually have isochore (zone) name,
                this prefix could be Top, e.g. 'SO43' --> 'TopSO43'
            zonelist (list of int): Zones to use
            use_undef (bool): If True, then transition from UNDEF is also
                used.


        Returns:
            A pandas dataframe (ready for the xyz/Points class), None
            if a zonelog is missing
        """

        zlist = []
        # get the relevant logs:

        self.geometrics()

        # as zlog is float64; need to convert to int array with high
        # number as undef
        if self.zonelogname is not None:
            zlog = self._df[self.zonelogname].values
            zlog[np.isnan(zlog)] = Well.UNDEF_INT
            zlog = np.rint(zlog).astype(int)
        else:
            return None

        xv = self._df['X_UTME'].values
        yv = self._df['Y_UTMN'].values
        zv = self._df['Z_TVDSS'].values
        incl = self._df['Q_INCL'].values
        md = self._df['Q_MDEPTH'].values

        if self.mdlogname is not None:
            md = self._df[self.mdlogname].values

        self.logger.info('\n')
        self.logger.info(zlog)
        self.logger.info(xv)
        self.logger.info(zv)

        self.logger.info(self.get_logrecord(self.zonelogname))
        if zonelist is None:
            # need to declare as list; otherwise Py3 will get dict.keys
            zonelist = list(self.get_logrecord(self.zonelogname).keys())

        self.logger.info('Find values for {}'.format(zonelist))

        ztops, ztopnames, zisos, zisonames = (
            _wellmarkers.extract_ztops(self, zonelist, xv, yv, zv, zlog, md,
                                       incl, tops=tops,
                                       incl_limit=incl_limit,
                                       prefix=top_prefix, use_undef=use_undef))

        if tops:
            zlist = ztops
        else:
            zlist = zisos

        self.logger.debug(zlist)

        if tops:
            df = pd.DataFrame(zlist, columns=ztopnames)
        else:
            df = pd.DataFrame(zlist, columns=zisonames)

        self.logger.info(df)

        return df

    def get_zone_interval(self, zonevalue, resample=1):

        """Extract the X Y Z ID line (polyline) segment for a given
        zonevalue.

        Args:
            zonevalue (int): The zone value to extract
            resample (int): If given, resample every N'th sample to make
                polylines smaller in terms of bit and bytes.
                1 = No resampling.


        Returns:
            A pandas dataframe X Y Z ID (ready for the xyz/Polygon class),
            None if a zonelog is missing or actual zone does dot
            exist in the well.
        """

        if resample < 1 or not isinstance(resample, int):
            raise KeyError('Key resample of wrong type (must be int >= 1)')

        df = self.get_filled_dataframe()

        # the technical solution here is to make a tmp column which
        # will add one number for each time the actual segment is repeated,
        # not straightforward... (thanks to H. Berland for tip)

        df['ztmp'] = df[self.zonelogname]
        df['ztmp'] = (df[self.zonelogname] != zonevalue).astype(int)

        df['ztmp'] = (df.ztmp != df.ztmp.shift()).cumsum()

        df = df[df[self.zonelogname] == zonevalue]

        m1 = df['ztmp'].min()
        m2 = df['ztmp'].max()
        if np.isnan(m1):
            self.logger.debug('Returns (no data)')
            return None

        df2 = df.copy()

        dflist = []
        for m in range(m1, m2 + 1):
            df = df2.copy()
            df = df2[df2['ztmp'] == m]
            if len(df.index) > 0:
                dflist.append(df)

        dxlist = []
        for i in range(len(dflist)):
            dx = dflist[i]
            dx = dx.rename(columns={'ztmp': 'ID'})
            cols = [x for x in dx.columns
                    if x not in ['X_UTME', 'Y_UTMN', 'Z_TVDSS', 'ID']]

            dx = dx.drop(cols, axis=1)
            # rename columns:
            dx.columns = ['X_UTME', 'Y_UTMN', 'Z_TVDSS', 'ID']
            # now resample every N'th
            if resample > 1:
                dx = pd.concat([dx.iloc[::resample, :], dx.tail(1)])

            dxlist.append(dx)

        df = pd.concat(dxlist)
        df.reset_index(inplace=True, drop=True)

        self.logger.debug('DF from well:\n{}'.format(df))
        return df

    # =========================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # =========================================================================

    # -------------------------------------------------------------------------
    # Import/Export methods for various formats
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Special methods for nerds
    # -------------------------------------------------------------------------

    def _convert_np_carr_int(self, np_array):
        """Convert numpy 1D array to C array, assuming int type.

        The numpy is always a double (float64), so need to convert first
        """

        carr = _cxtgeo.new_intarray(self.nrow)

        np_array = np_array.astype(np.int32)

        _cxtgeo.swig_numpy_to_carr_i1d(np_array, carr)

        return carr

    def _convert_np_carr_double(self, np_array):
        """Convert numpy 1D array to C array, assuming double type."""
        carr = _cxtgeo.new_doublearray(self.nrow)

        _cxtgeo.swig_numpy_to_carr_1d(np_array, carr)

        return carr

    def _convert_carr_double_np(self, carray, nlen=None):
        """Convert a C array to numpy, assuming double type."""
        if nlen is None:
            nlen = len(self._df.index)

        nparray = _cxtgeo.swig_carr_to_numpy_1d(nlen, carray)

        return nparray
