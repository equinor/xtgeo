# -*- coding: utf-8 -*-
"""XTGeo well module, working with one single well"""

from __future__ import print_function, absolute_import

import sys
from copy import deepcopy
from distutils.version import StrictVersion

import numpy as np
import pandas as pd

import xtgeo
import xtgeo.cxtgeo._cxtgeo as _cxtgeo
import xtgeo.common.constants as const

from . import _wellmarkers
from . import _well_io
from . import _well_roxapi
from . import _well_oper

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


# pylint: disable=too-many-public-methods


# ======================================================================================
# METHODS as wrappers to class init + import


def well_from_file(
    wfile,
    fformat="rms_ascii",
    mdlogname=None,
    zonelogname=None,
    lognames="all",
    lognames_strict=False,
    strict=False,
):
    """Make an instance of a Well directly from file import.

    Args:
        mfile (str): File path, either a string or a pathlib.Path instance
        fformat (str): See :meth:`Well.from_file`
        mdlogname (str): See :meth:`Well.from_file`
        zonelogname (str): See :meth:`Well.from_file`
        lognames (str or list): Name or list of lognames to import, default is "all"
        strict (bool): See :meth:`Well.from_file`

    Example::

        import xtgeo
        mywell = xtgeo.well_from_file('somewell.xxx')

    .. versionchanged:: 2.1.0 Added ``lognames`` and ``lognames_strict``
    .. versionchanged:: 2.1.0 ``strict`` now defaults to False

    """

    obj = Well()

    obj.from_file(
        wfile,
        fformat=fformat,
        mdlogname=mdlogname,
        zonelogname=zonelogname,
        strict=strict,
        lognames=lognames,
        lognames_strict=lognames_strict,
    )

    return obj


def well_from_roxar(
    project,
    name,
    trajectory="Drilled trajectory",
    logrun="log",
    lognames="all",
    lognames_strict=False,
    inclmd=False,
    inclsurvey=False,
):

    """This makes an instance of a Well directly from Roxar RMS.

    For arguments, see :meth:`Well.from_roxar`.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mywell = xtgeo.well_from_roxar(project, '31_3-1', trajectory='Drilled',
                                       logrun='log', lognames=mylogs)

    .. versionchanged:: 2.1.0 lognames defaults to "all", not None
    """

    obj = Well()

    obj.from_roxar(
        project,
        name,
        trajectory=trajectory,
        logrun=logrun,
        lognames=lognames,
        lognames_strict=lognames_strict,
        inclmd=inclmd,
        inclsurvey=inclsurvey,
    )

    return obj


# ======================================================================================
# CLASS


class Well(object):  # pylint: disable=useless-object-inheritance
    """Class for a well in the XTGeo framework.

    The well logs are stored as Pandas dataframe, which make manipulation
    easy and fast.

    The well trajectory are here represented as logs, and XYZ have magic names:
    X_UTME, Y_UTMN, Z_TVDSS, which are the three first Pandas columns.

    Other geometry logs has also 'semi-magic' names:

    M_MDEPTH or Q_MDEPTH: Measured depth, either real/true (M_xx) or
    quasi computed/estimated (Q_xx). The Quasi may be incorrect for
    all uses, but sufficient for some computations.

    Similar for M_INCL, Q_INCL, M_AZI, Q_ASI.

    All Pandas values (yes, discrete also!) are stored as float64
    format, and undefined values are Nan. Integers are stored as Float due
    to the lacking support for 'Integer Nan' (currently lacking in Pandas,
    but may come in later Pandas versions).

    Note there is a method that can return a dataframe (copy) with Integer
    and Float columns, see :meth:`get_filled_dataframe`.

    The instance can be made either from file or (todo!) by spesification::

        >>> well1 = Well('somefilename')  # assume RMS ascii well
        >>> well2 = Well('somefilename', fformat='rms_ascii')
        >>> well3 = xtgeo.well_from_file('somefilename')

    For arguments, see method under :meth:`from_file`.

    """

    VALID_LOGTYPES = {"DISC", "CONT"}

    def __init__(self, *args, **kwargs):

        # instance attributes for whole well
        self._rkb = None  # well RKB height
        self._xpos = None  # well head X pos
        self._ypos = None  # well head Y pos
        self._wname = None  # well name
        self._filesrc = None  # source file if any

        # instance attributes well log names
        self._wlognames = list()  # A list of log names
        self._wlogtype = dict()  # dictionary of log types, 'DISC' or 'CONT'
        self._wlogrecord = dict()  # code record for 'DISC' logs
        self._mdlogname = None
        self._zonelogname = None

        self._df = None  # pandas dataframe with all log values

        if args:
            # make instance from file import
            wfile = args[0]
            fformat = kwargs.get("fformat", "rms_ascii")
            mdlogname = kwargs.get("mdlogname", None)
            zonelogname = kwargs.get("zonelogname", None)
            strict = kwargs.get("strict", False)
            lognames = kwargs.get("lognames", "all")
            self.from_file(
                wfile,
                fformat=fformat,
                mdlogname=mdlogname,
                zonelogname=zonelogname,
                lognames=lognames,
                strict=strict,
            )

        else:
            # dummy
            self._xx = kwargs.get("xx", 0.0)

            # # make instance by kw spesification ... todo
            # raise RuntimeWarning('Cannot initialize a Well object without '
            #                      'import at the current stage.')

        self._ensure_consistency()
        logger.info("Ran __init__for Well() %s", id(self))

    def __repr__(self):
        # should be able to newobject = eval(repr(thisobject))
        myrp = (
            "{0.__class__.__name__} (filesrc={0._filesrc!r}, "
            "name={0._wname!r},  ID={1})".format(self, id(self))
        )
        return myrp

    def __str__(self):
        # user friendly print
        return self.describe(flush=False)

    def __del__(self):
        logger.info("Deleting %s instance %s", self.__class__.__name__, id(self))

    # Consistency checking. As well log names are columns in the Pandas DF,
    # there are additional attributes per log that have to be "in sync"
    def _ensure_consistency(self):  # pragma: no coverage
        """Ensure consistency within an object (private function)"""

        if self._df is None:
            return

        self._wlognames = list(self._df.columns)

        for logname in self._wlognames:
            if logname not in self._wlogtype:
                self._wlogtype[logname] = "CONT"  # continuous as default
                self._wlogrecord[logname] = None  # None as default
            else:
                if self._wlogtype[logname] not in self.VALID_LOGTYPES:
                    self._wlogtype[logname] = "CONT"
                    self._wlogrecord[logname] = None  # None as default

            if logname not in self._wlogrecord:
                if self._wlogtype[logname] == "DISC":
                    # it is a discrete log with missing record; try to find
                    # a default one based on current values...
                    lvalues = self._df[logname].values.round(decimals=0)
                    lmin = int(lvalues.min())
                    lmax = int(lvalues.max())

                    lvalues = lvalues.astype("int")
                    codes = {}
                    for lval in range(lmin, lmax + 1):
                        if lval in lvalues:
                            codes[lval] = str(lval)

                    self._wlogrecord = codes

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def rkb(self):
        """ Returns RKB height for the well (read only)."""
        return self._rkb

    @property
    def xpos(self):
        """ Returns well header X position (read only)."""
        return self._xpos

    @property
    def ypos(self):
        """ Returns well header Y position (read only)."""
        return self._ypos

    @property
    def wellname(self):
        """ Returns well name (read only) (see also name attribute)."""
        return self._wname

    @property
    def name(self):
        """ Returns or set (rename) a well name."""
        return self._wname

    @name.setter
    def name(self, newname):
        self._wname = newname

    @property
    def xwellname(self):
        """Returns well name on a file syntax safe form (/ and space replaced
        with _).
        """
        xname = self._wname
        xname = xname.replace("/", "_")
        xname = xname.replace(" ", "_")
        return xname

    @property
    def shortwellname(self):
        """Returns well name on a short name form where blockname and spaces
        are removed (read only).

        This should cope with both North Sea style and Haltenbanken style.

        E.g.: '31/2-G-5 AH' -> 'G-5AH', '6472_11-F-23_AH_T2' -> 'F-23AHT2'

        """
        return self.get_short_wellname(self.wellname)

    @property
    def truewellname(self):
        """Returns well name on the assummed form aka '31/2-E-4 AH2'."""
        xname = self.xwellname
        if "/" not in xname:
            xname = xname.replace("_", "/", 1)
            xname = xname.replace("_", " ")
        return xname

    @property
    def mdlogname(self):
        """ Returns name of MD log, if any (None if missing)."""
        return self._mdlogname

    @mdlogname.setter
    def mdlogname(self, mname):
        if mname in self._wlognames:
            self._mdlogname = mname
        else:
            self._mdlogname = None

    @property
    def zonelogname(self):
        """ Returns or sets name of zone log, return None if missing."""
        return self._zonelogname

    @zonelogname.setter
    def zonelogname(self, zname):
        if zname in self._wlognames:
            self._zonelogname = zname
        else:
            self._zonelogname = None

    @property
    def dataframe(self):
        """Returns or set the Pandas dataframe object for all logs."""
        return self._df

    @dataframe.setter
    def dataframe(self, dfr):
        self._df = dfr.copy()
        self._ensure_consistency()

    @property
    def nrow(self):
        """Returns the Pandas dataframe object number of rows"""
        return len(self._df.index)

    @property
    def ncol(self):
        """Returns the Pandas dataframe object number of columns"""
        return len(self._df.columns)

    @property
    def nlogs(self):
        """Returns the Pandas dataframe object number of columns"""
        return len(self._df.columns) - 3

    @property
    def lognames_all(self):
        """Returns the Pandas dataframe column names as list (including
        mandatory X_UTME Y_UTMN Z_TVDSS)."""
        self._ensure_consistency()
        return self._wlognames

    @property
    def lognames(self):
        """Returns the Pandas dataframe column as list (excluding
        mandatory X_UTME Y_UTMN Z_TVDSS)"""
        return list(self._df)[3:]

    # ==================================================================================
    # Methods
    # ==================================================================================

    @staticmethod
    def get_short_wellname(wellname):
        """Returns well name on a short name form where blockname and spaces
        are removed (read only).
        This should cope with both North Sea style and Haltenbanken style.
        E.g.: '31/2-G-5 AH' -> 'G-5AH', '6472_11-F-23_AH_T2' -> 'F-23AHT2'
        """
        newname = []
        first1 = False
        first2 = False
        for letter in wellname:
            if first1 and first2:
                newname.append(letter)
                continue
            if letter in ("_", "/"):
                first1 = True
                continue
            if first1 and letter == "-":
                first2 = True
                continue

        xname = "".join(newname)
        xname = xname.replace("_", "")
        xname = xname.replace(" ", "")
        return xname

    def from_file(
        self,
        wfile,
        fformat="rms_ascii",
        mdlogname=None,
        zonelogname=None,
        strict=False,
        lognames="all",
        lognames_strict=False,
    ):
        """Import well from file.

        Args:
            wfile (str): Name of file as string or pathlib.Path
            fformat (str): File format, rms_ascii (rms well) is
                currently supported and default format.
            mdlogname (str): Name of measured depth log, if any
            zonelogname (str): Name of zonation log, if any
            strict (bool): If True, then import will fail if
                zonelogname or mdlogname are asked for but not present
                in wells. If False, and e.g. zonelogname is not present, the
                attribute ``zonelogname`` will be set to None.
            lognames (str or list): Name or list of lognames to import, default is "all"
            lognames_strict (bool): Flag to require all logs in lognames (unless "all")
                or to just accept that subset that is present. Default is `False`.


        Returns:
            Object instance (optionally)

        Example:
            Here the from_file method is used to initiate the object
            directly::

            >>> mywell = Well('31_2-6.w')

        .. versionchanged:: 2.1.0 ``lognames`` and ``lognames_strict`` added
        .. versionchanged:: 2.1.0 ``strict`` now defaults to False
        """

        wfile = xtgeo._XTGeoFile(wfile)

        wfile.check_file(raiseerror=OSError)

        if fformat is None or fformat == "rms_ascii":
            _well_io.import_rms_ascii(
                self,
                wfile.name,
                mdlogname=mdlogname,
                zonelogname=zonelogname,
                strict=strict,
                lognames=lognames,
                lognames_strict=lognames_strict,
            )
        else:
            logger.error("Invalid file format")

        self._ensure_consistency()
        self._filesrc = wfile.name
        return self

    def to_file(self, wfile, fformat="rms_ascii"):
        """
        Export well to file

        Args:
            wfile (str): Name of file or pathlib.Path instance
            fformat (str): File format ('rms_ascii'/'rmswell', 'hdf5')

        Example::

            xwell = Well("somefile.rmswell")
            xwell.dataframe['PHIT'] += 0.1
            xwell.to_file("somefile_copy.rmswell")

        """
        wfile = xtgeo._XTGeoFile(wfile, mode="wb")

        wfile.check_folder(raiseerror=OSError)

        self._ensure_consistency()

        if fformat in (None, "rms_ascii", "rmswell"):
            _well_io.export_rms_ascii(self, wfile.name)

        elif fformat == "hdf5":
            with pd.HDFStore(wfile, "a", complevel=9, complib="zlib") as store:
                logger.info("export to HDF5 %s", wfile.name)
                store[self._wname] = self._df
                meta = dict()
                meta["name"] = self._wname
                store.get_storer(self._wname).attrs["metadata"] = meta

    def from_roxar(self, *args, **kwargs):
        """Import (retrieve) well from roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        Args:
            project (str): Magic string 'project' or file path to project
            wname (str): Name of well, as shown in RMS.
            lognames (:obj:list or :obj:str): List of lognames to import, or
                use simply 'all' for current logs for this well.
            lognames_strict (bool); Flag to require all logs or to just provide
                a subset that is present. Default is `False`.
            realisation (int): Currently inactive
            trajectory (str): Name of trajectory in RMS
            logrun (str): Name of logrun in RMS
            inclmd (bool): Include MDEPTH as log M_MEPTH from RMS
            inclsurvey (bool): Include M_AZI and M_INCL from RMS
        """

        # use *args, **kwargs since this method is overrided in blocked_well, and
        # signature should be the same

        project = args[0]
        wname = args[1]
        lognames = kwargs.get("lognames", "all")
        lognames_strict = kwargs.get("lognames_strict", False)
        trajectory = kwargs.get("trajectory", "Drilled trajecetry")
        logrun = kwargs.get("logrun", "log")
        inclmd = kwargs.get("inclmd", False)
        inclsurvey = kwargs.get("inclsurvey", False)
        realisation = kwargs.get("realisation", 0)

        logger.debug("Not in use: realisation %s", realisation)

        _well_roxapi.import_well_roxapi(
            self,
            project,
            wname,
            trajectory=trajectory,
            logrun=logrun,
            lognames=lognames,
            lognames_strict=lognames_strict,
            inclmd=inclmd,
            inclsurvey=inclsurvey,
        )
        self._ensure_consistency()

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = xtgeo.common.XTGDescription()
        dsc.title("Description of Well instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("File source", self._filesrc)
        dsc.txt("Well name", self._wname)
        dsc.txt("RKB", self._rkb)
        dsc.txt("Well head", self._xpos, self._ypos)
        dsc.txt("Name of all columns", self.lognames_all)
        dsc.txt("Name of log columns", self.lognames)
        for wlog in self.lognames:
            rec = self.get_logrecord(wlog)
            if rec is not None and len(rec) > 3:
                string = "("
                nlen = len(rec)
                for idx, (code, val) in enumerate(rec.items()):
                    if idx < 2:
                        string += "{}: {} ".format(code, val)
                    elif idx == nlen - 1:
                        string += "...  {}: {})".format(code, val)
            else:
                string = "{}".format(rec)
            dsc.txt("Logname", wlog, self.get_logtype(wlog), string)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    def isdiscrete(self, logname):
        """Return True of log is discrete, otherwise False.

        Args:
            logname (str): Name of log to check if discrete or not

        .. versionadded: 2.2.0
        """

        if logname in self._wlognames and self.get_logtype(logname) == "DISC":
            return True
        return False

    def copy(self):
        """Copy a Well instance to a new unique Well instance."""

        # pylint: disable=protected-access

        new = Well()
        new._wlogtype = deepcopy(self._wlogtype)
        new._wlogrecord = deepcopy(self._wlogrecord)
        new._rkb = self._rkb
        new._xpos = self._xpos = None
        new._ypos = self._ypos
        new._wname = self._wname
        if self._df is None:
            new._df = None
        else:
            new._df = self._df.copy()
        new._mdlogname = self._mdlogname
        new._zonelogname = self._zonelogname

        new._ensure_consistency()
        return new

    def rename_log(self, lname, newname):
        """Rename a log, e.g. Poro to PORO"""
        self._ensure_consistency()

        if lname not in self.lognames:
            raise ValueError("Input log does not exist")

        if newname in self.lognames:
            raise ValueError("New log name exists already")

        self._wlogtype[newname] = self._wlogtype.pop(lname)
        self._wlogrecord[newname] = self._wlogrecord.pop(lname)

        # rename in dataframe
        self._df.rename(index=str, columns={lname: newname}, inplace=True)

        if self._mdlogname == lname:
            self._mdlogname = newname

        if self._zonelogname == lname:
            self._zonelogname = newname

    def create_log(self, lname, logtype="CONT", logrecord=None, value=0.0, force=True):
        """Create a new log with initial values.

        If the logname already exists, it will be silently overwritten, unless
        the option force=False.

        Args:
            lname (str): name of new log
            logtype (str): Must be 'CONT' (default) or 'DISC' (discrete)
            logrecord (dict): A dictionary of key: values for 'DISC' logs
            value (float): initia value to set_index
            force (bool): If True, and lname exists, it will be overwritten, if
               False, no new log will be made. Will return False.

        Returns:
            True ff a new log is made (either new or force overwrite an
            existing) or False if the new log already exists,
            and ``force=False``.

        """

        if lname in self.lognames and force is False:
            return False

        self._wlogtype[lname] = logtype
        self._wlogrecord[lname] = logrecord

        # make a new column
        self._df[lname] = float(value)
        self._ensure_consistency()
        return True

    def delete_log(self, lname):
        """Delete/remove an existing log, or list of logs.

        Will continue silently if a log does not exist.

        Args:
            lname(str or list): A logname or a list of lognames

        Returns:
            Number of logs deleted
        """

        return _well_oper.delete_log(self, lname)

    delete_logs = delete_log  # alias function

    def get_logtype(self, lname):
        """Returns the type of a give log (e.g. DISC or CONT)"""
        self._ensure_consistency()

        if lname in self._wlogtype:
            return self._wlogtype[lname]
        return None

    def set_logtype(self, lname, ltype):
        """Sets the type of a give log (e.g. DISC or CONT)"""

        self._ensure_consistency()

        valid = {"DISC", "CONT"}

        if ltype in valid:
            self._wlogtype[lname] = ltype
        else:
            raise ValueError("Try to set invalid log type: {}".format(ltype))

    def get_logrecord(self, lname):
        """Returns the record (dict) of a give log. None if not exists"""

        if lname in self._wlogtype:
            return self._wlogrecord[lname]

        return None

    def set_logrecord(self, lname, newdict):
        """Sets the record (dict) of a given discrete log"""

        self._ensure_consistency()
        if lname not in self.lognames:
            raise ValueError("No such logname: {}".format(lname))

        if self._wlogtype[lname] == "CONT":
            raise ValueError("Cannot set a log record for a continuous log")

        if not isinstance(newdict, dict):
            raise ValueError("Input is not a dictionary")

        self._wlogrecord[lname] = newdict

    def get_logrecord_codename(self, lname, key):
        """Returns the name entry of a log record, for a given key

        Example::

            # get the name for zonelog entry no 4:
            zname = well.get_logrecord_codename('ZONELOG', 4)
        """

        zlogdict = self.get_logrecord(lname)
        if key in zlogdict:
            return zlogdict[key]

        return None

    def get_carray(self, lname):
        """Returns the C array pointer (via SWIG) for a given log.

        Type conversion is double if float64, int32 if DISC log.
        Returns None of log does not exist.
        """
        if lname in self._df:
            np_array = self._df[lname].values
        else:
            return None

        if self.get_logtype(lname) == "DISC":
            carr = self._convert_np_carr_int(np_array)
        else:
            carr = self._convert_np_carr_double(np_array)

        return carr

    def get_filled_dataframe(
        self, fill_value=const.UNDEF, fill_value_int=const.UNDEF_INT
    ):
        """Fill the Nan's in the dataframe with real UNDEF values.

        This module returns a copy of the dataframe in the object; it
        does not change the instance.

        Note that DISC logs will be casted to columns with integer
        as datatype.

        Returns:
            A pandas dataframe where Nan er replaces with preset
                high XTGeo UNDEF values, or user defined values.

        """

        lnames = self.lognames

        newdf = self._df.copy()

        # make a dictionary of datatypes
        dtype = {"X_UTME": "float64", "Y_UTMN": "float64", "Z_TVDSS": "float64"}

        dfill = {"X_UTME": const.UNDEF, "Y_UTMN": const.UNDEF, "Z_TVDSS": const.UNDEF}

        for lname in lnames:
            if self.get_logtype(lname) == "DISC":
                dtype[lname] = np.int32
                dfill[lname] = fill_value_int
            else:
                dtype[lname] = np.float64
                dfill[lname] = fill_value

        # now first fill Nan's (because int cannot be converted if Nan)
        newdf = newdf.fillna(dfill)

        # now cast to dtype (dep on Pandas version)
        if StrictVersion(pd.__version__) >= StrictVersion("0.19.0"):
            newdf = newdf.astype(dtype)
        else:
            for k, var in dtype.items():
                newdf[k] = newdf[k].astype(var)

        return newdf

    def create_relative_hlen(self):
        """Make a relative length of a well, as a log.

        The first well og entry defines zero, then the horizontal length
        is computed relative to that by simple geometric methods.
        """

        # extract numpies from XYZ trajectory logs
        xv = self._df["X_UTME"].values
        yv = self._df["Y_UTMN"].values
        zv = self._df["Z_TVDSS"].values

        # get number of rows in pandas
        nlen = self.nrow

        ier, _tlenv, _dtlenv, hlenv, _dhlenv = _cxtgeo.pol_geometrics(
            xv, yv, zv, nlen, nlen, nlen, nlen
        )

        if ier != 0:
            raise RuntimeError(
                "Error code from _cxtgeo.pol_geometrics is {}".format(ier)
            )

        self._df["R_HLEN"] = pd.Series(hlenv, index=self._df.index)

    def geometrics(self):
        """Compute some well geometrical arrays MD, INCL, AZI, as logs.

        These are kind of quasi measurements hence the logs will named
        with a Q in front as Q_MDEPTH, Q_INCL, and Q_AZI.

        These logs will be added to the dataframe. If the mdlogname
        attribute does not exist in advance, it will be set to 'Q_MDEPTH'.

        Returns:
            False if geometrics cannot be computed

        """
        if self._df.size < 3:
            logger.warning(
                "Cannot compute geometrics for %s. Too few  " "trajectory points",
                self.name,
            )
            return False

        # extract numpies from XYZ trajetory logs
        ptr_xv = self.get_carray("X_UTME")
        ptr_yv = self.get_carray("Y_UTMN")
        ptr_zv = self.get_carray("Z_TVDSS")

        # get number of rows in pandas
        nlen = self.nrow

        ptr_md = _cxtgeo.new_doublearray(nlen)
        ptr_incl = _cxtgeo.new_doublearray(nlen)
        ptr_az = _cxtgeo.new_doublearray(nlen)

        ier = _cxtgeo.well_geometrics(
            nlen, ptr_xv, ptr_yv, ptr_zv, ptr_md, ptr_incl, ptr_az, 0,
        )

        if ier != 0:
            sys.exit(-9)

        dnumpy = self._convert_carr_double_np(ptr_md)
        self._df["Q_MDEPTH"] = pd.Series(dnumpy, index=self._df.index)

        dnumpy = self._convert_carr_double_np(ptr_incl)
        self._df["Q_INCL"] = pd.Series(dnumpy, index=self._df.index)

        dnumpy = self._convert_carr_double_np(ptr_az)
        self._df["Q_AZI"] = pd.Series(dnumpy, index=self._df.index)

        if not self._mdlogname:
            self._mdlogname = "Q_MDEPTH"

        # delete tmp pointers
        _cxtgeo.delete_doublearray(ptr_xv)
        _cxtgeo.delete_doublearray(ptr_yv)
        _cxtgeo.delete_doublearray(ptr_zv)
        _cxtgeo.delete_doublearray(ptr_md)
        _cxtgeo.delete_doublearray(ptr_incl)
        _cxtgeo.delete_doublearray(ptr_az)

        return True

    def truncate_parallel_path(
        self, other, xtol=None, ytol=None, ztol=None, itol=None, atol=None
    ):
        """Truncate (remove) the part of the well trajectory that is
        ~parallel with other.

        Args:
            other (Well): Other well to compare with
            xtol (float): Tolerance in X (East) coord for measuring unit
            ytol (float): Tolerance in Y (North) coord for measuring unit
            ztol (float): Tolerance in Z (TVD) coord for measuring unit
            itol (float): Tolerance in inclination (degrees)
            atol (float): Tolerance in azimuth (degrees)
        """

        if xtol is None:
            xtol = 0.0
        if ytol is None:
            ytol = 0.0
        if ztol is None:
            ztol = 0.0
        if itol is None:
            itol = 0.0
        if atol is None:
            atol = 0.0

        if self._df.size < 3 or other._df.size < 3:
            # xtg.warn('Too few points to truncate parallel path')
            return

        # extract numpies from XYZ trajectory logs
        xv1 = self._df["X_UTME"].values
        yv1 = self._df["Y_UTMN"].values
        zv1 = self._df["Z_TVDSS"].values

        xv2 = other._df["X_UTME"].values
        yv2 = other._df["Y_UTMN"].values
        zv2 = other._df["Z_TVDSS"].values

        ier = _cxtgeo.well_trunc_parallel(
            xv1, yv1, zv1, xv2, yv2, zv2, xtol, ytol, ztol, itol, atol, 0
        )

        if ier != 0:
            raise RuntimeError("Unexpected error")

        self._df = self._df[self._df["X_UTME"] < const.UNDEF_LIMIT]
        self._df.reset_index(drop=True, inplace=True)

    def may_overlap(self, other):
        """Consider if well may overlap in X Y coordinates with other well,
        True/False
        """

        if self._df.size < 2 or other._df.size < 2:
            return False

        # extract numpies from XYZ trajectory logs
        xmin1 = np.nanmin(self.dataframe["X_UTME"].values)
        xmax1 = np.nanmax(self.dataframe["X_UTME"].values)
        ymin1 = np.nanmin(self.dataframe["Y_UTMN"].values)
        ymax1 = np.nanmax(self.dataframe["Y_UTMN"].values)

        xmin2 = np.nanmin(other.dataframe["X_UTME"].values)
        xmax2 = np.nanmax(other.dataframe["X_UTME"].values)
        ymin2 = np.nanmin(other.dataframe["Y_UTMN"].values)
        ymax2 = np.nanmax(other.dataframe["Y_UTMN"].values)

        if xmin1 > xmax2 or ymin1 > ymax2:
            return False
        if xmin2 > xmax1 or ymin2 > ymax1:
            return False

        return True

    def limit_tvd(self, tvdmin, tvdmax):
        """Truncate the part of the well that is outside tvdmin, tvdmax.

        Range will be in tvdmin <= tvd <= tvdmax.

        Args:
            tvdmin (float): Minimum TVD
            tvdmax (float): Maximum TVD
        """
        self._df = self._df[self._df["Z_TVDSS"] >= tvdmin]
        self._df = self._df[self._df["Z_TVDSS"] <= tvdmax]

        self._df.reset_index(drop=True, inplace=True)

    def downsample(self, interval=4, keeplast=True):
        """Downsample by sampling every N'th element (coarsen only).

        Args:
            interval (int): Sampling interval.
            keeplast (bool): If True, the last element from the original
                dataframe is kept, to avoid that the well is shortened.
        """

        if self._df.size < 2 * interval:
            return

        dfr = self._df[::interval]

        if keeplast:
            dfr.append(self._df.iloc[-1])

        self._df = dfr.reset_index(drop=True)

    def rescale(self, delta=0.15, tvdrange=None):
        """Rescale (refine or coarse) a well by sampling a delta along the
        trajectory, in MD.

        Args:
            delta (float): Step length
            tvdrange (tuple of floats): Resampling can be limited to TVD interval

        .. versionchanged:: 2.2.0 Added tvdrange
        """
        _well_oper.rescale(self, delta=delta, tvdrange=tvdrange)

    def get_polygons(self):
        """Return a Polygons object from the well trajectory.

        .. versionadded:: 2.1.0
        """

        dfr = self._df.copy()

        keep = ("X_UTME", "Y_UTMN", "Z_TVDSS")
        for col in dfr.columns:
            if col not in keep:
                dfr.drop(labels=col, axis=1, inplace=True)
        dfr["POLY_ID"] = 1
        dfr["NAME"] = self.xwellname
        poly = xtgeo.Polygons()
        poly.dataframe = dfr
        poly.name = self.xwellname

        return poly

    def get_fence_polyline(self, sampling=20, nextend=2, tvdmin=None, asnumpy=True):
        """Return a fence polyline as a numpy array or a Polygons object.

        The result will aim for a regular sampling interval, useful for extracting
        fence plots (cross-sections).

        Args:
            sampling (float): Sampling interval i.e. horizonal distance (input)
            nextend (int): Number if sampling to extend; e.g. 2 * 20
            tvdmin (float): Minimum TVD starting point.
            as_numpy (bool): If True, a numpy array, otherwise a Polygons
                object with 5 columns where the 2 last are HLEN and POLY_ID
                and the POLY_ID will be set to 0.

        Returns:
            A numpy array of shape (NLEN, 5) in F order,
            Or a Polygons object with 5 columns
            If not possible, return False

        .. versionchanged:: 2.1.0 improved algorithm
        """

        poly = self.get_polygons()

        if tvdmin is not None:
            poly.dataframe = poly.dataframe[poly.dataframe[poly.zname] >= tvdmin]
            poly.dataframe.reset_index(drop=True, inplace=True)

        return poly.get_fence(distance=sampling, nextend=nextend, asnumpy=asnumpy)

    def report_zonation_holes(self, threshold=5):
        """Reports if well has holes in zonation, less or equal to N samples.

        Zonation may have holes due to various reasons, and
        usually a few undef samples indicates that something is wrong.
        This method reports well and start interval of the "holes"

        The well shall have zonelog from import (via zonelogname attribute) and
        preferly a MD log (via mdlogname attribute); however if the
        latter is not present, a report withou MD values will be present.

        Args:
            threshold (int): Number of samples (max.) that defines a hole, e.g.
                5 means that undef samples in the range [1, 5] (including 5) is
                applied

        Returns:
            A Pandas dataframe as a report. None if no list is made.

        Raises:
            RuntimeError if zonelog is not present
        """

        dfr = _well_oper.report_zonation_holes(self, threshold=threshold)

        return dfr

    def get_zonation_points(
        self, tops=True, incl_limit=80, top_prefix="Top", zonelist=None, use_undef=False
    ):

        """Extract zonation points from Zonelog and make a marker list.

        Currently it is either 'Tops' or 'Zone' (thicknesses); default
        is tops (i.e. tops=True).

        The `zonelist` can be a list of zones, or a tuple with two members specifying
        first and last member. Note however that the zonation shall be without jumps
        and increasing. E.g.::

            zonelist=(1, 5)  # meaning [1, 2, 3, 4, 5]
            # or
            zonelist=[1, 2, 3, 4]
            # while _not_ legal:
            zonelist=[1, 4, 8]

        Zone numbers less than 0 are not accepted

        Args:
            tops (bool): If True then compute tops, else (thickness) points.
            incl_limit (float): If given, and usezone is True, the max
                angle of inclination to be  used as input to zonation points.
            top_prefix (str): As well logs usually have isochore (zone) name,
                this prefix could be Top, e.g. 'SO43' --> 'TopSO43'
            zonelist (list of int or tuple): Zones to use
            use_undef (bool): If True, then transition from UNDEF is also
                used.


        Returns:
            A pandas dataframe (ready for the xyz/Points class), None
            if a zonelog is missing
        """

        # make a copy of the well instance as some tmp well logs are made
        scopy = self.copy()

        dfr = _wellmarkers.get_zonation_points(
            scopy, tops, incl_limit, top_prefix, zonelist, use_undef
        )

        del scopy

        return dfr

    def get_zone_interval(self, zonevalue, resample=1, extralogs=None):

        """Extract the X Y Z ID line (polyline) segment for a given
        zonevalue.

        Args:
            zonevalue (int): The zone value to extract
            resample (int): If given, downsample every N'th sample to make
                polylines smaller in terms of bit and bytes.
                1 = No downsampling.
            extralogs (list of str): List of extra log names to include


        Returns:
            A pandas dataframe X Y Z ID (ready for the xyz/Polygon class),
            None if a zonelog is missing or actual zone does dot
            exist in the well.
        """

        if resample < 1 or not isinstance(resample, int):
            raise KeyError("Key resample of wrong type (must be int >= 1)")

        dff = self.get_filled_dataframe()

        # the technical solution here is to make a tmp column which
        # will add one number for each time the actual segment is repeated,
        # not straightforward... (thanks to H. Berland for tip)

        dff["ztmp"] = dff[self.zonelogname]
        dff["ztmp"] = (dff[self.zonelogname] != zonevalue).astype(int)

        dff["ztmp"] = (dff.ztmp != dff.ztmp.shift()).cumsum()

        dff = dff[dff[self.zonelogname] == zonevalue]

        m1v = dff["ztmp"].min()
        m2v = dff["ztmp"].max()
        if np.isnan(m1v):
            logger.debug("Returns (no data)")
            return None

        df2 = dff.copy()

        dflist = []
        for mvv in range(m1v, m2v + 1):
            dff9 = df2.copy()
            dff9 = df2[df2["ztmp"] == mvv]
            if dff9.index.shape[0] > 0:
                dflist.append(dff9)

        dxlist = []

        useloglist = ["X_UTME", "Y_UTMN", "Z_TVDSS", "POLY_ID"]
        if extralogs is not None:
            useloglist.extend(extralogs)

        # pylint: disable=consider-using-enumerate
        for ivv in range(len(dflist)):
            dxf = dflist[ivv]
            dxf = dxf.rename(columns={"ztmp": "POLY_ID"})
            cols = [xxx for xxx in dxf.columns if xxx not in useloglist]

            dxf = dxf.drop(cols, axis=1)

            # now (down) resample every N'th
            if resample > 1:
                dxf = pd.concat([dxf.iloc[::resample, :], dxf.tail(1)])

            dxlist.append(dxf)

        dff = pd.concat(dxlist)
        dff.reset_index(inplace=True, drop=True)

        logger.debug("Dataframe from well:\n%s", dff)
        return dff

    def get_fraction_per_zone(
        self,
        dlogname,
        dcodes,
        zonelist=None,
        incl_limit=80,
        count_limit=3,
        zonelogname=None,
    ):

        """Get fraction of a discrete parameter, e.g. a facies, per zone.

        It can be constrained by an inclination.

        Also, it needs to be evaluated only of ZONE is complete; either
        INCREASE or DECREASE ; hence a quality flag is made and applied.

        Args:
            dlogname (str): Name of discrete log, e.g. 'FACIES'
            dnames (list of int): Codes of facies (or similar) to report for
            zonelist (list of int): Zones to use
            incl_limit (float): Inclination limit for well path.
            count_limit (int): Minimum number of counts required per segment
                for valid calculations
            zonelogname (str). If None, the Well().zonelogname attribute is
                applied

        Returns:
            A pandas dataframe (ready for the xyz/Points class), None
            if a zonelog is missing or or dlogname is missing,
            list is zero length for any reason.
        """

        dfr = _wellmarkers.get_fraction_per_zone(
            self,
            dlogname,
            dcodes,
            zonelist=zonelist,
            incl_limit=incl_limit,
            count_limit=count_limit,
            zonelogname=zonelogname,
        )

        return dfr

    def get_surface_picks(self, surf):
        """Get a :class:`.Points` instance where well crosses the surface (horizon
        picks).

        There may be several points in the Points() dataframe attribute.
        Also a ``DIRECTION`` column will show 1 if surface is penetrated from
        above, and -1 if penetrated from below.

        Args:
            surf (RegularSurface): The surface instance

        Returns:
            A :class:`.Points` instance, or None if no crossing points

        .. versionadded:: 2.8.0

        """

        return _wellmarkers.get_surface_picks(self, surf)

    def make_ijk_from_grid(self, grid, grid_id="", algorithm=2, activeonly=True):
        """Look through a Grid and add grid I J K as discrete logs.

        Note that the the grid counting has base 1 (first row is 1 etc).

        By default, log (i.e. column names in the dataframe) will be
        ICELL, JCELL, KCELL, but you can add a tag (ID) to that name.

        Args:
            grid (Grid): A XTGeo Grid instance
            grid_id (str): Add a tag (optional) to the current log name
            algorithm (int): Which interbal algorithm to use, default is 2 (expert
                setting)
            activeonly (bool): If True, only active cells are applied (algorithm 2 only)
        Raises:
            RuntimeError: 'Error from C routine, code is ...'

        .. versionchanged:: 2.9.0 Added keys for and `activeonly`
        """

        _well_oper.make_ijk_from_grid(
            self, grid, grid_id=grid_id, algorithm=algorithm, activeonly=activeonly,
        )

    def make_zone_qual_log(self, zqname):
        """Create a zone quality/indicator (flag) log.

        This routine looks through to zone log and flag intervals according
        to neighbouring zones:

        * 0: Undetermined flag

        * 1: Zonelog interval numbering increases,
             e.g. for zone 2: 1 1 1 1 2 2 2 2 2 5 5 5 5 5

        * 2: Zonelog interval numbering decreases,
             e.g. for zone 2: 6 6 6 2 2 2 2 1 1 1

        * 3: Interval is a U turning point, e.g. 0 0 0 2 2 2 1 1 1

        * 4: Interval is a inverse U turning point, 3 3 3 2 2 2 5 5

        * 9: Interval is bounded by one or more missing sections,
             e.g. 1 1 1 2 2 2 -999 -999

        If a log with the name exists, it will be silently replaced

        Args:
            zqname (str): Name of quality log
        """

        _well_oper.make_zone_qual_log(self, zqname)

    def get_gridproperties(
        self, gridprops, grid=("ICELL", "JCELL", "KCELL"), prop_id="_model"
    ):
        """Look through a Grid and add a set of grid properties as logs.

        The name of the logs will ...

        This can be done to sample model properties along a well.

        Args:
            gridprops (Grid): A XTGeo GridProperties instance (a collection
                of properties) or a single GridProperty instance
            grid (Grid or tuple): A XTGeo Grid instance or a reference
                via tuple. If this is tuple with log names,
                it states that these logs already contains
                the gridcell IJK numbering.
            prop_id (str): Add a tag (optional) to the current log name, e.g
                as PORO_model, where _model is the tag.

        Raises:
            None

        ..versionadded:: 2.1.0

        """

        _well_oper.get_gridproperties(self, gridprops, grid=grid, prop_id=prop_id)

    # ==================================================================================
    # PRIVATE METHODS
    # should not be applied outside the class
    # ==================================================================================

    # ----------------------------------------------------------------------------------
    # Import/Export methods for various formats
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Special methods for nerds, todo is to move to private module
    # ----------------------------------------------------------------------------------

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
