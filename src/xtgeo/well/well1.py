# -*- coding: utf-8 -*-
"""XTGeo well module, working with one single well."""

import sys
from copy import deepcopy
from distutils.version import StrictVersion
from typing import Union, Optional, List, Dict
from pathlib import Path
import io
from collections import OrderedDict

import numpy as np
import pandas as pd

import xtgeo
import xtgeo.common.constants as const
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

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
    wfile: Union[str, Path],
    fformat: Optional[str] = "rms_ascii",
    mdlogname: Optional[str] = None,
    zonelogname: Optional[str] = None,
    lognames: Optional[Union[str, List[str]]] = "all",
    lognames_strict: Optional[bool] = False,
    strict: Optional[bool] = False,
) -> "Well":
    """Make an instance of a Well directly from file import.

    Args:
        wfile: File path, either a string or a pathlib.Path instance
        fformat: See :meth:`Well.from_file`
        mdlogname: Name of Measured Depth log if any, see :meth:`Well.from_file`
        zonelogname: Name of Zonelog, if any, see :meth:`Well.from_file`
        lognames: Name or list of lognames to import, default is "all"
        lognames_strict: If True, all lognames must be present.
        strict: If True, then import will fail if zonelogname or mdlogname are asked
            for but not present in wells. See :meth:`Well.from_file`

    Example::

        import xtgeo
        mywell = xtgeo.well_from_file("somewell.xxx")

    .. versionchanged:: 2.1 Added ``lognames`` and ``lognames_strict``
    .. versionchanged:: 2.1 ``strict`` now defaults to False
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
    project: Union[str, object],
    name: str,
    trajectory: Optional[str] = "Drilled trajectory",
    logrun: Optional[str] = "log",
    lognames: Optional[Union[str, List[str]]] = "all",
    lognames_strict: Optional[bool] = False,
    inclmd: Optional[bool] = False,
    inclsurvey: Optional[bool] = False,
) -> "Well":
    """This makes an instance of a Well directly from Roxar RMS.

    For further details, see :meth:`Well.from_roxar`.

    Args:
        project: Path to project or magic ``project`` variable in RMS.
        name: Name of Well
        trajectory: Name of trajectory in RMS.
        logrun: Name of logrun in RMS.
        lognames: List of lognames to import or use 'all' for all present logs
        lognames_strict: If True and log is not in lognames is a list, an Exception will
            be raised.
        inclmd: If True, a Measured Depth log will be included.
        inclsurvey: If True, logs for azimuth and deviation will be included.

    Returns:
        Well instance.

    Example::

        # inside RMS:
        import xtgeo
        mylogs = ['ZONELOG', 'GR', 'Facies']
        mywell = xtgeo.well_from_roxar(
            project, "31_3-1", trajectory="Drilled", logrun="log", lognames=mylogs
        )

    .. versionchanged:: 2.1 lognames defaults to "all", not None
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


class Well:
    """Class for a well in the XTGeo framework.

    The well logs are stored in a Pandas dataframe, which make manipulation
    easy and fast.

    The well trajectory are here represented as logs, and XYZ have magic names:
    ``X_UTME``, ``Y_UTMN``, ``Z_TVDSS``, which are the three first Pandas columns.

    Other geometry logs has also 'semi-magic' names:

    M_MDEPTH or Q_MDEPTH: Measured depth, either real/true (M_xx) or
    quasi computed/estimated (Q_xx). The Quasi may be incorrect for
    all uses, but sufficient for some computations.

    Similar for M_INCL, Q_INCL, M_AZI, Q_ASI.

    All Pandas values (yes, discrete also!) are currently stored as float64
    format, and undefined values are Nan. Integers are stored as Float due
    to the (historic) lacking support for 'Integer Nan'. In coming versions,
    use of ``pandas.NA`` (available from Pandas version 1.0) may be implemented.

    Note there is a method that can return a dataframe (copy) with Integer
    and Float columns, see :meth:`get_filled_dataframe`.

    The instance can be made either from file or (todo!) by specification::

        >>> well1 = Well('somefilename')  # assume RMS ascii well
        >>> well2 = Well('somefilename', fformat='rms_ascii')
        >>> well3 = xtgeo.well_from_file('somefilename')

    """

    VALID_LOGTYPES = {"DISC", "CONT"}

    def __init__(
        self,
        wfile: Optional[Union[str, Path]] = None,
        fformat: Optional[str] = "rms_ascii",
        mdlogname: Optional[str] = None,
        zonelogname: Optional[str] = None,
        strict: Optional[bool] = False,
        lognames: Optional[Union[str, list]] = "all",
    ):
        """Instantating a Well object.

        Args:
            wfile: Input file, or leave blank
            fformat: File format input, default is ``rms_ascii`` unless file extension
                tells us something else (e.g. hdf).
            mdlogname: Name of MD log column, e.g. 'MDepth'
            zonelogname: Name of zonelog column, .e.g. 'ZONELOG'
            strict: Applies to lognames, if True, then all names in ``lognames`` will
                be forced.
            lognames: A list of lognames to load, which makes it possible to load a
                subset of logs. A string "all" will load all current logs.

        """
        # instance attributes for whole well
        self._rkb = None  # well RKB height
        self._xpos = None  # well head X pos
        self._ypos = None  # well head Y pos
        self._wname = None  # well name
        self._filesrc = None  # source file if any
        self._mdlogname = None
        self._zonelogname = None

        # instance attributes well log names
        self._wlognames = list()  # A list of log names
        self._wlogtypes = dict()  # dictionary of log types, 'DISC' or 'CONT'
        self._wlogrecords = dict()  # code record for 'DISC' logs

        self._df = None  # pandas dataframe with all log values

        self._metadata = xtgeo.MetaDataWell()

        if wfile is not None:
            # make instance from file import
            wfile = Path(wfile)
            self.from_file(
                wfile,
                fformat=fformat,
                mdlogname=mdlogname,
                zonelogname=zonelogname,
                lognames=lognames,
                strict=strict,
            )

        else:
            logger.info("Instantate Well() object without file")

        self._ensure_consistency()
        self._metadata.required = self

        logger.info("Ran __init__for Well() %s", id(self))

    def __repr__(self):  # noqa: D105
        # should be able to newobject = eval(repr(thisobject))
        myrp = (
            "{0.__class__.__name__} (filesrc={0._filesrc!r}, "
            "name={0._wname!r},  ID={1})".format(self, id(self))
        )
        return myrp

    def __str__(self):  # noqa: D105
        # user friendly print
        return self.describe(flush=False)

    def _ensure_consistency(self):  # pragma: no coverage
        """Ensure consistency within an object (private function).

        Consistency checking. As well log names are columns in the Pandas DF,
        there are additional attributes per log that have to be "in sync".
        """
        if self._df is None:
            return

        self._wlognames = list(self._df.columns)

        for logname in self._wlognames:
            if logname not in self._wlogtypes:
                self._wlogtypes[logname] = "CONT"  # continuous as default
                self._wlogrecords[logname] = None  # None as default
            else:
                if self._wlogtypes[logname] not in self.VALID_LOGTYPES:
                    self._wlogtypes[logname] = "CONT"
                    self._wlogrecords[logname] = None  # None as default

            if logname not in self._wlogrecords:
                if self._wlogtypes[logname] == "DISC":
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

                    self._wlogrecords = codes

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def metadata(self):
        """Return metadata object instance of type MetaDataRegularSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. This is a bit dangerous so
        # further check must be done to validate. TODO.
        if not isinstance(obj, xtgeo.MetaDataWell):
            raise ValueError("Input obj not an instance of MetaDataRegularCube")

        self._metadata = obj

    @property
    def rkb(self):
        """Returns RKB height for the well (read only)."""
        return self._rkb

    @property
    def xpos(self):
        """Returns well header X position (read only)."""
        return self._xpos

    @property
    def ypos(self) -> float:
        """Returns well header Y position (read only)."""
        return self._ypos

    @property
    def wellname(self):
        """str: Returns well name, read only."""
        return self._wname

    @property
    def name(self):
        """Returns or set (rename) a well name."""
        return self._wname

    @name.setter
    def name(self, newname):
        self._wname = newname

    # alias
    wname = name

    @property
    def safewellname(self):
        """Get well name on syntax safe form; '/' and spaces replaced with '_'."""
        xname = self._wname
        xname = xname.replace("/", "_")
        xname = xname.replace(" ", "_")
        return xname

    @property
    def xwellname(self):
        """See safewellname."""
        return self.safewellname

    @property
    def shortwellname(self):
        """str: Well name on a short form where blockname/spaces removed (read only).

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
        """str: Returns name of MD log, if any (None if missing)."""
        return self._mdlogname

    @mdlogname.setter
    def mdlogname(self, mname):
        if mname in self._wlognames:
            self._mdlogname = mname
        else:
            self._mdlogname = None

    @property
    def zonelogname(self):
        """str: Returns or sets name of zone log, return None if missing."""
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
        """int: Returns the Pandas dataframe object number of rows."""
        return len(self._df.index)

    @property
    def ncol(self):
        """int: Returns the Pandas dataframe object number of columns."""
        return len(self._df.columns)

    @property
    def nlogs(self):
        """int: Returns the Pandas dataframe object number of columns."""
        return len(self._df.columns) - 3

    @property
    def lognames_all(self):
        """list: Returns dataframe column names as list, including mandatory coords."""
        self._ensure_consistency()
        return self._wlognames

    @property
    def lognames(self):
        """list: Returns the Pandas dataframe column as list excluding coords."""
        return list(self._df)[3:]

    # ==================================================================================
    # Methods
    # ==================================================================================

    @staticmethod
    def get_short_wellname(wellname):
        """Well name on a short name form where blockname and spaces are removed.

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

    def describe(self, flush=True):
        """Describe an instance by printing to stdout."""
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

        .. versionchanged:: 2.1 ``lognames`` and ``lognames_strict`` added
        .. versionchanged:: 2.1 ``strict`` now defaults to False
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

    def to_file(
        self,
        wfile: Union[str, Path, io.BytesIO],
        fformat: Optional[str] = "rms_ascii",
    ):
        """Export well to file or memory stream.

        Args:
            wfile: File name or stream.
            fformat: File format ('rms_ascii'/'rmswell', 'hdf/hdf5/h5').

        Example::

            xwell = Well("somefile.rmswell")
            xwell.dataframe['PHIT'] += 0.1
            xwell.to_file("somefile_copy.rmswell")

        """
        wfile = xtgeo._XTGeoFile(wfile, mode="wb", obj=self)

        wfile.check_folder(raiseerror=OSError)

        self._ensure_consistency()

        if fformat in (None, "rms_ascii", "rms_asc", "rmsasc", "rmswell"):
            _well_io.export_rms_ascii(self, wfile.name)

        elif fformat in ("hdf", "hdf5", "h5"):
            self.to_hdf(wfile)

        return wfile.file

    def from_hdf(
        self,
        wfile: Union[str, Path],
    ):
        """Read well data from HDF.

        Warning:
            This implementation is currently experimental and only recommended
            for testing.

        Read well from as HDF5 formatted file, with xtgeo specific layout.

        Args:
            wfile: Well file or stream

        Returns:
            Well() instance.


        .. versionadded:: 2.14
        """
        wfile = xtgeo._XTGeoFile(wfile, mode="wb", obj=self)
        if wfile.detect_fformat() != "hdf":
            raise ValueError("Wrong file format detected")

        _well_io.import_hdf5_well(self, wfile)

        _self: self.__class__ = self
        return _self  # to make obj = xtgeo.Well().from_hdf(stream) work

    def to_hdf(
        self,
        wfile: Union[str, Path],
        compression: Optional[str] = "lzf",
    ) -> Path:
        """Export well to HDF based file.

        Warning:
            This implementation is currently experimental and only recommended
            for testing.

        Args:
            wfile: HDF File name to write to export to.

        Returns:
            A Path instance to actual file applied.

        .. versionadded:: 2.14
        """
        wfile = xtgeo._XTGeoFile(wfile, mode="wb", obj=self)

        wfile.check_folder(raiseerror=OSError)

        _well_io.export_hdf5_well(self, wfile, compression=compression)

        return wfile.file

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
        trajectory = kwargs.get("trajectory", "Drilled trajectory")
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

    def to_roxar(self, *args, **kwargs):
        """Export (save/store) a well to a roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated.

        The current implementation will either update existing well names
        (then well log array size must not change), or it will make a new well in RMS.

        Note:
           When project is file path (direct access, outside RMS) then
           ``to_roxar()`` will implicitly do a project save. Otherwise, the project
           will not be saved until the user do an explicit project save action.

        Args:
            project (str): Magic string 'project' or file path to project
            wname (str): Name of well, as shown in RMS.
            lognames (:obj:list or :obj:str): List of lognames to save, or
                use simply 'all' for current logs for this well. Default is 'all'
            realisation (int): Currently inactive
            trajectory (str): Name of trajectory in RMS
            logrun (str): Name of logrun in RMS

        .. versionadded:: 2.12
        .. versionchanged:: 2.15
            Saving to new wells enabled (earlier only modifying existing)

        """
        # use *args, **kwargs since this method is overrided in blocked_well, and
        # signature should be the same

        project = args[0]
        wname = args[1]
        lognames = kwargs.get("lognames", "all")
        trajectory = kwargs.get("trajectory", "Drilled trajectory")
        logrun = kwargs.get("logrun", "log")
        realisation = kwargs.get("realisation", 0)

        logger.debug("Not in use: realisation %s", realisation)

        _well_roxapi.export_well_roxapi(
            self,
            project,
            wname,
            lognames=lognames,
            trajectory=trajectory,
            logrun=logrun,
            realisation=realisation,
        )

    def get_wlogs(self) -> OrderedDict:
        """Get a compound dictionary with well log metadata.

        The result will be an Ordered dict on the form:

        ``{"X_UTME": ["CONT", None], ... "Facies": ["DISC", {1: "BG", 2: "SAND"}]}``
        """
        res = OrderedDict()

        for key in self._wlognames:
            wtype = "CONT"
            wrecord = None
            if key in self._wlogtypes:
                wtype = self._wlogtypes[key]
            if key in self._wlogrecords:
                wrecord = self._wlogrecords[key]

            res[key] = [wtype, wrecord]

        return res

    def set_wlogs(self, wlogs: OrderedDict):
        """Set a compound dictionary with well log metadata.

        This operation is somewhat risky as it may lead to inconsistency, so use with
        care! Typically, one will use :meth:`get_wlogs` first and then modify some
        attributes.

        Args:
            wlogs: Input data dictionary

        Raises:
            ValueError: Invalid log type found in input:
            ValueError: Invalid log record found in input:
            ValueError: Invalid input key found:
            ValueError: Invalid log record found in input:

        """
        for key in self._wlognames:
            if key in wlogs.keys():
                typ, rec = wlogs[key]

                if typ in Well.VALID_LOGTYPES:
                    self._wlogtypes[key] = deepcopy(typ)
                else:
                    raise ValueError(f"Invalid log type found in input: {typ}")

                if rec is None or isinstance(rec, dict):
                    self._wlogrecords[key] = deepcopy(rec)
                else:
                    raise ValueError(f"Invalid log record found in input: {rec}")

            else:
                raise ValueError(f"Key for column not found in input: {key}")

        for key in wlogs.keys():
            if key not in self._wlognames:
                raise ValueError(f"Invalid input key found: {key}")

        self._ensure_consistency()

    def isdiscrete(self, logname):
        """Return True of log is discrete, otherwise False.

        Args:
            logname (str): Name of log to check if discrete or not

        .. versionadded:: 2.2.0
        """
        if logname in self._wlognames and self.get_logtype(logname) == "DISC":
            return True
        return False

    def copy(self):
        """Copy a Well instance to a new unique Well instance."""
        # pylint: disable=protected-access

        new = Well()
        new._wlogtypes = deepcopy(self._wlogtypes)
        new._wlogrecords = deepcopy(self._wlogrecords)
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
        """Rename a log, e.g. Poro to PORO."""
        self._ensure_consistency()

        if lname not in self.lognames:
            raise ValueError("Input log does not exist")

        if newname in self.lognames:
            raise ValueError("New log name exists already")

        self._wlogtypes[newname] = self._wlogtypes.pop(lname)
        self._wlogrecords[newname] = self._wlogrecords.pop(lname)

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

        self._wlogtypes[lname] = logtype
        self._wlogrecords[lname] = logrecord

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
        """Returns the type of a give log (e.g. DISC or CONT)."""
        self._ensure_consistency()

        if lname in self._wlogtypes:
            return self._wlogtypes[lname]
        return None

    def set_logtype(self, lname, ltype):
        """Sets the type of a give log (e.g. DISC or CONT)."""
        self._ensure_consistency()

        valid = {"DISC", "CONT"}

        if ltype in valid:
            self._wlogtypes[lname] = ltype
        else:
            raise ValueError("Try to set invalid log type: {}".format(ltype))

    def get_logrecord(self, lname):
        """Returns the record (dict) of a given log name, None if not exists."""
        if lname in self._wlogtypes:
            return self._wlogrecords[lname]

        return None

    def set_logrecord(self, lname, newdict):
        """Sets the record (dict) of a given discrete log."""
        self._ensure_consistency()
        if lname not in self.lognames:
            raise ValueError("No such logname: {}".format(lname))

        if self._wlogtypes[lname] == "CONT":
            raise ValueError("Cannot set a log record for a continuous log")

        if not isinstance(newdict, dict):
            raise ValueError("Input is not a dictionary")

        self._wlogrecords[lname] = newdict

    def get_logrecord_codename(self, lname, key):
        """Returns the name entry of a log record, for a given key.

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

        ier, _, _, hlenv, _ = _cxtgeo.pol_geometrics(xv, yv, zv, nlen, nlen, nlen, nlen)

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
            nlen, ptr_xv, ptr_yv, ptr_zv, ptr_md, ptr_incl, ptr_az, 0
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
        """Truncate the part of the well trajectory that is ~parallel with other.

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
        """Consider if well overlap in X Y coordinates with other well, True/False."""
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
        """Rescale (refine or coarse) by sampling a delta along the trajectory, in MD.

        Args:
            delta (float): Step length
            tvdrange (tuple of floats): Resampling can be limited to TVD interval

        .. versionchanged:: 2.2 Added tvdrange
        """
        _well_oper.rescale(self, delta=delta, tvdrange=tvdrange)

    def get_polygons(self, skipname=False):
        """Return a Polygons object from the well trajectory.

        Args:
            skipname (bool): If True then name column is omitted

        .. versionadded:: 2.1
        .. versionchanged:: 2.13 Added `skipname` key
        """
        dfr = self._df.copy()

        keep = ("X_UTME", "Y_UTMN", "Z_TVDSS")
        for col in dfr.columns:
            if col not in keep:
                dfr.drop(labels=col, axis=1, inplace=True)
        dfr["POLY_ID"] = 1

        if not skipname:
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

        .. versionchanged:: 2.1 improved algorithm
        """
        poly = self.get_polygons()

        if tvdmin is not None:
            poly.dataframe = poly.dataframe[poly.dataframe[poly.zname] >= tvdmin]
            poly.dataframe.reset_index(drop=True, inplace=True)

        return poly.get_fence(distance=sampling, nextend=nextend, asnumpy=asnumpy)

    def create_surf_distance_log(
        self,
        surf: object,
        name: Optional[str] = "DIST_SURF",
    ):
        """Make a log that is vertical distance to a regular surface.

        If the trajectory is above the surface (i.e. more shallow), then the
        distance sign is positive.

        Args:
            surf: The RegularSurface instance.
            name: The name of the new log. If it exists it will be overwritten.

        Example::

            mywell.rescale()  # optional
            thesurf = xtgeo.RegularSurface("some.gri")
            mywell.create_surf_distance_log(thesurf, name="sdiff")

        """
        _well_oper.create_surf_distance_log(self, surf, name)

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
        """Extract the X Y Z ID line (polyline) segment for a given zonevalue.

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

    def mask_shoulderbeds(
        self,
        inputlogs: List[str],
        targetlogs: List[str],
        nsamples: Optional[Union[int, Dict[str, float]]] = 2,
        strict: Optional[bool] = False,
    ) -> bool:
        """Mask data around zone boundaries or other discrete log boundaries.

        This operates on number of samples, hence the actual distance which is masked
        depends on the sampling interval (ie. count) or on distance measures.
        Distance measures are TVD (true vertical depth) or MD (measured depth).

        .. image:: ../../docs/images/wells-mask-shoulderbeds.png
           :width: 300
           :align: center

        Args:
            inputlogs: List of input logs, must be of discrete type.
            targetlogs: List of logs where mask is applied.
            nsamples: Number of samples around boundaries to filter, per side, i.e.
                value 2 means 2 above and 2 below, in total 4 samples.
                As alternative specify nsamples indirectly with a relative distance,
                as a dictionary with one record, as {"tvd": 0.5} or {"md": 0.7}.
            strict: If True, will raise Exception of any of the input or target log
                names are missing.

        Returns:
            True if any operation has been done. False in case nothing has been done,
                 e.g. no targetlogs for this particular well and ``strict`` is False.

        Raises:
            ValueError: Various messages when wrong or inconsistent input.

        Example:
            >>> mywell1.mask_shoulderbeds(["ZONELOG", "FACIES"], ["PHIT", "KLOGH"])
            >>> mywell2.mask_shoulderbeds(["ZONELOG"], ["PHIT"], nsamples={"tvd": 0.8})

        """
        return _well_oper.mask_shoulderbeds(
            self, inputlogs, targetlogs, nsamples, strict
        )

    def get_surface_picks(self, surf):
        """Return :class:`.Points` obj where well crosses the surface (horizon picks).

        There may be several points in the Points() dataframe attribute.
        Also a ``DIRECTION`` column will show 1 if surface is penetrated from
        above, and -1 if penetrated from below.

        Args:
            surf (RegularSurface): The surface instance

        Returns:
            A :class:`.Points` instance, or None if no crossing points

        .. versionadded:: 2.8

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

        .. versionchanged:: 2.9 Added keys for and `activeonly`
        """
        _well_oper.make_ijk_from_grid(
            self, grid, grid_id=grid_id, algorithm=algorithm, activeonly=activeonly
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

        .. versionadded:: 2.1

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
