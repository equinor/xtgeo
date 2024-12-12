"""XTGeo well module, working with one single well."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo import _cxtgeo
from xtgeo.common._xyz_enum import _AttrType
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_LIMIT
from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGDescription
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.metadata.metadata import MetaDataWell
from xtgeo.xyz import _xyz_data
from xtgeo.xyz.polygons import Polygons

from . import _well_aux, _well_io, _well_oper, _well_roxapi, _wellmarkers

if TYPE_CHECKING:
    import io
    from pathlib import Path

logger = null_logger(__name__)

# ======================================================================================
# Functions, as wrappers to class methods


def well_from_file(
    wfile: str | Path,
    fformat: str | None = "rms_ascii",
    mdlogname: str | None = None,
    zonelogname: str | None = None,
    lognames: str | list[str] | None = "all",
    lognames_strict: bool | None = False,
    strict: bool | None = False,
) -> Well:
    """Make an instance of a Well directly from file import.

    Args:
        wfile: File path for well, either a string or a pathlib.Path instance
        fformat: "rms_ascii" or "hdf5"
        mdlogname: Name of Measured Depth log, if any
        zonelogname: Name of Zonelog, if any
        lognames: Name or list of lognames to import, default is "all"
        lognames_strict: If True, all lognames must be present.
        strict: If True, then import will fail if zonelogname or mdlogname are asked
            for but those names are not present in wells.

    Example::

        >>> import xtgeo
        >>> import pathlib
        >>> welldir = pathlib.Path("../foo")
        >>> mywell = xtgeo.well_from_file(welldir / "OP_1.w")

    .. versionchanged:: 2.1 Added ``lognames`` and ``lognames_strict``
    .. versionchanged:: 2.1 ``strict`` now defaults to False
    """
    return Well._read_file(
        wfile,
        fformat=fformat,
        mdlogname=mdlogname,
        zonelogname=zonelogname,
        strict=strict,
        lognames=lognames,
        lognames_strict=lognames_strict,
    )


def well_from_roxar(
    project: str | object,
    name: str,
    trajectory: str | None = "Drilled trajectory",
    logrun: str | None = "log",
    lognames: str | list[str] | None = "all",
    lognames_strict: bool | None = False,
    inclmd: bool | None = False,
    inclsurvey: bool | None = False,
) -> Well:
    """This makes an instance of a Well directly from Roxar RMS.

    Note this method works only when inside RMS, or when RMS license is
    activated (through the roxar environment).

    Args:
        project: Path to project or magic the ``project`` variable in RMS.
        name: Name of Well, as shown in RMS.
        trajectory: Name of trajectory in RMS.
        logrun: Name of logrun in RMS.
        lognames: List of lognames to import, or use 'all' for all present logs
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
    # TODO - mdlogname and zonelogname
    return Well._read_roxar(
        project,
        name,
        trajectory=trajectory,
        logrun=logrun,
        lognames=lognames,
        lognames_strict=lognames_strict,
        inclmd=inclmd,
        inclsurvey=inclsurvey,
    )


class Well:
    """Class for a single well in the XTGeo framework.

    The well logs are stored in a Pandas dataframe, which make manipulation
    easy and fast.

    The well trajectory are here represented as first 3 columns in the dataframe,
    and XYZ have pre-defined names: ``X_UTME``, ``Y_UTMN``, ``Z_TVDSS``.

    Other geometry logs may has also 'semi-defined' names, but this is not a strict
    rule:

    ``M_MDEPTH`` or ``Q_MDEPTH``: Measured depth, either real/true (M_xx) or
    quasi computed/estimated (Q_xx). The Quasi may be incorrect for
    all uses, but sufficient for some computations.

    Similar for ``M_INCL``, ``Q_INCL``, ``M_AZI``, ``Q_ASI``.

    All Pandas values (yes, discrete also!) are currently stored as float64
    format, and undefined values are Nan. Integers are stored as Float due
    to the (historic) lacking support for 'Integer Nan'.

    Note there is a method that can return a dataframe (copy) with Integer
    and Float columns, see :meth:`get_filled_dataframe`.

    The instance can be made either from file or by specification::

        >>> well1 = xtgeo.well_from_file(well_dir + '/OP_1.w')
        >>> well2 = xtgeo.Well(rkb=32.0, xpos=1234.0, ypos=4567.0, wname="Foo",
                    df: mydataframe, ...)

    Args:
        rkb: Well RKB height
        xpos: Well head X pos
        ypos: Well head Y pos
        wname: well name
        df: A pandas dataframe with log values, expects columns to include
          'X_UTME', 'Y_UTMN', 'Z_TVDSS' for x, y and z coordinates.
          Other columns should be log values.
        filesrc: source file if any
        mdlogname: Name of Measured Depth log, if any.
        zonelogname: Name of Zonelog, if any
        wlogtypes: dictionary of log types, 'DISC' (discrete) or 'CONT' (continuous),
            defaults to to 'CONT'.
        wlogrecords: dictionary of codes for 'DISC' logs, None for no codes given,
            defaults to None.
    """

    def __init__(
        self,
        rkb: float = 0.0,
        xpos: float = 0.0,
        ypos: float = 0.0,
        wname: str = "",
        df: pd.DataFrame | None = None,
        mdlogname: str | None = None,
        zonelogname: str | None = None,
        wlogtypes: dict[str, str] | None = None,
        wlogrecords: dict[str, str] | None = None,
        filesrc: str | Path | None = None,
    ):
        # state variables from args
        self._rkb = rkb
        self._xpos = xpos
        self._ypos = ypos
        self._wname = wname
        self._filesrc = filesrc
        self._mdlogname = mdlogname
        self._zonelogname = zonelogname

        self._wdata = _xyz_data._XYZData(df, wlogtypes, wlogrecords)

        self._ensure_consistency()

        # additional state variables
        self._metadata = MetaDataWell()
        self._metadata.required = self

    def __repr__(self):  # noqa: D105
        # should (in theory...) be able to newobject = eval(repr(thisobject))
        return (
            f"{self.__class__.__name__} (rkb={self._rkb}, xpos={self._xpos}, "
            f"ypos={self._ypos}, wname='{self._wname}', "
            f"filesrc='{self._filesrc}', mdlogname='{self._mdlogname}', "
            f"zonelogname='{self._zonelogname}', \n"
            f"wlogtypes='{self._wdata.attr_types}', "
            f"\nwlogrecords='{self._wdata.attr_records}', "
            f"df=\n{repr(self._wdata.data)}))"
        )

    def __str__(self):  # noqa: D105
        # user friendly print
        return self.describe(flush=False)

    def _ensure_consistency(self):
        """Ensure consistency"""
        self._wdata.ensure_consistency()

        if self._mdlogname not in self._wdata.data:
            self._mdlogname = None

        if self._zonelogname not in self._wdata.data:
            self._zonelogname = None

    def ensure_consistency(self):
        """Ensure consistency for the instance.

        .. versionadded:: 3.5
        """
        # public version, added oct-23
        self._ensure_consistency()

    # ==================================================================================
    # Properties
    # ==================================================================================

    @property
    def xname(self):
        """Return or set name of X coordinate column."""
        return self._wdata.xname

    @xname.setter
    def xname(self, new_xname: str):
        self._wdata.xname = new_xname

    @property
    def yname(self):
        """Return or set name of Y coordinate column."""
        return self._wdata.yname

    @yname.setter
    def yname(self, new_yname: str):
        self._wdata.yname = new_yname

    @property
    def zname(self):
        """Return or set name of Z coordinate column."""
        return self._wdata.zname

    @zname.setter
    def zname(self, new_zname: str):
        self._wdata.zname = new_zname

    @property
    def metadata(self):
        """Return metadata object instance of type MetaDataRegularSurface."""
        return self._metadata

    @metadata.setter
    def metadata(self, obj):
        # The current metadata object can be replaced. This is a bit dangerous so
        # further check must be done to validate. TODO.
        if not isinstance(obj, MetaDataWell):
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
        return xname.replace(" ", "_")

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
        if mname in self.get_lognames():
            self._mdlogname = mname
        else:
            self._mdlogname = None

    @property
    def zonelogname(self):
        """str: Returns or sets name of zone log, return None if missing."""
        return self._zonelogname

    @zonelogname.setter
    def zonelogname(self, zname):
        if zname in self.get_lognames():
            self._zonelogname = zname
        else:
            self._zonelogname = None

    @property
    def dataframe(self):
        """Returns or set the Pandas dataframe object for all logs."""
        warnings.warn(
            "Direct access to the dataframe property in Well class will be deprecated "
            "in xtgeo 5.0. Use `get_dataframe()` instead.",
            PendingDeprecationWarning,
        )
        return self._wdata.get_dataframe(copy=False)  # get a view, for backward compat.

    @dataframe.setter
    def dataframe(self, dfr):
        warnings.warn(
            "Direct access to the dataframe property in Well class will be deprecated "
            "in xtgeo 5.0. Use `set_dataframe()` instead.",
            PendingDeprecationWarning,
        )
        self.set_dataframe(dfr)  # this will include consistency checking!

    @property
    def nrow(self):
        """int: Returns the Pandas dataframe object number of rows."""
        return len(self._wdata.data.index)

    @property
    def ncol(self):
        """int: Returns the Pandas dataframe object number of columns."""
        return len(self._wdata.data.columns)

    @property
    def nlogs(self):
        """int: Returns the Pandas dataframe object number of columns."""
        return len(self._wdata.data.columns) - 3

    @property
    def lognames_all(self):
        """list: Returns dataframe column names as list, including mandatory coords."""
        return self.get_lognames()

    @property
    def lognames(self):
        """list: Returns the Pandas dataframe column as list excluding coords."""
        return list(self._wdata.data)[3:]

    @property
    def wlogtypes(self):
        """Returns wlogtypes"""
        return {name: atype.name for name, atype in self._wdata.attr_types.items()}

    @property
    def wlogrecords(self):
        """Returns wlogrecords"""
        return deepcopy(self._wdata.attr_records)

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
        return xname.replace(" ", "")

    def describe(self, flush=True):
        """Describe an instance by printing to stdout."""
        dsc = XTGDescription()

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
                        string += f"{code}: {val} "
                    elif idx == nlen - 1:
                        string += f"...  {code}: {val})"
            else:
                string = f"{rec}"
            dsc.txt("Logname", wlog, self.get_logtype(wlog), string)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    @classmethod
    def _read_file(
        cls,
        wfile: str | Path,
        fformat: str | None = "rms_ascii",
        **kwargs,
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

            >>> mywell = Well().from_file(well_dir + '/OP_1.w')

        .. versionchanged:: 2.1 ``lognames`` and ``lognames_strict`` added
        .. versionchanged:: 2.1 ``strict`` now defaults to False
        """

        wfile = FileWrapper(wfile)
        fmt = wfile.fileformat(fformat)

        kwargs = _well_aux._data_reader_factory(fmt)(wfile, **kwargs)
        return cls(**kwargs)

    def to_file(
        self,
        wfile: str | Path | io.BytesIO,
        fformat: str | None = "rms_ascii",
    ):
        """Export well to file or memory stream.

        Args:
            wfile: File name or stream.
            fformat: File format ('rms_ascii'/'rmswell', 'hdf/hdf5/h5').

        Example::

            >>> xwell = Well(well_dir + '/OP_1.w')
            >>> dfr = xwell.get_dataframe()
            >>> dfr['Poro'] += 0.1
            >>> xwell.set_dataframe(dfr)
            >>> filename = xwell.to_file(outdir + "/somefile_copy.rmswell")

        """
        wfile = FileWrapper(wfile, mode="wb", obj=self)

        wfile.check_folder(raiseerror=OSError)

        self._ensure_consistency()

        if not fformat or fformat in (
            None,
            "rms_ascii",
            "rms_asc",
            "rmsasc",
            "rmswell",
        ):
            _well_io.export_rms_ascii(self, wfile.name)

        elif fformat in FileFormat.HD5.value:
            self.to_hdf(wfile)

        else:
            extensions = FileFormat.extensions_string([FileFormat.HDF])
            raise InvalidFileFormatError(
                f"File format {fformat} is invalid for a well type. "
                f"Supported formats are {extensions}, 'rms_ascii', 'rms_asc', "
                "'rmsasc', 'rmswell'."
            )

        return wfile.file

    def to_hdf(
        self,
        wfile: str | Path,
        compression: str | None = "lzf",
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
        wfile = FileWrapper(wfile, mode="wb", obj=self)

        wfile.check_folder(raiseerror=OSError)

        _well_io.export_hdf5_well(self, wfile, compression=compression)

        return wfile.file

    @classmethod
    def _read_roxar(
        cls,
        project: str | object,
        name: str,
        trajectory: str | None = "Drilled trajectory",
        logrun: str | None = "log",
        lognames: str | list[str] | None = "all",
        lognames_strict: bool | None = False,
        inclmd: bool | None = False,
        inclsurvey: bool | None = False,
    ):
        kwargs = _well_roxapi.import_well_roxapi(
            project,
            name,
            trajectory=trajectory,
            logrun=logrun,
            lognames=lognames,
            lognames_strict=lognames_strict,
            inclmd=inclmd,
            inclsurvey=inclsurvey,
        )
        return cls(**kwargs)

    def to_roxar(self, *args, **kwargs):
        """Export (save/store) a well to a roxar project.

        Note this method works only when inside RMS, or when RMS license is
        activated in terminal.

        The current implementation will either update the existing well
        (then well log array size must not change), or it will make a new well in RMS.


        Args:
            project (str, object): Magic string 'project' or file path to project
            wname (str): Name of well, as shown in RMS.
            lognames (:obj:list or :obj:str): List of lognames to save, or
                use simply 'all' for current logs for this well. Default is 'all'
            realisation (int): Currently inactive
            trajectory (str): Name of trajectory in RMS, default is "Drilled trajectory"
            logrun (str): Name of logrun in RMS, defaault is "log"
            update_option (str): None | "overwrite" | "append". This only applies
                when the well (wname) exists in RMS, and rules are based on name
                matching. Default is None which means that all well logs in
                RMS are emptied and then replaced with the content from xtgeo.
                The "overwrite" option will replace logs in RMS with logs from xtgeo,
                and append new if they do not exist in RMS. The
                "append" option will only append logs if name does not exist in RMS
                already. Reading only a subset of logs and then use "overwrite" or
                "append" may speed up execution significantly.

        Note:
           When project is file path (direct access, outside RMS) then
           ``to_roxar()`` will implicitly do a project save. Otherwise, the project
           will not be saved until the user do an explicit project save action.

        Example::

            # assume that existing logs in RMS are ["PORO", "PERMH", "GR", "DT", "FAC"]
            # read only one existing log (faster)

            wll = xtgeo.well_from_roxar(project, "WELL1", lognames=["PORO"])
            dfr = wll.get_dataframe()
            dfr["PORO"] += 0.2  # add 0.2 to PORO log
            wll.set_dataframe(dfr)
            wll.create_log("NEW", value=0.333)  # create a new log with constant value

            # the "option" is a variable... for output, ``lognames="all"`` is default
            if option is None:
                # remove all current logs in RMS; only logs will be PORO and NEW
                wll.to_roxar(project, "WELL1", update_option=option)
            elif option == "overwrite":
                # keep all original logs but update PORO and add NEW
                wll.to_roxar(project, "WELL1", update_option=option)
            elif option == "append":
                # keep all original logs as they were (incl. PORO) and add NEW
                wll.to_roxar(project, "WELL1", update_option=option)

        Note:
            The keywords ``lognames`` and ``update_option`` will interact

        .. versionadded:: 2.12
        .. versionchanged:: 2.15
            Saving to new wells enabled (earlier only modifying existing)
        .. versionchanged:: 3.5
            Add key ``update_option``
        """
        # use *args, **kwargs since this method is overrided in blocked_well, and
        # signature should be the same (TODO: change this to keywords; think this is
        # a python 2.7 relict?)

        project = args[0]
        wname = args[1]
        lognames = kwargs.get("lognames", "all")
        trajectory = kwargs.get("trajectory", "Drilled trajectory")
        logrun = kwargs.get("logrun", "log")
        realisation = kwargs.get("realisation", 0)
        update_option = kwargs.get("update_option")

        logger.debug("Not in use: realisation %s", realisation)

        _well_roxapi.export_well_roxapi(
            self,
            project,
            wname,
            lognames=lognames,
            trajectory=trajectory,
            logrun=logrun,
            realisation=realisation,
            update_option=update_option,
        )

    def get_lognames(self):
        """Get the lognames for all logs."""
        return list(self._wdata.data)

    def get_wlogs(self) -> dict:
        """Get a compound dictionary with well log metadata.

        The result will be an dict on the form:

        ``{"X_UTME": ["CONT", None], ... "Facies": ["DISC", {1: "BG", 2: "SAND"}]}``
        """
        res = {}

        for key in self.get_lognames():
            wtype = _AttrType.CONT.value
            wrecord = None
            if key in self._wdata.attr_types:
                wtype = self._wdata.attr_types[key].name
            if key in self._wdata.attr_records:
                wrecord = self._wdata.attr_records[key]

            res[key] = [wtype, wrecord]

        return res

    def set_wlogs(self, wlogs: dict):
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
        for key in self.get_lognames():
            if key in wlogs:
                typ, rec = wlogs[key]
                self._wdata.set_attr_type(key, typ)
                self._wdata.set_attr_record(key, deepcopy(rec))

        self._ensure_consistency()

    def isdiscrete(self, logname):
        """Return True of log is discrete, otherwise False.

        Args:
            logname (str): Name of log to check if discrete or not

        .. versionadded:: 2.2.0
        """
        return (
            logname in self.get_lognames()
            and self.get_logtype(logname) == _AttrType.DISC.value
        )

    def copy(self):
        """Copy a Well instance to a new unique Well instance."""
        return Well(
            self.rkb,
            self.xpos,
            self.ypos,
            self.wname,
            self._wdata.data.copy(),
            self.mdlogname,
            self.zonelogname,
            self.wlogtypes,
            self.wlogrecords,
            self._filesrc,
        )

    def rename_log(self, lname, newname):
        """Rename a log, e.g. Poro to PORO."""
        self._wdata.rename_attr(lname, newname)

        if self._mdlogname == lname:
            self._mdlogname = newname

        if self._zonelogname == lname:
            self._zonelogname = newname

    def create_log(
        self,
        lname: str,
        logtype: str = _AttrType.CONT.value,
        logrecord: dict | None = None,
        value: float = 0.0,
        force: bool = True,
    ) -> bool:
        """Create a new log with initial values.

        If the logname already exists, it will be silently overwritten, unless
        the option force=False.

        Args:
            lname: name of new log
            logtype: Must be 'CONT' (default) or 'DISC' (discrete)
            logrecord: A dictionary of key: values for 'DISC' logs
            value: initial value to set
            force: If True, and lname exists, it will be overwritten, if
               False, no new log will be made. Will return False.

        Returns:
            True ff a new log is made (either new or force overwrite an
            existing) or False if the new log already exists,
            and ``force=False``.

        Note::

            A new log can also be created by adding it to the dataframe directly, but
            with less control over e.g. logrecord

        """
        return self._wdata.create_attr(lname, logtype, logrecord, value, force)

    def copy_log(
        self,
        lname: str,
        newname: str,
        force: bool = True,
    ) -> bool:
        """Copy a log from an existing to a name

        If the new log already exists, it will be silently overwritten, unless
        the option force=False.

        Args:
            lname: name of existing log
            newname: name of new log

        Returns:
            True if a new log is made (either new or force overwrite an
            existing) or False if the new log already exists,
            and ``force=False``.

        Note::

            A copy can also be done directly in the dataframe, but with less
            consistency checks; hence this method is recommended

        """
        return self._wdata.copy_attr(lname, newname, force)

    def delete_log(self, lname: str | list[str]) -> int:
        """Delete/remove an existing log, or list of logs.

        Will continue silently if a log does not exist.

        Args:
            lname: A logname or a list of lognames

        Returns:
            Number of logs deleted

        Note::

            A log can also be deleted by simply removing it from the dataframe.

        """
        logger.debug("Deleting log(s) %s...", lname)
        return self._wdata.delete_attr(lname)

    delete_logs = delete_log  # alias function

    def get_logtype(self, lname) -> str | None:
        """Returns the type of a given log (e.g. DISC or CONT), None if not present."""
        if lname in self._wdata.attr_types:
            return self._wdata.attr_types[lname].name
        return None

    def set_logtype(self, lname, ltype):
        """Sets the type of a give log (e.g. DISC or CONT)."""
        self._wdata.set_attr_type(lname, ltype)

    def get_logrecord(self, lname):
        """Returns the record (dict) of a given log name, None if not exists."""

        return self._wdata.get_attr_record(lname)

    def set_logrecord(self, lname, newdict):
        """Sets the record (dict) of a given discrete log."""
        self._wdata.set_attr_record(lname, newdict)

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

    def get_dataframe(self, copy: bool = True):
        """Get a copy (default) or a view of the dataframe.

        Args:
            copy: If True, return a deep copy. A view (copy=False) will be faster and
                more memory efficient, but less "safe" for some cases when manipulating
                dataframes.

        .. versionchanged:: 3.7 Added `copy` keyword
        """
        return self._wdata.get_dataframe(copy=copy)

    def get_filled_dataframe(self, fill_value=UNDEF, fill_value_int=UNDEF_INT):
        """Fill the Nan's in the dataframe with real UNDEF values.

        This module returns a copy of the dataframe in the object; it
        does not change the instance.

        Note that DISC logs will be casted to columns with integer
        as datatype.

        Returns:
            A pandas dataframe where Nan er replaces with preset
                high XTGeo UNDEF values, or user defined values.

        """
        return self._wdata.get_dataframe_copy(
            infer_dtype=True,
            filled=True,
            fill_value=fill_value,
            fill_value_int=fill_value_int,
        )

    def set_dataframe(self, dfr):
        """Set the dataframe."""
        self._wdata.set_dataframe(dfr)

    def create_relative_hlen(self):
        """Make a relative length of a well, as a log.

        The first well og entry defines zero, then the horizontal length
        is computed relative to that by simple geometric methods.
        """
        self._wdata.create_relative_hlen()

    def geometrics(self):
        """Compute some well geometrical arrays MD, INCL, AZI, as logs.

        These are kind of quasi measurements hence the logs will named
        with a Q in front as Q_MDEPTH, Q_INCL, and Q_AZI.

        These logs will be added to the dataframe. If the mdlogname
        attribute does not exist in advance, it will be set to 'Q_MDEPTH'.

        Returns:
            False if geometrics cannot be computed

        """
        rvalue = self._wdata.geometrics()

        if not self._mdlogname:
            self._mdlogname = "Q_MDEPTH"

        return rvalue

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

        this_df = self.get_dataframe()
        other_df = other.get_dataframe()

        if this_df.shape[0] < 3 or other_df.shape[0] < 3:
            raise ValueError(
                f"Too few points to truncate parallel path, was "
                f"{this_df.size} and {other_df.size}, must be >3"
            )

        # extract numpies from XYZ trajectory logs
        xv1 = self._wdata.data[self.xname].values
        yv1 = self._wdata.data[self.yname].values
        zv1 = self._wdata.data[self.zname].values

        xv2 = other_df[self.xname].values
        yv2 = other_df[self.yname].values
        zv2 = other_df[self.zname].values

        ier = _cxtgeo.well_trunc_parallel(
            xv1, yv1, zv1, xv2, yv2, zv2, xtol, ytol, ztol, itol, atol, 0
        )

        if ier != 0:
            raise RuntimeError("Unexpected error")

        dfr = self.get_dataframe()
        dfr = dfr[dfr[self.xname] < UNDEF_LIMIT]
        self.set_dataframe(dfr)

    def may_overlap(self, other):
        """Consider if well overlap in X Y coordinates with other well, True/False."""
        dataframe = self.get_dataframe()
        other_dataframe = other.get_dataframe()

        if dataframe.size < 2 or other_dataframe.size < 2:
            return False

        # extract numpies from XYZ trajectory logs
        xmin1 = np.nanmin(dataframe[self.xname].values)
        xmax1 = np.nanmax(dataframe[self.xname].values)
        ymin1 = np.nanmin(dataframe[self.yname].values)
        ymax1 = np.nanmax(dataframe[self.yname].values)

        xmin2 = np.nanmin(other_dataframe[self.xname].values)
        xmax2 = np.nanmax(other_dataframe[self.xname].values)
        ymin2 = np.nanmin(other_dataframe[self.yname].values)
        ymax2 = np.nanmax(other_dataframe[self.yname].values)

        if xmin1 > xmax2 or ymin1 > ymax2:
            return False
        return not (xmin2 > xmax1 or ymin2 > ymax1)

    def limit_tvd(self, tvdmin, tvdmax):
        """Truncate the part of the well that is outside tvdmin, tvdmax.

        Range will be in tvdmin <= tvd <= tvdmax.

        Args:
            tvdmin (float): Minimum TVD
            tvdmax (float): Maximum TVD
        """
        dfr = self.get_dataframe()
        dfr = dfr[dfr[self.zname] >= tvdmin]
        dfr = dfr[dfr[self.zname] <= tvdmax]
        self.set_dataframe(dfr)

    def downsample(self, interval=4, keeplast=True):
        """Downsample by sampling every N'th element (coarsen only).

        Args:
            interval (int): Sampling interval.
            keeplast (bool): If True, the last element from the original
                dataframe is kept, to avoid that the well is shortened.
        """
        dataframe = self.get_dataframe()

        if dataframe.size < 2 * interval:
            return

        dfr = dataframe[::interval].copy()

        if keeplast:
            dfr = pd.concat([dfr, dataframe.iloc[-1:]], ignore_index=True)

        self.set_dataframe(dfr.reset_index(drop=True))

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
        dfr = self._wdata.data.copy()

        keep = (self.xname, self.yname, self.zname)
        for col in dfr.columns:
            if col not in keep:
                dfr.drop(labels=col, axis=1, inplace=True)
        dfr["POLY_ID"] = 1

        if not skipname:
            dfr["NAME"] = self.xwellname
        poly = Polygons()
        poly.set_dataframe(dfr)
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
            poly_df = poly.get_dataframe()
            poly_df = poly_df[poly_df[poly.zname] >= tvdmin]
            poly_df.reset_index(drop=True, inplace=True)
            poly.set_dataframe(poly_df)

        return poly.get_fence(distance=sampling, nextend=nextend, asnumpy=asnumpy)

    def create_surf_distance_log(
        self,
        surf: object,
        name: str | None = "DIST_SURF",
    ):
        """Make a log that is vertical distance to a regular surface.

        If the trajectory is above the surface (i.e. more shallow), then the
        distance sign is positive.

        Args:
            surf: The RegularSurface instance.
            name: The name of the new log. If it exists it will be overwritten.

        Example::

            mywell.rescale()  # optional
            thesurf = xtgeo.surface_from_file("some.gri")
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
        return _well_oper.report_zonation_holes(self, threshold=threshold)

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

        return _wellmarkers.get_zonation_points(
            self, tops, incl_limit, top_prefix, zonelist, use_undef
        )

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

        if self.zonelogname not in dff.columns:
            return None

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

        useloglist = [self.xname, self.yname, self.zname, "POLY_ID"]
        if extralogs is not None:
            useloglist.extend(extralogs)

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
        return _wellmarkers.get_fraction_per_zone(
            self,
            dlogname,
            dcodes,
            zonelist=zonelist,
            incl_limit=incl_limit,
            count_limit=count_limit,
            zonelogname=zonelogname,
        )

    def mask_shoulderbeds(
        self,
        inputlogs: list[str],
        targetlogs: list[str],
        nsamples: int | dict[str, float] | None = 2,
        strict: bool | None = False,
    ) -> bool:
        """Mask data around zone boundaries or other discrete log boundaries.

        This operates on number of samples, hence the actual distance which is masked
        depends on the sampling interval (ie. count) or on distance measures.
        Distance measures are TVD (true vertical depth) or MD (measured depth).

        .. image:: images/wells-mask-shoulderbeds.png
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
            >>> mywell1 = Well(well_dir + '/OP_1.w')
            >>> mywell2 = Well(well_dir + '/OP_2.w')
            >>> did_succeed = mywell1.mask_shoulderbeds(["Zonelog", "Facies"], ["Perm"])
            >>> did_succeed = mywell2.mask_shoulderbeds(
            ...     ["Zonelog"],
            ...     ["Perm"],
            ...     nsamples={"tvd": 0.8}
            ... )

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
