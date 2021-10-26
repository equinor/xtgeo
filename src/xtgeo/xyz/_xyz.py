# -*- coding: utf-8 -*-
"""XTGeo xyz module (base class)"""
import functools
import io
import pathlib
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Optional, Union

import deprecation
import numpy as np
import pandas as pd
import xtgeo
import xtgeo.common.sys as xtgeosys
from xtgeo.common import XTGDescription, XTGeoDialog
from xtgeo.xyz import _xyz_io, _xyz_roxapi

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def _data_reader_factory(file_format):
    if file_format == "xyz":
        return _xyz_io.import_xyz
    if file_format == "zmap_ascii":
        return _xyz_io.import_zmap
    if file_format == "rms_attr":
        return _xyz_io.import_rms_attr
    raise ValueError(f"Unknown file format {file_format}")


def _allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of Points/Polygons and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    # Introduced post xtgeo version 2.15
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        # Checking if we are doing an initialization from file or surface and raise a
        # deprecation warning if we are.
        if len(args) == 1:

            if isinstance(args[0], (str, pathlib.Path)):
                warnings.warn(
                    "Initializing directly from file name is deprecated and will be "
                    "removed in xtgeo version 4.0. Use: "
                    "poi = xtgeo.points_from_file('some_file.xx') instead (Points) or "
                    "pol = xtgeo.polygons_from_file('some_file.xx') instead (Polygons)",
                    DeprecationWarning,
                )
                pfile = args[0]
                fformat = kwargs.get("fformat", "guess")
                ispoly = kwargs.get("is_polygons", False)
                pfile = xtgeosys._XTGeoFile(pfile)
                if fformat is None or fformat == "guess":
                    fformat = pfile.detect_fformat()
                else:
                    fformat = pfile.generic_format_by_proposal(fformat)  # default
                kwargs = _data_reader_factory(fformat)(pfile, is_polygons=ispoly)

            elif isinstance(args[0], xtgeo.RegularSurface):
                warnings.warn(
                    "Initializing directly from RegularSurface is deprecated "
                    "and will be removed in xtgeo version 4.0. Use: "
                    "poi = xtgeo.points_from_surface(regsurf) instead",
                    DeprecationWarning,
                )
                zname = kwargs.get("zname", "Z_TVDSS")
                kwargs = XYZ._read_surface(args[0], zname=zname, return_cls=False)

            elif isinstance(args[0], (list, np.ndarray, pd.DataFrame)):
                # initialisation from an list-like object without 'values' keyword
                # should be possible, i.e. Points(some_list) is same as
                # Points(values=some_list)
                kwargs["values"] = args[0]

            else:
                raise TypeError("Input argument of unknown type: ", type(args[0]))

        return func(cls, **kwargs)

    return wrapper


class XYZ:
    """Base class for Points and Polygons in XTGeo.

    The XYZ base class have common methods and properties for Points and Polygons.
    The underlying datatype is a Pandas dataframe with minimal 3 (Points) or 4
    (Polygons) columns, where the two first represent X and Y coordinates.

    The third column is a number, which may represent the depth, thickness,
    or other property. For Polygons, there is a 4'th column which is an integer
    representing poly-line ID.

    Additional columns are possible but certainly not required. These are free
    attributes with user-defined names. These names (with data-type) are
    stored in an ordered dict: self._attrs as {"somename": "type", ...}.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column, default is X_UTME.
        yname: Name of second (Y) mandatory column, default is Y_UTMN.
        zname: Name of third (Z) mandatory column, default is Z_TVDSS.
        pname: Name of fourth columns (mandatory for Polygons), default is POLY_ID.
        name: A given name for the Points/Polygons object.
        is_polygons: Shall be True for Polygons(), False for Points()
        attributes: An ordered dict for addional columns (attributes) on the
            form {"WellName": "str", "SomeCode": "int"}
        kwargs: Additonal keys, mostly for internal usage

    Note:
        Do not use the XYZ class directly. Use the :class:`Points` or :class:`Polygons`
        classes!
    """

    @_allow_deprecated_init
    def __init__(
        self,
        values: Optional[Union[list, np.ndarray, pd.DataFrame]] = None,
        xname: Optional[str] = "X_UTME",
        yname: Optional[str] = "Y_UTMN",
        zname: Optional[str] = "Z_TVDSS",
        pname: Optional[str] = "POLY_ID",
        name: Optional[str] = "unknown",
        is_polygons: Optional[bool] = False,
        attributes: Optional[dict] = None,
        **kwargs,
    ):
        dataframe = kwargs.get("dataframe", None)
        self._filesrc = kwargs.get("filesrc", None)

        if values is not None and dataframe is not None:
            raise ValueError("Conflicting 'values' and 'dataframe' input!")

        self._xname = xname
        self._yname = yname
        self._zname = zname
        self._pname = pname
        self._ispolygons = is_polygons
        self._name = name
        self._df = dataframe

        # The _nwells is introduced for methods that count the number of wells used
        # actually in input. From 2.16
        self._nwells = None

        # additional input, given through **kwargs. For the import from file routines
        # (class methods), the 'dataframe' key may be populated to avoid a second
        # round of processing. In such cases, 'values' shall be None to avoid
        # conflicts.

        # other attributes as (name: type), where type is
        # ~ ('str', 'int', 'float', 'bool')
        if attributes is None:
            self._attrs = OrderedDict()
        else:
            self._attrs = OrderedDict(attributes)

        if values is not None:

            if isinstance(values, (list, np.ndarray, pd.DataFrame)):
                # make instance from a list(-like) of 3 or 4 tuples or similar
                logger.info("Instance from list")
                self._from_list_like(values)

        logger.info("XYZ Instance initiated (base class) ID %s", id(self))

    @property
    def xname(self) -> str:
        """Returns or set the name of the X column."""
        return self._xname

    @xname.setter
    def xname(self, newname):
        self._df_column_rename(newname, self._xname)
        self._xname = newname

    @property
    def yname(self) -> str:
        """Returns or set the name of the Y column."""
        return self._yname

    @yname.setter
    def yname(self, newname):
        self._df_column_rename(newname, self._yname)
        self._yname = newname

    @property
    def zname(self) -> str:
        """Returns or set the name of the Z column."""
        return self._zname

    @zname.setter
    def zname(self, newname):
        self._df_column_rename(newname, self._zname)
        self._zname = newname

    @property
    def pname(self) -> str:
        """Returns or set the name of the POLY_ID column."""
        return self._pname

    @pname.setter
    def pname(self, value):
        try:
            self._check_name(value)
        except ValueError as verr:
            if "does not exist" in str(verr):
                return
        self._df_column_rename(value, self._pname)
        self._pname = value

    @property
    def attributes(self) -> dict:
        """Returns a dictionary with attribute names and type, or None."""
        return self._attrs

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    @property
    def filesrc(self) -> str:
        """Returns the filesrc attribute, file name or description (read-only).

        Unless it is a valid file name, a `Derived from:` prefix is applied, e.g.
        `Derived from: list-like`
        """

    @property
    def nrow(self) -> int:
        """Returns the Pandas dataframe object number of rows."""
        if self.dataframe is None:
            return 0
        return len(self.dataframe.index)

    @property
    def nwells(self) -> int:
        """Returns the number of wells associated with the XYZ set (read-only).

        If zero or not relevant, None is returned.

        .. versionadded:: 2.16
        """
        return self._nwells

    def _df_column_rename(self, newname, oldname):
        if isinstance(newname, str):
            if oldname and self._df is not None:
                self._df.rename(columns={oldname: newname}, inplace=True)
        else:
            raise ValueError(f"Wrong type of input to {newname}; must be string")

    def _check_name(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Wrong type of input; must be string, was {type(value)}")

        if value not in self._df.columns:
            raise ValueError(
                f"{value} does not exist as a column name, must be "
                f"one of: f{self._df.columns}"
            )

    def copy(self):
        """Returns a a deep copy of an instance"""

        mycopy = self.__class__()
        mycopy._df = self._df.copy()
        mycopy._ispolygons = self._ispolygons
        mycopy._xname = self._xname
        mycopy._yname = self._yname
        mycopy._zname = self._zname
        mycopy._pname = self._pname
        mycopy._filesrc = self._filesrc = None
        mycopy._attrs = deepcopy(self._attrs)

        return mycopy

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = XTGDescription()
        dsc.title("Description of {} instance".format(self.__class__.__name__))
        dsc.txt("Object ID", id(self))
        dsc.txt("xname, yname, zname", self._xname, self._yname, self._zname)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    # ==================================================================================
    # Import and export
    # ==================================================================================
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_file() or xtgeo.polygons_from_file() instead",
    )
    def from_file(self, pfile, fformat="xyz", **kwargs):
        """Import Points or Polygons from a file (deprecated).

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'rms_attr': RMS points formats with attributes (extra columns)

        * 'guess': Try to choose file format based on extension

        Args:
            pfile (str): Name of file or pathlib.Path instance
            fformat (str): File format, see list above

        Returns:
            Object instance (needed optionally)

        Raises:
            OSError: if file is not present or wrong permissions.

        .. deprecated:: 2.16
           Use xtgeo.points_from_file() or xtgeo.polygons_from_file() instead.
        """
        is_polygons = kwargs.get("is_polygons", False)

        pfile = xtgeo._XTGeoFile(pfile)
        pfile.check_file(raiseerror=OSError)

        logger.info("Reading from file %s...", pfile.name)

        if fformat is None or fformat == "guess":
            fformat = pfile.detect_fformat()
        else:
            fformat = pfile.generic_format_by_proposal(fformat)  # default

        newkwargs = _data_reader_factory(fformat)(pfile, is_polygons=is_polygons)
        self._reset(**newkwargs)

    def _reset(self, **kwargs):
        self._df = kwargs.get("dataframe", self._df)
        self._ispolygons = kwargs.get(False, self._ispolygons)
        self._attrs = kwargs.get("attributes", self._attrs)
        self.xname = kwargs.get("xname", self._xname)
        self.yname = kwargs.get("yname", self._yname)
        self.zname = kwargs.get("zname", self._zname)
        self.pname = kwargs.get("pname", self._pname)

    @classmethod
    def _read_file(
        cls,
        pfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = None,
        is_polygons: Optional[bool] = False,
        **kwargs,
    ):
        """Import Points or Polygons from a file.

        Supported import formats (fformat):

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'rms_attr': RMS points formats with attributes (extra columns)

        * 'guess': Try to choose file format based on extension

        Args:
            pfile: File-like or memory stream instance.
            fformat (str): File format, None/guess/xyz/pol/poi...
            is_polygons: True if Polygons
            **kwargs: For special kw args


        Example::

            >>> myxyz = _XYZ._read_file('myfile.x')

        Raises:
            OSError: if file is not present or wrong permissions.

        """
        pfile = xtgeo._XTGeoFile(pfile)
        if fformat is None or fformat == "guess":
            fformat = pfile.detect_fformat()
        else:
            fformat = pfile.generic_format_by_proposal(fformat)  # default
        kwargs = _data_reader_factory(fformat)(pfile, is_polygons=is_polygons)

        kwargs["filesrc"] = pfile.name
        if kwargs.get("return_cls", True) is False:
            return kwargs

        return cls(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_surface() instead",
    )
    def from_surface(self, surf, zname="Z_TVDSS"):
        """Get points as X Y Value from a surface object nodes (deprecated).

        Note that undefined surface nodes will not be included.

        Args:
            surf (RegularSurface): A XTGeo RegularSurface object instance.
            zname (str): Name of value column (3'rd column).

        Example::

            topx = RegularSurface('topx.gri')
            topx_aspoints = Points()
            topx_aspoints.from_surface(topx)

            # alternative shortform:
            topx_aspoints = Points(topx)  # get an instance directly

            topx_aspoints.to_file('mypoints.poi')  # export as XYZ file

        .. deprecated:: 2.16 Use xtgeo.points_from_surface() instead.
        """

        # check if surf is instance from RegularSurface
        if not isinstance(surf, xtgeo.surface.RegularSurface):
            raise TypeError("Given surf is not a RegularSurface object")

        val = surf.values
        xc, yc = surf.get_xy_values()

        coor = []
        for vv in [xc, yc, val]:
            vv = np.ma.filled(vv.flatten(order="C"), fill_value=np.nan)
            vv = vv[~np.isnan(vv)]
            coor.append(vv)

        # now populate the dataframe:
        xc, yc, val = coor  # pylint: disable=unbalanced-tuple-unpacking
        ddatas = OrderedDict()
        ddatas[self._xname] = xc
        ddatas[self._yname] = yc
        ddatas[zname] = val
        dfr = pd.DataFrame(ddatas)
        kwargs = {}
        kwargs["dataframe"] = dfr
        kwargs["zname"] = zname
        self._reset(**kwargs)

    @classmethod
    def _read_surface(cls, surf, zname="Z_TVDSS", name="unknown", **kwargs):
        """Get points as (X, Y, Value) from a surface object nodes.

        Note that undefined surface nodes will not be included. This method
        is perhaps only meaningful for Points.

        Args:
            surf (RegularSurface): A XTGeo RegularSurface object instance.
            zname (str): Name of value column (3'rd column)
            name (str): Name of the instance

        """
        # subsitutes from_surface()
        # check if surf is instance from RegularSurface
        if not isinstance(surf, xtgeo.surface.RegularSurface):
            raise TypeError("Given surf is not a RegularSurface object")

        val = surf.values
        xc, yc = surf.get_xy_values()

        coor = []
        for vv in [xc, yc, val]:
            vv = np.ma.filled(vv.flatten(order="C"), fill_value=np.nan)
            vv = vv[~np.isnan(vv)]
            coor.append(vv)

        # now populate the dataframe:
        xc, yc, val = coor  # pylint: disable=unbalanced-tuple-unpacking
        ddatas = OrderedDict()
        ddatas["X_UTME"] = xc
        ddatas["Y_UTMN"] = yc
        ddatas[zname] = val

        kwargs["dataframe"] = pd.DataFrame(ddatas)
        kwargs["zname"] = zname
        kwargs["name"] = name
        kwargs["filesrc"] = "Derived from: RegularSurface"

        if kwargs.get("return_cls", True) is False:
            return kwargs

        return cls(**kwargs)

    def _from_list_like(self, plist):
        """Import Points or Polygons from a list-like input.

        The following 'list-like' inputs are possible:

        * List of tuples [(x1, y1, z1, <id1>), (x2, y2, z2, <id2>), ...].
        * List of lists  [[x1, y1, z1, <id1>], [x2, y2, z2, <id2>], ...].
        * List of numpy arrays  [nparr1, nparr2, ...] where nparr1 is first row.
        * A numpy array with shape [nrow, ncol], where ncol >= 3
        * An existing pandas dataframe

        Points scenaria:
        * 3 columns, X Y Z
        * 4 or more columns: rest columns are attributes and must match len(self._attrs)

        Polygons scenaria:
        * 3 columns, X Y Z. Here P column is assigned 0 afterwards
        * 4 or more columns:
           - if totnum = lenattrs + 3 then POLY_ID is missing and will be made
           - if totnum = lenattrs + 4 then assume that 4'th column is POLY_ID

        It is currently not much error checking that lists/tuples are consistent, e.g.
        if there always is either 3 or 4 elements per tuple, or that 4 number is
        an integer.

        Args:
            plist (str): List of tuples, each tuple is length 3 or 4

        Raises:
            ValueError: If something is wrong with input

        .. versionadded:: 2.6
        .. versionupdated:: 2.16

        """
        self._filesrc = "Derived from: numpy array"
        if isinstance(plist, list):
            # convert list/tuples to a 2D numpy and process the numpy below
            logger.info("Input list-like is a list, convert to a numpy...")
            try:
                plist = np.array(plist)
            except Exception as exc:
                warnings.warn(f"Cannot convert list to numpy array: {str(exc)}")
                raise
            self._filesrc = "Derived from: list input"

        if isinstance(plist, pd.DataFrame):
            # convert input dataframe to a 2D numpy and process the numpy below
            plist = plist.to_numpy(copy=True)
            self._filesrc = "Derived from: dataframe input"

        if isinstance(plist, np.ndarray):
            logger.info("Process numpy to points")
            if plist.ndim != 2:
                raise ValueError("Input numpy array must two-dimensional")
            totnum = plist.shape[1]
            lenattrs = len(self._attrs) if self._attrs is not None else 0
            attr_first_col = 3
            if totnum == 3 + lenattrs:
                self._df = pd.DataFrame(
                    plist[:, :3], columns=[self._xname, self._yname, self._zname]
                )
                self._df = self._df.astype(float)
                if self._ispolygons:
                    # pname column is missing but assign 0 as ID
                    self._df[self._pname] = 0

            elif totnum == 4 + lenattrs and self._ispolygons:
                self._df = pd.DataFrame(
                    plist[:, :4],
                    columns=[self._xname, self._yname, self._zname, self._pname],
                )
                attr_first_col = 4
            else:
                raise ValueError(
                    f"Wrong length detected of row: {totnum}. "
                    "Are attributes set correct?"
                )
            self._df.dropna()
            self._df = self._df.astype(np.float64)
            if self._ispolygons:
                self._df[self._pname] = self._df[self._pname].astype(np.int32)

            if lenattrs > 0:
                for enum, (key, dtype) in enumerate(self._attrs.items()):
                    self._df[key] = plist[:, attr_first_col + enum]
                    self._df = self._df.astype({key: dtype})

        else:
            raise TypeError("Not possible to make XYZ from given input")

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use Points() or Polygons() initialisation instead",
    )
    def from_list(self, plist):
        """Import Points or Polygons from a list-like input (deprecated).

        The following inputs are possible:

        * List of tuples [(x1, y1, z1, <id1>), (x2, y2, z2, <id2>), ...].
        * List of lists  [[x1, y1, z1, <id1>], [x2, y2, z2, <id2>], ...].
        * List of numpy arrays  [nparr1, nparr2, ...] where nparr1 is first row.
        * A numpy array with shape [??1, ??2] ...
        * An existing pandas dataframe

        It is currently not much error checking that lists/tuples are consistent, e.g.
        if there always is either 3 or 4 elements per tuple, or that 4 number is
        an integer.

        Args:
            plist (str): List of tuples, each tuple is length 3 or 4.

        Raises:
            ValueError: If something is wrong with input

        .. versionadded:: 2.6
        .. versionupdated:: 2.16
        .. deprecated:: 2.16
           Use xtgeo.Points() or xtgeo.Polygons() directly.
        """

        first = plist[0]
        if len(first) == 3:
            self._df = pd.DataFrame(
                plist, columns=[self._xname, self._yname, self._zname]
            )

            # add ID 0 for Polygons if input is missing
            if self._ispolygons:
                self._df[self.pname] = 0

        elif len(first) == 4:
            self._df = pd.DataFrame(
                plist, columns=[self._xname, self._yname, self._zname, self._pname]
            )
        else:
            raise ValueError(
                "Wrong length detected of first tuple: {}".format(len(first))
            )
        self._df.dropna(inplace=True)
        self._filesrc = "Derived from: list-like input"

    def to_file(
        self,
        pfile,
        fformat="xyz",
        attributes=False,
        pfilter=None,
        wcolumn=None,
        hcolumn=None,
        mdcolumn="M_MDEPTH",
        **kwargs,
    ):  # pylint: disable=redefined-builtin
        """Export XYZ (Points/Polygons) to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol / rms_attr /rms_wellpicks
            attributes (bool or list): List of extra columns to export (some formats)
                or True for all attributes present
            pfilter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}.
            wcolumn (str): Name of well column (rms_wellpicks format only)
            hcolumn (str): Name of horizons column (rms_wellpicks format only)
            mdcolumn (str): Name of MD column (rms_wellpicks format only)

        Returns:
            Number of points exported

        Note that the rms_wellpicks will try to output to:

        * HorizonName, WellName, MD  if a MD (mdcolumn) is present,
        * HorizonName, WellName, X, Y, Z  otherwise

        Note:
            For backward compatibility, the key ``filter`` can be applied instead of
            ``pfilter``.

        Raises:
            KeyError if pfilter is set and key(s) are invalid

        """
        filter_deprecated = kwargs.get("filter", None)
        if filter_deprecated is not None and pfilter is None:
            pfilter = filter_deprecated

        pfile = xtgeo._XTGeoFile(pfile)
        pfile.check_folder(raiseerror=OSError)

        if self.dataframe is None:
            ncount = 0
            logger.warning("Nothing to export!")
            return ncount

        if fformat is None or fformat in ["xyz", "poi", "pol"]:
            # NB! reuse export_rms_attr function, but no attributes
            # are possible
            ncount = _xyz_io.export_rms_attr(
                self, pfile.name, attributes=False, pfilter=pfilter
            )

        elif fformat == "rms_attr":
            ncount = _xyz_io.export_rms_attr(
                self, pfile.name, attributes=attributes, pfilter=pfilter
            )
        elif fformat == "rms_wellpicks":
            ncount = _xyz_io.export_rms_wpicks(
                self, pfile.name, hcolumn, wcolumn, mdcolumn=mdcolumn
            )

        if ncount is None:
            ncount = 0

        if ncount == 0:
            logger.warning("Nothing to export!")

        return ncount

    @classmethod
    def _read_roxar(
        cls,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
        attributes=False,
        is_polygons=False,
    ):
        """Load a points/polygons item from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Use::

          import xtgeo
          mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

        Note also that horizon/zone/faults name and category must exists
        in advance, otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones/faults: for example 'DL_depth'
                or use a folder notation on clipboard.

            stype (str): RMS folder type, 'horizons' (default) or 'zones' etc!
            realisation (int): Realisation number, default is 0
            attributes (bool): If True, attributes will be preserved (from RMS 11)
            is_polygons (bool): True if Polygons

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        """
        stype = stype.lower()
        valid_stypes = ["horizons", "zones", "faults", "clipboard"]

        if stype not in valid_stypes:
            raise ValueError(
                "Invalid stype, only {} stypes is supported.".format(valid_stypes)
            )

        kwargs = _xyz_roxapi.import_xyz_roxapi(
            project, name, category, stype, realisation, attributes, is_polygons
        )

        kwargs["filesrc"] = f"RMS: {name} ({category})"
        return cls(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_roxar() or xtgeo.polygons_from_roxar() instead",
    )
    def from_roxar(
        self,
        project: Union[str, Any],
        name: str,
        category: str,
        stype: Optional[str] = "horizons",
        realisation: Optional[int] = 0,
        attributes: Optional[bool] = False,
    ):
        """Load a points/polygons item from a Roxar RMS project (deprecated).

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that the preferred shortform for (use polygons as example)::

          import xtgeo
          mypoly = xtgeo.xyz.Polygons()
          mypoly.from_roxar(project, 'TopAare', 'DepthPolys')

        is now::

          import xtgeo
          mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

        Note also that horizon/zone/faults name and category must exists
        in advance, otherwise an Exception will be raised.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones/faults: for example 'DL_depth'
                or use a folder notation on clipboard.

            stype (str): RMS folder type, 'horizons' (default) or 'zones' etc!
            realisation (int): Realisation number, default is 0
            attributes (bool): If True, attributes will be preserved (from RMS 11)

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.

        .. deprecated:: 2.16
           Use xtgeo.points_from_roxar() or xtgeo.polygons_from_roxar()
        """
        stype = stype.lower()
        valid_stypes = ["horizons", "zones", "faults", "clipboard"]

        if stype not in valid_stypes:
            raise ValueError(
                "Invalid stype, only {} stypes is supported.".format(valid_stypes)
            )

        _xyz_roxapi.import_xyz_roxapi(
            self, project, name, category, stype, realisation, attributes
        )

        self._filesrc = "RMS: {} ({})".format(name, category)

    def to_roxar(
        self,
        project,
        name,
        category,
        stype="horizons",
        pfilter=None,
        realisation=0,
        attributes=False,
        is_polygons=False,
    ):
        """Export (store) a points/polygons item to a Roxar RMS project.

        The export to the RMS project can be done either within the project
        or outside the project.

        Note also that horizon/zone name and category must exists in advance,
        otherwise an Exception will be raised.

        Note:
            When project is file path (direct access, outside RMS) then
            ``to_roxar()`` will implicitly do a project save. Otherwise, the project
            will not be saved until the user do an explicit project save action.

        Args:
            project (str or special): Name of project (as folder) if
                outside RMS, og just use the magic project word if within RMS.
            name (str): Name of polygons item
            category (str): For horizons/zones/faults: for example 'DL_depth'
            pfilter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            stype (str): RMS folder type, 'horizons' (default), 'zones'
                or 'faults' or 'clipboard'  (in prep: well picks)
            realisation (int): Realisation number, default is 0
            attributes (bool): If True, attributes will be preserved (from RMS 11)


        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.
            NotImplementedError: Not supported in this ROXAPI version

        """

        valid_stypes = ["horizons", "zones", "faults", "clipboard", "horizon_picks"]

        if stype.lower() not in valid_stypes:
            raise ValueError(
                "Invalid stype, only {} stypes is supported.".format(valid_stypes)
            )

        _xyz_roxapi.export_xyz_roxapi(
            self,
            project,
            name,
            category,
            stype.lower(),
            pfilter,
            realisation,
            attributes,
            is_polygons,
        )
