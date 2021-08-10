# -*- coding: utf-8 -*-
"""XTGeo xyz module (base class)"""
from collections import OrderedDict
from copy import deepcopy
import pathlib
import functools
import warnings
import deprecation
import io
from typing import Union, Optional, Any

import pandas as pd

import xtgeo
from xtgeo.common import XTGeoDialog, XTGDescription
from xtgeo.xyz import _xyz_io
from xtgeo.xyz import _xyz_roxapi
import xtgeo.common.sys as xtgeosys

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


def _data_reader_factory(file_format):
    if file_format in ("xyz", "poi", "pol"):
        return _xyz_io.import_xyz
    if file_format == "zmap":
        return _xyz_io.import_zmap
    if file_format in ("rms_attr", "rmsattr"):
        return _xyz_io.import_rms_attr
    raise ValueError(f"Unknown file format {file_format}")


def allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of Points/Polygons and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    @functools.wraps(func)
    def wrapper(cls, spec, fformat="xyz", zname="Z_TVDSS", is_polygons=False):
        # Checking if we are doing an initialization from file and raise a
        # deprecation warning if we are.
        derived_importname = "points_from_file"
        if is_polygons:
            derived_importname = "polygons_from_file"

        if isinstance(spec, (str, pathlib.Path)):
            warnings.warn(
                "Initializing directly from file name is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                f"some = xtgeo.{derived_importname}('some_file.xx') instead",
                DeprecationWarning,
            )
            pfile = xtgeosys._XTGeoFile(spec)
            print("XXX", pfile)
            if fformat is None or fformat == "guess":
                fformat = pfile.detect_fformat()
            else:
                fformat = pfile.generic_format_by_proposal(fformat)  # default
            _data_reader_factory(fformat)(pfile, zname=zname)
            return func(
                cls, spec, fformat=fformat, zname=zname, is_polygons=is_polygons
            )

        return func(cls, spec, zname=zname, is_polygons=is_polygons)

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
    stored in ordered dict: self._attrs as {"somename": "type", ...}
    """

    @allow_deprecated_init
    def __init__(
        self,
        spesification: Optional[Any] = None,
        xname: Optional[str] = "X_UTME",
        yname: Optional[str] = "X_UTME",
        zname: Optional[str] = "X_UTME",
        pname: Optional[str] = "POLY_ID",
        dfr: Optional[pd.DataFrame] = None,
        is_polygons: Optional[bool] = False,
    ):
        """Initiate instance"""

        self._df = dfr
        self._ispolygons = is_polygons
        self._xname = xname
        self._yname = yname
        self._zname = zname
        self._pname = "POLY_ID"
        self._filesrc = None
        # other attributes as (name: type), where type is
        # ~ ('str', 'int', 'float', 'bool')
        self._attrs = OrderedDict()

        if spesification is not None:

            if isinstance(spesification, list):
                # make instance from a list of 3 or 4 tuples
                logger.info("Instance from list")
                self.from_list(spesification)

        logger.info("XYZ Instance initiated (base class) ID %s", id(self))

    @property
    def xname(self):
        """Returns or set the name of the X column."""
        return self._xname

    @xname.setter
    def xname(self, newname):
        self._df_column_rename(newname, self._xname)
        self._xname = newname

    @property
    def yname(self):
        """Returns or set the name of the Y column."""
        return self._yname

    @yname.setter
    def yname(self, newname):
        self._df_column_rename(newname, self._yname)
        self._yname = newname

    @property
    def zname(self):
        """Returns or set the name of the Z column."""
        return self._zname

    @zname.setter
    def zname(self, newname):
        self._df_column_rename(newname, self._zname)
        self._zname = newname

    @property
    def pname(self):
        """Returns or set the name of the POLY_ID column."""
        return self._pname

    @pname.setter
    def pname(self, value):
        self._check_name(value)
        self._pname = value

    @property
    def attributes(self):
        """Returns a dictionary with attribute names and type, or None."""
        return self._attrs

    @property
    def dataframe(self):
        """Returns or set the Pandas dataframe object."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.copy()

    @property
    def nrow(self):
        """Returns the Pandas dataframe object number of rows."""
        if self.dataframe is None:
            return 0
        return len(self.dataframe.index)

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

    def delete_columns(self, clist, strict=False):
        """Delete one or more columns by name in a safe way.

        Note that the coordinate columns will be protected, as well as then
        POLY_ID column (pname atribute) if Polygons.

        Args:
            clist (list): Name of columns
            strict (bool): I False, will not trigger exception if a column is not
                found. Otherways a ValueError will be raised.

        Raises:
            ValueError: If strict is True and columnname not present

        Example::

            mypoly.delete_columns(["WELL_ID", mypoly.hname, mypoly.dhname])

        .. versionadded:: 2.1

        """
        if self._df is None:
            xtg.warnuser(
                "Trying to delete a column before a dataframe has been set - ignored"
            )
            return

        for cname in clist:
            if (
                self._ispolygons
                and cname in (self.xname, self.yname, self.zname, self.pname)
                or cname in (self.xname, self.yname, self.zname)
            ):
                xtg.warnuser(
                    "The column {} is protected and will not be deleted".format(cname)
                )
                continue

            if cname not in self._df:
                if strict is True:
                    raise ValueError("The column {} is not present".format(cname))

            if cname in self._df:
                self._df.drop(cname, axis=1, inplace=True)
                del self._attrs[cname]

    # ==================================================================================
    # Import and export
    # ==================================================================================
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_file() or xtgeo.polygons_from_file() instead",
    )
    def from_file(self, pfile, fformat="xyz"):
        """Import Points or Polygons from a file.

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

        """

        pfile = xtgeo._XTGeoFile(pfile)

        logger.info("Reading from file %s...", pfile.name)

        pfile.check_file(raiseerror=OSError)

        _, fext = pfile.splitext(lower=True)
        if fformat == "guess":
            if not fext:
                raise ValueError(f"File extension missing for file: {pfile}")

            fformat = fext

        if fformat in ["xyz", "poi", "pol"]:
            _xyz_io.import_xyz(self, pfile.name)
        elif fformat == "zmap":
            _xyz_io.import_zmap(self, pfile.name)
        elif fformat in ("rms_attr", "rmsattr"):
            _xyz_io.import_rms_attr(self, pfile.name)
        else:
            logger.error("Invalid file format (not supported): %s", fformat)
            raise ValueError(f"Invalid file format (not supported): {fformat}")

        logger.info("Reading from file %s... done", pfile.name)
        logger.debug("Dataframe head:\n%s", self._df.head())
        self._filesrc = pfile.name

        return self

    @classmethod
    def _read_file(
        cls, pfile: Union[str, pathlib.Path, io.BytesIO], fformat: Optional[str] = None
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

        Returns:
            Object instance.

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
        _data_reader_factory(fformat)(pfile)
        return cls()

    def from_list(self, plist):
        """Import Points or Polygons from a list.

        [(x1, y1, z1, <id1>), (x2, y2, z2, <id2>), ...]

        It is currently not much error checking that lists/tuples are consistent, e.g.
        if there always is either 3 or 4 elements per tuple, or that 4 number is
        an integer.



        Args:
            plist (str): List of tuples, each tuple is length 3 or 4

        Raises:
            ValueError: If something is wrong with input

        .. versionadded: 2.6
        """

        first = plist[0]
        if len(first) == 3:
            self._df = pd.DataFrame(
                plist, columns=[self._xname, self._yname, self._zname]
            )
            print("XXX DONG")

            # add ID 0 for Polygons if input is missing
            if self._ispolygons:
                print("XXX DING")
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
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            wcolumn (str): Name of well column (rms_wellpicks format only)
            hcolumn (str): Name of horizons column (rms_wellpicks format only)
            mdcolumn (str): Name of MD column (rms_wellpicks format only)

        Returns:
            Number of points exported

        Note that the rms_wellpicks will try to output to:

        * HorizonName, WellName, MD  if a MD (mdcolumn) is present,
        * HorizonName, WellName, X, Y, Z  otherwise

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

    def from_roxar(
        self, project, name, category, stype="horizons", realisation=0, attributes=False
    ):
        """Load a points/polygons item from a Roxar RMS project.

        The import from the RMS project can be done either within the project
        or outside the project.

        Note that a shortform (for polygons) to::

          import xtgeo
          mypoly = xtgeo.xyz.Polygons()
          mypoly.from_roxar(project, 'TopAare', 'DepthPolys')

        is::

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
        )

    def append(self, other, attributes=None):
        """Append objects by adding dataframes together.

        Args:
            other: Points or Polygons object
            attributes: List of attribute names one wants to append in addition.
                Note that the two polygons must share attribute name if this should
                have any meaning. As default, attributes are None which means that
                any additional columns are not applied, and the resulting Polygons()
                instance will only keep the basic columns. This is intentional
                behaviour.

        Example::

            p1 = xtgeo.polygons_from_file("some1.pol")
            p2 = xtgeo.polygons_from_file("some2.pol")

            p1.append(p2)

        """
        df1 = self.dataframe
        df2 = other.dataframe

        other.xname = self.xname
        other.yname = self.yname
        other.zname = self.zname

        for key in self.attributes.copy().keys():
            if attributes is not None and key in attributes:
                continue
            else:
                self.delete_columns([key])

        for key in other.attributes.copy().keys():
            if attributes is not None and key in attributes:
                continue
            else:
                other.delete_columns([key])

        if self._ispolygons is True:
            # the polygon ID shall not be repeatable; need to deduce the polygons ID
            # from both objects and than ensure that counting in the second (df2)
            # is altered to avoid overlap
            id1 = df1[self.pname]
            id2 = df2[other.pname]
            diff = id1.max() - id2.min()
            id2 += diff + 1
            other.pname = self.pname

        # just append the two dataframes
        newdf = df1.append(df2, ignore_index=True)
        self.dataframe = newdf
