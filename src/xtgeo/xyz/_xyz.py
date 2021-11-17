# -*- coding: utf-8 -*-
"""XTGeo xyz module (base class)"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import deprecation
import pandas as pd
import xtgeo
from xtgeo.common import XTGDescription, XTGeoDialog
from xtgeo.xyz import _xyz_io, _xyz_roxapi

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class XYZ(ABC):
    """Abstract base class for XYZ objects, i.e. Points and Polygons in XTGeo.

    The XYZ base class have common methods and properties for Points and Polygons. The
    underlying data storage is a Pandas dataframe with minimal 3 (Points) or 4
    (Polygons) columns, where the two first represent X and Y coordinates.

    The third column is a number, which may represent the depth, thickness, or other
    property. For Polygons, there is a 4'th column which is an integer representing
    poly-line ID, which is handled in the Polygons class. Similarly, Points can have
    additional columns called `attributes`.

    Note:
        Do cannot use the XYZ class directly. Use the :class:`Points` or
        :class:`Polygons` classes!
    """

    def __init__(
        self,
        xname: Optional[str] = "X_UTME",
        yname: Optional[str] = "Y_UTMN",
        zname: Optional[str] = "Z_TVDSS",
        name: Optional[str] = "unknown",
        filesrc=None,
    ):
        """Concrete initialisation for base class _XYZ."""
        # attributes in common with _XYZ:
        self._xname = xname
        self._yname = yname
        self._zname = zname
        self._name = name
        self._filesrc = filesrc

        self._df = None

        logger.info("Instantation of _XYZ")

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

    # @property
    # def pname(self) -> str:
    #     """Returns or set the name of the POLY_ID column."""
    #     return self._pname

    # @pname.setter
    # def pname(self, value):
    #     try:
    #         self._check_name(value)
    #     except ValueError as verr:
    #         if "does not exist" in str(verr):
    #             return
    #     self._df_column_rename(value, self._pname)
    #     self._pname = value

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

    @abstractmethod
    def copy(self):
        """Returns a a deep copy of an instance"""
        ...

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = XTGDescription()
        dsc.title(f"Description of {self.__class__.__name__} instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("xname, yname, zname", self._xname, self._yname, self._zname)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    # ==================================================================================
    # Import and export
    # ==================================================================================

    @abstractmethod
    def from_file(self, pfile, fformat="xyz"):
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
        ...

    @abstractmethod
    def from_list(self, plist):
        """Create Points or Polygons from a list-like input (deprecated).

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
        .. versionchanged:: 2.16
        .. deprecated:: 2.16
           Use xtgeo.Points() or xtgeo.Polygons() directly.
        """
        ...

        # first = plist[0]
        # if len(first) == 3:
        #     self._df = pd.DataFrame(
        #         plist, columns=[self._xname, self._yname, self._zname]
        #     )

        #     # add ID 0 for Polygons if input is missing
        #     if self._ispolygons:
        #         self._df[self.pname] = 0

        # elif len(first) == 4:
        #     self._df = pd.DataFrame(
        #         plist, columns=[self._xname, self._yname, self._zname, self._pname]
        #     )
        # else:
        #     raise ValueError(
        #         "Wrong length detected of first tuple: {}".format(len(first))
        #     )
        # self._df.dropna(inplace=True)
        # self._filesrc = "Derived from: list-like input"

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

    # @classmethod
    # @abstractmethod
    # def _read_roxar(cls, ...):
    #     cls,
    #     project,
    #     name,
    #     category,
    #     stype="horizons",
    #     realisation=0,
    #     attributes=False,
    #     # is_polygons=False,
    # ):
    #     """Load a points/polygons item from a Roxar RMS project.

    #     The import from the RMS project can be done either within the project
    #     or outside the project.

    #     Use::

    #       import xtgeo
    #       mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

    #     Note also that horizon/zone/faults name and category must exists
    #     in advance, otherwise an Exception will be raised.

    #     Args:
    #         project (str or special): Name of project (as folder) if
    #             outside RMS, og just use the magic project word if within RMS.
    #         name (str): Name of polygons item
    #         category (str): For horizons/zones/faults: for example 'DL_depth'
    #             or use a folder notation on clipboard.

    #         stype (str): RMS folder type, 'horizons' (default) or 'zones' etc!
    #         realisation (int): Realisation number, default is 0
    #         attributes (bool): If True, attributes will be preserved (from RMS 11)
    #         is_polygons (bool): True if Polygons

    #     Returns:
    #         Object instance updated

    #     Raises:
    #         ValueError: Various types of invalid inputs.

    #     """
    #     stype = stype.lower()
    #     valid_stypes = ["horizons", "zones", "faults", "clipboard"]

    #     if stype not in valid_stypes:
    #         raise ValueError(
    #             "Invalid stype, only {} stypes is supported.".format(valid_stypes)
    #         )

    #     kwargs = _xyz_roxapi.import_xyz_roxapi(
    #         project, name, category, stype, realisation, attributes, is_polygons
    #     )

    #     kwargs["filesrc"] = f"RMS: {name} ({category})"
    #     return cls(**kwargs)

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
