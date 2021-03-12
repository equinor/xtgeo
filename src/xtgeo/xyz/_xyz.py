# -*- coding: utf-8 -*-
"""XTGeo xyz module (base class)"""
from collections import OrderedDict
from copy import deepcopy
import pathlib

import pandas as pd

import xtgeo
from xtgeo.common import XTGeoDialog, XTGDescription
from xtgeo.xyz import _xyz_io
from xtgeo.xyz import _xyz_roxapi

xtg = XTGeoDialog()
logger = xtg.functionlogger(__name__)


class XYZ:
    """Base class for Points and Polygons in XTGeo."""

    def __init__(self, *args, **kwargs):
        """Initiate instance"""

        self._df = None
        self._ispolygons = False
        self._xname = "X_UTME"
        self._yname = "Y_UTMN"
        self._zname = "Z_TVDSS"
        self._pname = "POLY_ID"
        self._mname = "M_MDEPTH"
        self._filesrc = None
        # other attributes name: type, where type is
        # ~ ('str', 'int', 'float', 'bool')
        self._attrs = OrderedDict()

        if len(args) == 1:
            if isinstance(args[0], (str, pathlib.Path)):
                # make instance from file import
                pfile = args[0]
                logger.info("Instance from file")
                fformat = kwargs.get("fformat", "guess")
                self.from_file(pfile, fformat=fformat)

            if isinstance(args[0], list):
                # make instance from a list of 3 or 4 tuples
                plist = args[0]
                logger.info("Instance from list")
                self.from_list(plist)

        logger.info("XYZ Instance initiated (base class) ID %s", id(self))

    @property
    def xname(self):
        """ Returns or set the name of the X column."""
        return self._xname

    @xname.setter
    def xname(self, newname):
        self._df_column_rename(newname, self._xname)
        self._xname = newname

    @property
    def yname(self):
        """ Returns or set the name of the Y column."""
        return self._yname

    @yname.setter
    def yname(self, newname):
        self._df_column_rename(newname, self._yname)
        self._yname = newname

    @property
    def zname(self):
        """ Returns or set the name of the Z column."""
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
        mycopy._mname = self._mname
        mycopy._filescr = self._filesrc = None
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

        froot, fext = pfile.splitext(lower=True)
        if fformat == "guess":
            if not fext:
                logger.critical("File extension missing. STOP")
                raise SystemExit

            fformat = fext

        if fformat in ["xyz", "poi", "pol"]:
            _xyz_io.import_xyz(self, pfile.name)
        elif fformat == "zmap":
            _xyz_io.import_zmap(self, pfile.name)
        elif fformat in ("rms_attr", "rmsattr"):
            _xyz_io.import_rms_attr(self, pfile.name)
        else:
            logger.error("Invalid file format (not supported): %s", fformat)
            raise ValueError("Invalid file format (not supported): %s", fformat)

        logger.info("Reading from file %s... done", pfile.name)
        logger.debug("Dataframe head:\n%s", self._df.head())
        self._filesrc = pfile.name

        return self

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
