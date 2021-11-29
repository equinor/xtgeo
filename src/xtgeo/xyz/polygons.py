"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a INT column called 'POLY_ID'
# which identifies each polygon piece.
import functools
import io
import pathlib
import warnings
from copy import deepcopy
from typing import Any, List, Optional, TypeVar, Union

import deprecation
import numpy as np
import pandas as pd
import shapely.geometry as sg
import xtgeo
from xtgeo.common import inherit_docstring
from xtgeo.xyz import _xyz_io, _xyz_roxapi
from xtgeo.xyz._xyz_io import _ValidDataFrame

from . import _xyz_oper
from ._xyz import XYZ
from ._xyz_io import _convert_idbased_xyz

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

Wells = TypeVar("Wells")


class ValidationError(ValueError):
    ...


# ======================================================================================
# Private functions outside class
# ======================================================================================


def _data_reader_factory(file_format):
    if file_format == "xyz":
        return _xyz_io.import_xyz
    if file_format == "zmap_ascii":
        return _xyz_io.import_zmap
    raise ValueError(f"Unknown file format {file_format}")


def _file_importer(
    pfile: Union[str, pathlib.Path, io.BytesIO],
    fformat: Optional[str] = None,
):
    """General function for polygons_from_file and (deprecated) method from_file."""
    pfile = xtgeo._XTGeoFile(pfile)
    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat()
    else:
        fformat = pfile.generic_format_by_proposal(fformat)  # default
    kwargs = _data_reader_factory(fformat)(pfile, is_polygons=True)

    kwargs["name"] = "polygons"
    kwargs["filesrc"] = pfile.name

    return kwargs


def _roxar_importer(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
):
    stype = stype.lower()
    valid_stypes = ["horizons", "zones", "faults", "clipboard"]

    if stype not in valid_stypes:
        raise ValueError(f"Invalid stype, only {valid_stypes} stypes is supported.")

    kwargs = _xyz_roxapi.import_xyz_roxapi(
        project, name, category, stype, realisation, None, True
    )

    kwargs["name"] = "polygons"
    kwargs["filesrc"] = f"Derived from: RMS {name} ({category})"
    return kwargs


def _wells_importer(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    zonelogname: Optional[str] = None,
    zone: Optional[int] = None,
    resample: Optional[int] = 1,
):
    """Get line segments from a list of wells and a single zone number.

    A future extension is that zone could be a list of zone numbers and/or mechanisms
    to retrieve well segments by other measures, e.g. >= depth.
    """

    if not wells:
        raise ValueError("No valid input wells")

    # wells in a scalar context is allowed if one well
    if not isinstance(wells, list):
        wells = [wells]

    # wells may be just files which need to be imported, which is a bit more fragile
    if isinstance(wells[0], (str, pathlib.Path)):
        wells = [xtgeo.well_from_file(wll) for wll in wells]
    dflist = []
    maxid = 0
    for well in wells:
        if zonelogname is not None:
            well.zonelogname = zonelogname

        wp = well.get_zone_interval(zone, resample=resample)
        wp["WellName"] = well.name
        if wp is not None:
            # as well segments may have overlapping POLY_ID:
            wp["POLY_ID"] += maxid
            maxid = wp["POLY_ID"].max() + 1
            dflist.append(wp)

    if dflist:
        dfr = pd.concat(dflist, ignore_index=True)
        dfr.reset_index(inplace=True, drop=True)
    else:
        return None
    kwargs = {}
    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["attributes"] = {"WellName": "str"}
    kwargs["filesrc"] = "Derived from: Well segments"
    return kwargs


def _allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of Polygons and should be deleted once the deprecation period has expired,
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
                    "pol = xtgeo.polygons_from_file('some_file.xx') instead!",
                    DeprecationWarning,
                )
                fformat = kwargs.get("fformat", "guess")
                kwargs = _file_importer(args[0], fformat)

            elif isinstance(args[0], (list, np.ndarray, pd.DataFrame)):
                # initialisation from an list-like object without 'values' keyword
                # should be possible, i.e. Polygons(some_list) is same as
                # Polygons(values=some_list)
                kwargs["values"] = args[0]

            else:
                raise TypeError("Input argument of unknown type: ", type(args[0]))

        return func(cls, **kwargs)

    return wrapper


# ======================================================================================
# FUNCTIONS as wrappers to class init + import
# ======================================================================================


def polygons_from_file(
    pfile: Union[str, pathlib.Path], fformat: Optional[str] = "guess"
) -> "Polygons":
    """Make an instance of a Polygons object directly from file import.

    Supported formats are:

        * 'xyz' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'guess': Try to choose file format based on extension

    Args:
        pfile (str): Name of file
        fformat (str): See :meth:`Polygons.from_file`

    Example::

        import xtgeo
        mypoly = xtgeo.polygons_from_file('somefile.xyz')
    """
    return Polygons(**_file_importer(pfile, fformat=fformat))


def polygons_from_roxar(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
) -> "Polygons":
    """Load a Polygons instance from Roxar RMS project.

    Note also that horizon/zone/faults name and category must exists
    in advance, otherwise an Exception will be raised.

    Args:
        project: Name of project (as folder) if outside RMS, or just use the magic
            `project` word if within RMS.
        name: Name of polygons item
        category: For horizons/zones/faults: for example 'DL_depth'
            or use a folder notation on clipboard.
        stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard',
            'faults', ...
        realisation: Realisation number, default is 0

    Example::

        import xtgeo
        mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')
    """

    kwargs = _roxar_importer(
        project,
        name,
        category,
        stype,
        realisation,
    )

    return Polygons(**kwargs)


def polygons_from_wells(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    zonelogname: Optional[str] = None,
    zone: Optional[int] = 1,
    resample: Optional[int] = 1,
) -> "Polygons":

    """Get polygons from wells and a single zone number.

    Args:
        wells: List of XTGeo well objects, a single XTGeo well or a list of well files.
            If a list of well files, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        zonelogname: Name of zonelog; if not given it will be taken from the well
            property well.zonelogname
        zone: The zone number to extract the linepiece from
        resample: If given, resample every N'th sample to make
            polylines smaller in terms of bits and bytes.
            1 = No resampling, which means just use well sampling (which can be rather
            dense; typically 15 cm).


    Returns:
        None if empty data, otherwise a Polygons() instance.

    Example::

        wells = ["w1.w", "w2.w"]
        points = xtgeo.polygons_from_wells(wells, zonelogname="ZONELOG", zone=2)

    Note:
        This method replaces the deprecated method :py:meth:`~Polygons.from_wells`.
        The latter returns the number of wells that contribute with polygon segments.
        This is now implemented through the function `get_nwells()`. Hence the
        following code::

            nwells_applied = poly.from_wells(...)  # deprecated method
            # vs
            poly = xtgeo.polygons_from_wells(...)
            nwells_applied = poly.get_nwells()

    .. versionadded: 2.16
    """
    kwargs = _wells_importer(wells, zonelogname, zone, resample)

    if kwargs is not None:
        return Polygons(**kwargs)

    return None


########################################################################################
# Polygons class
########################################################################################
class Polygons(XYZ):  # pylint: disable=too-many-public-methods
    """Class for a Polygons object (connected points) in XTGeo.

    The term Polygons is here used in a wider context, as it includes
    polylines that do not connect into closed polygons. A Polygons
    instance may contain several pieces of polylines/polygons, which are
    identified by POLY_ID.

    The polygons are stored in Python as a Pandas dataframe, which
    allow for flexible manipulation and fast execution.

    A Polygons instance will have 4 mandatory columns; here by default names:

    * X_UTME - for X UTM coordinate (Easting)
    * Y_UTMN - For Y UTM coordinate (Northing)
    * Z_TVDSS - For depth or property from mean SeaLevel; Depth positive down
    * POLY_ID - for polygon ID as there may be several polylines segments

    Each Polygons instance can also a informal name (through the name attribute).
    Default is 'polygons'. E.g. if a well fence, it is logical to name the
    instance to be the same as the well name.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column, default is X_UTME.
        yname: Name of second (Y) mandatory column, default is Y_UTMN.
        zname: Name of third (Z) mandatory column, default is Z_TVDSS.
        pname: Name of forth (P) mandatory enumerating column, default is POLY_ID.
        name: A given name for the Points/Polygons object, defaults to 'points'.
        attributes: A dictionary for attribute columns as 'name: type', e.g.
            {"WellName": "str", "IX": "int"}. This is applied when values are input
            and is to name and type the extra attribute columns in a point set.
        filesrc: Spesify input file name or other origin (informal)
    """

    @_allow_deprecated_init
    def __init__(
        self,
        values: Optional[Union[list, np.ndarray, pd.DataFrame]] = None,
        xname: Optional[str] = "X_UTME",
        yname: Optional[str] = "Y_UTMN",
        zname: Optional[str] = "Z_TVDSS",
        pname: Optional[str] = "POLY_ID",
        name: Optional[str] = "polygons",
        attributes: Optional[dict] = None,
        filesrc: Optional[str] = None,
    ):

        super().__init__(xname, yname, zname, name, filesrc)

        # additonal state properties for Polygons
        self._pname = pname

        self._hname = None
        self._dhname = None
        self._tname = None
        self._dtname = None

        self._df, self._filesrc = _xyz_io.initialize_by_values(
            values, self._zname, attributes, True
        )
        logger.info("Initiated Polygons")

    # ==================================================================================
    # Properties
    # See also base class
    # ==================================================================================

    @property
    def attributes(self) -> dict:
        """Returns a dictionary with attribute names and type, or None."""
        # this is not stored as state variable any more, but is dynamically stored in
        # all pandas columns after the 4 first.
        attrs = {}
        for col in self._df.columns[4:]:
            if col:
                name = col
                dtype = str(self._df[col].dtype).lower()
                if "int" in dtype:
                    dtype = "int"
                elif "float" in dtype:
                    dtype = "float"
                else:
                    dtype = "str"
                attrs[name] = dtype

        if attrs:
            return attrs
        return None

    @property
    def pname(self):
        """Returns or sets the name of the Polygons ID column."""
        return self._pname

    @pname.setter
    def pname(self, newname):
        if self.rename_column(self._pname, newname):
            self._pname = newname

    @property
    def hname(self):
        """Returns or set the name of the cumulative horizontal length.

        If the column does not exist, None is returned. Default name is H_CUMLEN.

        .. versionadded:: 2.1
        """
        return self._hname

    @hname.setter
    def hname(self, newname):
        if self._hname is not None:
            if self.rename_column(self._hname, newname):
                self._hname = newname
        else:
            raise ValueError("Cannot rename hname, which is currently None.")

    @property
    def dhname(self):
        """Returns or set the name of the delta horizontal length column if it exists.

        If the column does not exist, None is returned. Default name is H_DELTALEN.

        .. versionadded:: 2.1
        """
        return self._dhname

    @dhname.setter
    def dhname(self, newname):
        if self._dhname is not None:
            if self.rename_column(self._dhname, newname):
                self._dhname = newname
        else:
            raise ValueError("Cannot rename dhname, which is currently None.")

    @property
    def tname(self):
        """Returns or set the name of the cumulative total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._tname

    @tname.setter
    def tname(self, newname):
        if self._tname is not None:
            if self.rename_column(self._tname, newname):
                self._tname = newname
        else:
            raise ValueError("Cannot rename tname, which is currently None.")

    @property
    def dtname(self):
        """Returns or set the name of the delta total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._dtname

    @dtname.setter
    def dtname(self, newname):
        if self._dtname is not None:
            if self.rename_column(self._dtname, newname):
                self._dtname = newname
        else:
            raise ValueError("Cannot rename dtname, which is currently None.")

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input df is not a DataFrame, but a {type(df)} instance.")
        if len(df.columns) < 4:
            raise ValidationError(
                "Input dataframe has too few columns (need >= 4) "
                f"but has {len(df.columns)}"
            )
        self._xname, self._yname, self._zname, self._pname = df.columns[0:4]
        self._df = df.apply(deepcopy)  # see comment note in _xyz.py
        self._name_to_none_if_missing()

    # ==================================================================================
    # Private methods
    # ==================================================================================

    def _reset(self, **kwargs):
        """Used in deprecated methods."""
        self._df = kwargs.get("values", self._df)
        self._xname = kwargs.get("xname", self._xname)
        self._yname = kwargs.get("yname", self._yname)
        self._zname = kwargs.get("zname", self._zname)
        self._pname = kwargs.get("pname", self._pname)

        self._name = kwargs.get("name", self._name)
        self._filesrc = kwargs.get("filesrc", self._filesrc)

    def _name_to_none_if_missing(self):
        if self._dtname not in self._df.columns:
            self._dtname = None
        if self._dhname not in self._df.columns:
            self._dhname = None
        if self._tname not in self._df.columns:
            self._tname = None
        if self._hname not in self._df.columns:
            self._hname = None

    # ==================================================================================
    # I/O methods
    # from_ methods are deprecated
    # ==================================================================================

    # from stuff -----------------------------------------------------------------------

    @inherit_docstring(inherit_from=XYZ.from_file)
    def from_file(self, pfile, fformat="xyz"):
        self._reset(**_file_importer(pfile, fformat))

    @inherit_docstring(inherit_from=XYZ.from_list)
    @deprecation.deprecated(
        deprecated_in="2.15",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use direct Polygons() initialisation instead",
    )
    def from_list(self, plist):

        kwargs = {}
        kwargs["values"], kwargs["filesrc"] = _xyz_io._from_list_like(
            plist, "Z_TVDSS", None, True
        )
        self._reset(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.polygons_from_wells(...) instead",
    )
    def from_wells(self, wells, zone, resample=1):
        """Get line segments from a list of wells and a single zone number.

        Args:
            wells (list): List of XTGeo well objects
            zone (int): Which zone to apply
            resample (int): If given, resample every N'th sample to make
                polylines smaller in terms of bits and bytes.
                1 = No resampling which means well sampling (which can be rather
                dense; typically 15 cm).

        Returns:
            None if well list is empty; otherwise the number of wells that
            have one or more line segments to return

        """
        kwargs = _wells_importer(wells, None, zone, resample)

        if kwargs is not None:
            self._reset(**kwargs)

        nwells = self.get_nwells()
        # as the previous versions did not have the WellName column, this is dropped
        # here for backward compatibility:
        self.dataframe = self.dataframe.drop("WellName", axis=1)
        return nwells

    # to... stuff ----------------------------------------------------------------------

    def to_file(
        self,
        pfile,
        fformat="xyz",
    ):
        """Export Polygons to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol

        Returns:
            Number of polygon points exported
        """

        return _xyz_io.to_generic_file(
            self,
            pfile,
            fformat=fformat,
        )

    def to_roxar(
        self,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
    ):
        """Export (store) a Polygons item to a Roxar RMS project.

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
            stype (str): RMS folder type, 'horizons' (default), 'zones'
                or 'faults' or 'clipboard'  (in prep: well picks)
            realisation (int): Realisation number, default is 0


        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.
            NotImplementedError: Not supported in this ROXAPI version

        """

        valid_stypes = ["horizons", "zones", "faults", "clipboard"]

        if stype.lower() not in valid_stypes:
            raise ValueError(f"Invalid stype, only {valid_stypes} stypes is supported.")

        _xyz_roxapi.export_xyz_roxapi(
            self,
            project,
            name,
            category,
            stype.lower(),
            None,
            realisation,
            None,
            True,
        )

    # ==================================================================================
    # Other methods
    # ==================================================================================

    def copy(self):
        """Returns a deep copy of an instance"""
        mycopy = self.__class__()
        mycopy._df = self._df.apply(deepcopy)  # df.copy() is not fully deep!
        mycopy._xname = self._xname
        mycopy._yname = self._yname
        mycopy._zname = self._zname
        mycopy._pname = self._pname
        mycopy._hname = self._hname
        mycopy._dhname = self._dhname
        mycopy._tname = self._tname
        mycopy._dtname = self._dtname
        mycopy._name = self._name

        mycopy._filesrc = "Derived from: copied instance"

        return mycopy

    @inherit_docstring(inherit_from=XYZ.delete_columns)
    def delete_columns(self, clist, strict=False):
        _xyz_oper.delete_columns(self, clist, strict, True)

    def get_xyz_dataframe(self):
        """Get a dataframe copy from the Polygons points with no ID column.

        Convert from POLY_ID based to XYZ, where a new polygon is marked with a 999
        value as flag.
        """
        return _convert_idbased_xyz(self, self.dataframe)

    def get_shapely_objects(self):
        """Returns a list of Shapely LineString objects, one per POLY_ID.

        .. versionadded:: 2.1
        """
        spolys = []
        idgroups = self.dataframe.groupby(self.pname)

        for _idx, grp in idgroups:
            pxcor = grp[self.xname].values
            pycor = grp[self.yname].values
            pzcor = grp[self.zname].values
            spoly = sg.LineString(np.stack([pxcor, pycor, pzcor], axis=1))
            spolys.append(spoly)

        return spolys

    def get_boundary(self):
        """Get the XYZ window (boundaries) of the instance.

        Returns:
            (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        xmin = np.nanmin(self.dataframe[self.xname].values)
        xmax = np.nanmax(self.dataframe[self.xname].values)
        ymin = np.nanmin(self.dataframe[self.yname].values)
        ymax = np.nanmax(self.dataframe[self.yname].values)
        zmin = np.nanmin(self.dataframe[self.zname].values)
        zmax = np.nanmax(self.dataframe[self.zname].values)

        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def filter_byid(self, polyid=None):
        """Remove all line segments not in polyid.

        The instance is updated in-place.

        Args:
            polyid (int or list of int): Which ID(s) to keep, None means use first.

        Example::

            mypoly.filter_byid(polyid=[2, 4])  # keep POLY_ID 2 and 4

        .. versionadded:: 2.1
        """
        if polyid is None:
            polyid = int(self.dataframe[self.pname].iloc[0])

        if not isinstance(polyid, list):
            polyid = [polyid]

        dflist = []
        for pid in polyid:
            dflist.append(self.dataframe[self.dataframe[self.pname] == pid])

        self.dataframe = pd.concat(dflist)

    def tlen(self, tname="T_CUMLEN", dtname="T_DELTALEN", atindex=0):
        """Compute and add or replace columns for cum. total 3D length and delta length.

        The instance is updated in-place.

        Args:
            tname (str): Name of cumulative total length. Default is T_CUMLEN.
            dtname (str): Name of delta length column. Default is T_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1
        """
        _xyz_oper.tlen(self, tname=tname, dtname=dtname, atindex=atindex)

    def hlen(self, hname="H_CUMLEN", dhname="H_DELTALEN", atindex=0):
        """Compute and add/replace columns for cum. horizontal length and delta length.

        The instance is updated in-place.

        Args:
            hname (str): Name of cumulative horizontal length. Default is H_CUMLEN.
            dhname (str): Name of delta length column. Default is H_DELTALEN.
            atindex (int): Which index which shall be 0.0 for cumulative length.

        .. versionadded:: 2.1
        """
        _xyz_oper.hlen(self, hname=hname, dhname=dhname, atindex=atindex)

    def extend(self, distance, nsamples=1, mode2d=True):
        """Extend polyline by `distance` at both ends, nsmaples times.

        The instance is updated in-place.

        Args:
            distance (float): The horizontal distance (sampling) to extend
            nsamples (int): Number of samples to extend.
            mode2d (bool): XY extension (only True is supported)

        .. versionadded:: 2.1
        """
        _xyz_oper.extend(self, distance, nsamples, mode2d)

    def rescale(self, distance, addlen=False, kind="simple", mode2d=True):
        """Rescale (resample) by using a new increment.

        The increment (distance) may be a horizontal or a True 3D
        distance dependent on mode2d.

        The instance is updated in-place.

        If the distance is larger than the total input poly-line length,
        nothing is done. Note that the result distance may differ from the
        requested distance caused to rounding to fit original length.

        Hence actual distance is input distance +- 50%.

        Args:
            distance (float): New distance between points
            addlen (str): If True, total and horizontal cum. and delta length
                columns will be added.
            kind (str): What kind of rescaling: slinear/cubic/simple
            mode2d (bool): The distance may be a 2D (XY) ora 3D (XYZ) mode.

        .. versionchanged:: 2.1 a new algorithm

        """
        _xyz_oper.rescale_polygons(
            self, distance=distance, addlen=addlen, kind=kind, mode2d=mode2d
        )

    def get_fence(
        self, distance=20, atleast=5, nextend=2, name=None, asnumpy=True, polyid=None
    ):
        """Extracts a fence with constant horizontal sampling.

        Additonal H_CUMLEN and H_DELTALEN vectors will be added, suitable for
        X sections.

        Args:
            distance (float): New horizontal distance between points
            atleast (int): Minimum number of points. If the true length/atleast is
                less than distance, than distance will be be reset to
                length/atleast. Values below 3 are not permitted
            nextend (int): Number of samples to extend at each end. Note that
                in case of internal resetting of distance (due to 'atleast'), then
                nextend internally will be modified in order to fulfill the
                initial intention. Hence keep distance*nextend as target.
            name (str): Name of polygon (if asnumpy=False)
            asnumpy (bool): Return a [:, 5] numpy array with
                columns X.., Y.., Z.., HLEN, dH
            polyid (int): Which POLY_ID to use. Default (if None) is to use the
                first found.

        Returns:
            A numpy array (if asnumpy=True) or a new Polygons() object

        .. versionadded:: 2.1
        """
        logger.info("Getting fence within a Polygons instance...")
        return _xyz_oper.get_fence(
            self,
            distance=distance,
            atleast=atleast,
            nextend=nextend,
            name=name,
            asnumpy=asnumpy,
            polyid=polyid,
        )

    # ==================================================================================
    # Plotting
    # ==================================================================================

    def quickplot(
        self,
        filename=None,
        others=None,
        title="QuickPlot for Polygons",
        subtitle=None,
        infotext=None,
        linewidth=1.0,
        color="r",
    ):
        """Simple plotting of polygons using matplotlib.

        Args:
            filename (str): Name of plot file; None will plot to screen.
            others (list of Polygons): List of other polygon instances to plot
            title (str): Title of plot
            subtitle (str): Subtitle of plot
            infotext (str): Additonal info on plot.
            linewidth (float): Width of line.
            color (str): Name of color (may use matplotib shortcuts, e.g. 'r' for 'red')
        """
        mymap = xtgeo.plot.Map()
        mymap.canvas(title=title, subtitle=subtitle, infotext=infotext)

        if others:
            for other in others:
                lwid = linewidth / 2.0
                mymap.plot_polygons(
                    other, idname=other.pname, linewidth=lwid, color="black"
                )

        mymap.plot_polygons(self, idname=self.pname, linewidth=linewidth, color=color)

        if filename is None:
            mymap.show()
        else:
            mymap.savefig(filename)

    # ==================================================================================
    # Operations restricted to inside/outside polygons
    # See base class!
    # ==================================================================================
    ...
