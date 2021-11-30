# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class."""
import functools
import io
import pathlib
import warnings
from collections import OrderedDict
from copy import deepcopy
from typing import Any, List, Optional, TypeVar, Union

import deprecation
import numpy as np
import pandas as pd
import xtgeo
from xtgeo.common import inherit_docstring
from xtgeo.xyz import _xyz_io, _xyz_roxapi
from xtgeo.xyz._xyz_io import _ValidDataFrame

from . import _xyz_oper
from ._xyz import XYZ

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)

RegularSurface = TypeVar("RegularSurface")
Points = TypeVar("Points")
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
    if file_format == "rms_attr":
        return _xyz_io.import_rms_attr
    raise ValueError(f"Unknown file format {file_format}")


def _file_importer(
    pfile: Union[str, pathlib.Path, io.BytesIO],
    fformat: Optional[str] = None,
):
    """General function for points_from_file and (deprecated) method from_file."""
    pfile = xtgeo._XTGeoFile(pfile)
    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat()
    else:
        fformat = pfile.generic_format_by_proposal(fformat)  # default
    kwargs = _data_reader_factory(fformat)(pfile, is_polygons=False)

    kwargs["name"] = "points"
    kwargs["filesrc"] = pfile.name

    return kwargs


def _surface_importer(surf, zname="Z_TVDSS", name="points"):
    """General function for _read_surface() and (deprecated) method from_surface()."""
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
    dfr = pd.DataFrame(ddatas)
    kwargs = {}
    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["zname"] = zname
    kwargs["filesrc"] = f"Derived from: surface {surf.name}"
    kwargs["name"] = name
    return kwargs


def _roxar_importer(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
    attributes: Optional[bool] = False,
):
    stype = stype.lower()
    valid_stypes = ["horizons", "zones", "faults", "clipboard"]

    if stype not in valid_stypes:
        raise ValueError(f"Invalid stype, only {valid_stypes} stypes is supported.")

    kwargs = _xyz_roxapi.import_xyz_roxapi(
        project, name, category, stype, realisation, attributes, False
    )

    kwargs["filesrc"] = f"RMS: {name} ({category})"
    return kwargs


def _dataframe_importer(
    dfr: pd.DataFrame,
    east: Optional[str] = "X",
    north: Optional[str] = "Y",
    zvalues: Optional[str] = "Z",
    zname: Optional[str] = "Z_TVDSS",
    attributes: Optional[dict] = None,
    **kwargs,
) -> Points:
    """Generic dataframe importer, returning kwargs."""

    zvalues = kwargs.get("tvdmsl", zvalues)

    # some checks
    for cname in (east, north, zvalues):
        if cname not in dfr.columns:
            raise KeyError(f"Column {cname} does not exist in input datadrame.")

    if attributes is not None:
        acnames = list(attributes.values())
        for acname in acnames:
            if acname not in dfr.columns:
                raise KeyError(
                    f"Attribute column {acname} does not exist in input datadrame."
                )

    newdfr = _ValidDataFrame(
        {"X_UTME": dfr[east], "Y_UTMN": dfr[north], zname: dfr[zvalues]}
    )
    newattrs = {}
    if attributes:
        for target, source in attributes.items():
            newdfr[target] = dfr[source].copy()
            newattrs[target] = pd.to_numeric(newdfr[target])  # try to get correct dtype
            newattrs[target] = str(newattrs[target].dtype)

    newdfr.dropna(inplace=True)

    kwargs = {}
    kwargs["values"] = newdfr
    kwargs["attributes"] = newattrs
    kwargs["zname"] = zname

    return kwargs


def _wells_importer(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    tops: Optional[bool] = True,
    incl_limit: Optional[float] = None,
    top_prefix: Optional[str] = "Top",
    zonelist: Optional[list] = None,
    zonelogname: Optional[str] = None,
    use_undef: Optional[bool] = False,
):
    """General function importing from wells"""

    if not wells:
        raise ValueError("No valid input wells")

    # wells in a scalar context is allowed if one well
    if not isinstance(wells, list):
        wells = [wells]

    # wells are just files which need to be imported, which is a bit more fragile
    if isinstance(wells[0], (str, pathlib.Path)):
        wells = [xtgeo.well_from_file(wll) for wll in wells]

    dflist = []
    for well in wells:
        if zonelogname is not None:
            well.zonelogname = zonelogname

        wp = well.get_zonation_points(
            tops=tops,
            incl_limit=incl_limit,
            top_prefix=top_prefix,
            zonelist=zonelist,
            use_undef=use_undef,
        )

        if wp is not None:
            dflist.append(wp)

    if dflist:
        dfr = pd.concat(dflist, ignore_index=True)
    else:
        return None

    attrs = {}
    for col in dfr.columns:
        if col == "Zone":
            attrs[col] = "int"
        elif col == "ZoneName":
            attrs[col] = "str"
        elif col == "WellName":
            attrs[col] = "str"
        else:
            attrs[col] = "float"

    kwargs = {}
    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["attributes"] = attrs
    kwargs["filesrc"] = "Derived from: Wells"
    return kwargs


def _wells_dfrac_importer(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    dlogname: str,
    dcodes: List[int],
    incl_limit: Optional[float] = 90,
    count_limit: Optional[int] = 3,
    zonelist: Optional[list] = None,  # TODO: list like?
    zonelogname: Optional[str] = None,
) -> Points:

    """General function, get fraction of discrete code(s) e.g. facies per zone."""

    if not wells:
        raise ValueError("No valid input wells")

    # wells in a scalar context is allowed if one well
    if not isinstance(wells, list):
        wells = [wells]

    # wells are just files which need to be imported, which is a bit more fragile
    if isinstance(wells[0], (str, pathlib.Path)):
        wells = [xtgeo.well_from_file(wll) for wll in wells]

    dflist = []
    for well in wells:
        wpf = well.get_fraction_per_zone(
            dlogname,
            dcodes,
            zonelist=zonelist,
            incl_limit=incl_limit,
            count_limit=count_limit,
            zonelogname=zonelogname,
        )

        if wpf is not None:
            dflist.append(wpf)

    if dflist:
        dfr = pd.concat(dflist, ignore_index=True)
    else:
        return None

    attrs = {}
    for col in dfr.columns[3:]:
        if col.lower() == "zone":
            attrs[col] = "int"
        elif col.lower() == "zonename":
            attrs[col] = "str"
        elif col.lower() == "wellname":
            attrs[col] = "str"
        else:
            attrs[col] = "float"

    kwargs = {}

    kwargs["values"] = _ValidDataFrame(dfr)
    kwargs["attributes"] = attrs
    kwargs["zname"] = "DFRAC"

    return kwargs


def _allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of Points and should be deleted once the deprecation period has expired,
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
                    "poi = xtgeo.points_from_file('some_file.xx') instead!",
                    DeprecationWarning,
                )
                fformat = kwargs.get("fformat", "guess")
                kwargs = _file_importer(args[0], fformat)

            elif isinstance(args[0], xtgeo.RegularSurface):
                warnings.warn(
                    "Initializing directly from RegularSurface is deprecated "
                    "and will be removed in xtgeo version 4.0. Use: "
                    "poi = xtgeo.points_from_surface(regsurf) instead",
                    DeprecationWarning,
                )
                zname = kwargs.get("zname", "Z_TVDSS")
                kwargs = _surface_importer(args[0], zname=zname)

            elif isinstance(args[0], (list, np.ndarray, pd.DataFrame, _ValidDataFrame)):
                # initialisation from an list-like object without 'values' keyword
                # should be still be possible, i.e. Points(some_list) is same as
                # Points(values=some_list)
                kwargs["values"] = args[0]

            else:
                raise TypeError("Input argument of unknown type: ", type(args[0]))

        return func(cls, **kwargs)

    return wrapper


# ======================================================================================
# FUNCTIONS as wrappers to class init + import
# ======================================================================================


def points_from_file(
    pfile: Union[str, pathlib.Path], fformat: Optional[str] = "guess"
) -> Points:
    """Make an instance of a Points object directly from file import.

    Supported formats are:

        * 'xyz' or 'poi' or 'pol': Simple XYZ format

        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)

        * 'rms_attr': RMS points formats with attributes (extra columns)

        * 'guess': Try to choose file format based on extension


    Args:
        pfile: Name of file or pathlib object.
        fformat: File format, xyz/pol/... Default is `guess` where file
            extension or file signature is parsed to guess the correct format.

    Example::

        import xtgeo
        mypoints = xtgeo.points_from_file('somefile.xyz')
    """
    return Points(**_file_importer(pfile, fformat=fformat))


def points_from_roxar(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
    attributes: Optional[bool] = False,
) -> Points:
    """Load a Points instance from Roxar RMS project.

    The import from the RMS project can be done either within the project
    or outside the project.

    Note also that horizon/zone/faults name and category must exists
    in advance, otherwise an Exception will be raised.

    Args:
        project: Name of project (as folder) if outside RMS, or just use the
            magic `project` word if within RMS.
        name: Name of points item
        category: For horizons/zones/faults: for example 'DL_depth'
            or use a folder notation on clipboard.
        stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard', etc!
        realisation: Realisation number, default is 0
        attributes: If True, attributes will be preserved (from RMS 11)

    Example::

        import xtgeo
        mysurf = xtgeo.points_from_roxar(project, 'TopAare', 'DepthPoints')
    """

    kwargs = _roxar_importer(
        project,
        name,
        category,
        stype,
        realisation,
        attributes,
    )

    return Points(**kwargs)


def points_from_surface(
    regsurf: RegularSurface,
    zname: Optional[str] = "Z_TVDSS",
    name: Optional[str] = "points",
) -> Points:
    """This makes an instance of a Points directly from a RegularSurface object.

    Each surface node will be stored as a X Y Z point.

    Args:
        regsurf: XTGeo RegularSurface() instance
        zname: Name of third column
        name: Name of instance

    .. versionadded:: 2.16
       Replaces the from_surface() method.
    """

    return Points(**_surface_importer(regsurf, zname=zname, name=name))


def points_from_dataframe(
    dfr: pd.DataFrame,
    east: Optional[str] = "X",
    north: Optional[str] = "Y",
    zvalues: Optional[str] = "Z",
    zname: Optional[str] = "Z_TVDSS",
    attributes: Optional[dict] = None,
    **kwargs,
) -> Points:
    """This makes an instance of a Points directly from a DataFrame object.

    Args:
        dfr: Pandas dataframe.
        east: Name of easting column in input dataframe.
        north: Name of northing column in input dataframe.
        zvalues: Name of depth or other values column in input dataframe.
        zname: Name of third column in resulting poinset.
        attributes: Additional metadata columns, on form {"IX": "I", ...};
            "IX" here is the name of the target column, and "I" is the name in the
            input dataframe (source).

    Note:
        One can also use dataframe input directly to instance creating, e.g.::

            points = xtgeo.Points(dfr)

        In that case the order of columns is assumed to be correct order in input as
        X Y Z, and there is less flexibility.

    Note:
        For backward compatibility with the deprecated ``from_dataframe()``, using the
        key ``tvdmsl`` instead of ``zvalues`` is possible.

    .. versionadded:: 2.16
       Replaces the :meth:`from_dataframe()` method.
    """
    zvalues = kwargs.get("tvdmsl", zvalues)
    newkwargs = _dataframe_importer(dfr, east, north, zvalues, zname, attributes)
    return Points(**newkwargs)


def points_from_wells(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    tops: Optional[bool] = True,
    incl_limit: Optional[float] = None,
    top_prefix: Optional[str] = "Top",
    zonelogname: Optional[str] = None,
    zonelist: Optional[list] = None,  # TODO: list like?
    use_undef: Optional[bool] = False,
) -> Points:

    """Get tops or zone points data from a list of wells.

    Args:
        wells: List of XTGeo well objects, a single XTGeo well or a list of well files.
            If a list of well files, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        tops: Get the tops if True (default), otherwise zone.
        incl_limit: Inclination limit for zones (thickness points)
        top_prefix: Prefix used for Tops.
        zonelogname: If None, the zonelogname in the well object will be applied. This
            option is particualr useful if one uses wells directly from files.
        zonelist: Which zone numbers to apply.
        use_undef: If True, then transition from UNDEF is also used.

    Returns:
        None if empty data, otherwise a Points() instance.

    Example::

            wells = ["w1.w", "w2.w"]
            points = xtgeo.points_from_wells(wells)

    Note:
        The deprecated method :py:meth:`~Points.from_wells` returns the number of
        wells that contribute with points. This is now implemented through the
        function `get_nwells()`. Hence the following code::

            nwells_applied = poi.from_wells(...)  # deprecated method
            # vs
            poi = xtgeo.points_from_wells(...)
            nwells_applied = poi.get_nwells()

    .. versionadded: 2.16


    """
    kwargs = _wells_importer(
        wells, tops, incl_limit, top_prefix, zonelist, zonelogname, use_undef
    )

    if kwargs is not None:
        return Points(**kwargs)

    return None


def points_from_wells_dfrac(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    dlogname: str,
    dcodes: List[int],
    incl_limit: Optional[float] = 90,
    count_limit: Optional[int] = 3,
    zonelist: Optional[list] = None,  # TODO: list like?
    zonelogname: Optional[str] = None,
) -> Points:

    """Get fraction of discrete code(s) e.g. facies per zone.

    Args:
        wells: List of XTGeo well objects, a single XTGeo well or a list of well files.
            If a list of file names, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        dlogname: Name of discrete log (e.g. Facies)
        dcodes: Code(s) to get fraction for, e.g. [3]
        incl_limit: Inclination limit for zones (thickness points)
        count_limit: Min. no of counts per segment for valid result
        zonelist: Which zone numbers to apply, default None means all.
        zonelogname: If None, the zonelogname property in the well object will be
            applied. This option is particualr useful if one uses wells directly from
            files.

    Returns:
        None if empty data, otherwise a Points() instance.

    Example::

            wells = ["w1.w", "w2.w"]
            points = xtgeo.points_from_wells_dfrac(
                    wells, dlogname="Facies", dcodes=[4], zonelogname="ZONELOG"
                )

    Note:
        The deprecated method :py:meth:`~Points.dfrac_from_wells` returns the number of
        wells that contribute with points. This is now implemented through the
        method `get_nwells()`. Hence the following code::

            nwells_applied = poi.dfrac_from_wells(...)  # deprecated method
            # vs
            poi = xtgeo.points_from_wells_dfrac(...)
            nwells_applied = poi.get_nwells()

    .. versionadded:: 2.16 Replaces :meth:`~Points.dfrac_from_wells`
    """

    kwargs = _wells_dfrac_importer(
        wells, dlogname, dcodes, incl_limit, count_limit, zonelist, zonelogname
    )
    return Points(**kwargs)


########################################################################################
# Points class
########################################################################################
class Points(XYZ):  # pylint: disable=too-many-public-methods, function-redefined
    """Class for a Points data in XTGeo.

    The Points class is a subclass of the :py:class:`~xtgeo.xyz._xyz.XYZ` abstract
    class, and the point set itself is a `pandas <http://pandas.pydata.org>`_
    dataframe object.

    For points, 3 float columns (X Y Z) are mandatory. In addition it is possible to
    have addiotional points attribute columns, and such attributes may be integer,
    strings or floats.

    The instance can be made either from file (then as classmethod), from another
    object or by a spesification, e.g. from file or a surface::

        xp1 = xtgeo.points_from_file('somefilename', fformat='xyz')
        # or
        regsurf = xtgeo.surface_from_file("somefile.gri")
        xp2 = xtgeo.points_from_surface(regsurf)

    You can also initialise points from list of tuples/lists in Python, where
    each tuple is a (X, Y, Z) coordinate::

        plist = [(234, 556, 12), (235, 559, 14), (255, 577, 12)]
        mypoints = Points(values=plist)

    The tuples can also contain point attributes which needs spesification via
    an attributes dictionary::

        plist = [
            (234, 556, 12, "Well1", 22),
            (235, 559, 14, "Well2", 44),
            (255, 577, 12, "Well3", 55)]
        attrs = {"WellName": "str", "ID", "int"}
        mypoints = Points(values=plist, attributes=attrs)

    And points can be initialised from a 2D numpy array or an existing dataframe::

        np2d = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]).reshape(4, 3)
        mypoints = Points(values=np2d)

        mydf = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        mypoints = Points(values=mydf)

    Similar as for lists, attributes are alse possible for numpy and dataframes.

    Default column names in the dataframe:

    * X_UTME: UTM X coordinate as self._xname
    * Y_UTMN: UTM Y coordinate as self._yname
    * Z_TVDSS: Z coordinate, often depth below TVD SS, but may also be
      something else! Use zname attribute to change name.

    Note:
        Attributes may have undefined entries. Pandas version 0.21 (which is applied
        for RMS version up to 12.0.x) do not support NaN values for Integers. The
        solution is store undefined values as large numbers, xtgeo.UNDEF_INT
        (2000000000) for integers and xtgeo.UNDEF (10e32) for float values.
        This will change from xtgeo version 3.0 where Pandas version 1 and above will
        be required, which in turn support will pandas.NA entries.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column, default is X_UTME.
        yname: Name of second (Y) mandatory column, default is Y_UTMN.
        zname: Name of third (Z) mandatory column, default is Z_TVDSS.
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
        name: Optional[str] = "points",
        attributes: Optional[dict] = None,
        filesrc: Optional[str] = None,
    ):
        """Initialisation of Points()."""
        super().__init__(xname, yname, zname, name, filesrc)

        self._df, self._filesrc = _xyz_io.initialize_by_values(
            values, self._zname, attributes, False
        )

        logger.info("Initiated Points")

    # ==================================================================================
    # Dunder methods
    # See also base class
    # ==================================================================================

    def __eq__(self, value):
        """Magic method for ==."""
        return self.dataframe[self.zname] == value

    def __gt__(self, value):
        return self.dataframe[self.zname] > value

    def __ge__(self, value):
        return self.dataframe[self.zname] >= value

    def __lt__(self, value):
        return self.dataframe[self.zname] < value

    def __le__(self, value):
        return self.dataframe[self.zname] <= value

    # ==================================================================================
    # Properties
    # See also base class
    # ==================================================================================

    @property
    def attributes(self) -> dict:
        """Returns a dictionary with attribute names and type, or None."""
        # this is not stored as state variable now, but dynamically stored in
        # all pandas columns after the 3 first.
        attrs = {}
        for col in self._df.columns[3:]:
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
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        # this may easily trigger inconsistiencies; need a a number of validations
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input df is not a DataFrame, but a {type(df)} instance.")
        if len(df.columns) < 3:
            raise ValidationError("Input dataframe has too few columns (need >= 3)")
        self._xname, self._yname, self._zname = df.columns[0:3]
        self._df = df.apply(deepcopy)  # see comment note in _xyz.py

    # ==================================================================================
    # Private methods
    # ==================================================================================

    def _reset(self, **kwargs):
        """Used in deprecated methods."""
        self._df = kwargs.get("values", self._df)
        self._xname = kwargs.get("xname", self._xname)
        self._yname = kwargs.get("yname", self._yname)
        self._zname = kwargs.get("zname", self._zname)
        self._filesrc = kwargs.get("filesrc", self._filesrc)

    def _random(self, nrandom=10):
        """Generate nrandom random points within the range 0..1.

        Args:
            nrandom (int): Number of random points (default 10)

        .. versionadded:: 2.3
        """
        self._df = pd.DataFrame(
            np.random.rand(nrandom, 3), columns=[self._xname, self._yname, self._zname]
        )

    # ==================================================================================
    # I/O methods
    # In general from_ methods are deprecated
    # ==================================================================================

    # from ... -------------------------------------------------------------------------

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

        self._reset(**_surface_importer(surf, zname=zname))

    @inherit_docstring(inherit_from=XYZ.from_file)
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_file() instead",
    )
    def from_file(self, pfile, fformat="xyz"):
        self._reset(**_file_importer(pfile, fformat))

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_roxar() instead.",
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
        kwargs = _roxar_importer(
            project,
            name,
            category,
            stype,
            realisation,
            attributes,
        )

        self._reset(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_dataframe(...) or a direct xtgeo.Points(...) "
        "initialisation instead.",
    )
    def from_dataframe(self, dfr, east="X", north="Y", tvdmsl="Z", attributes=None):
        """Import points/polygons from existing Pandas dataframe.

        Args:
            dfr (dataframe): Pandas dataframe.
            east (str): Name of easting column in input dataframe.
            north (str): Name of northing column in input dataframe.
            tvdmsl (str): Name of depth column in input dataframe.
            attributes (dict): Additional metadata columns, on form {"IX": "I", ...};
                "IX" here is the name of the target column, and "I" is the name in the
                input dataframe.

        .. versionadded:: 2.13
        .. deprecated:: 2.16 See xtgeo.points_from_dataframe()
        """
        kwargs = _dataframe_importer(
            dfr=dfr,
            east=east,
            north=north,
            zvalues=tvdmsl,
            zname="Z_TVDSS",
            attributes=attributes,
        )
        self._reset(**kwargs)

    @inherit_docstring(inherit_from=XYZ.from_list)
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use direct Points() initialisation instead",
    )
    def from_list(self, plist):

        kwargs = {}
        kwargs["values"], kwargs["filesrc"] = _xyz_io._from_list_like(
            plist, "Z_TVDSS", None, False
        )
        self._reset(**kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_wells() instead.",
    )
    def from_wells(
        self,
        wells,
        tops=True,
        incl_limit=None,
        top_prefix="Top",
        zonelist=None,
        use_undef=False,
    ):

        """Get tops or zone points data from a list of wells.

        Args:
            wells (list): List of XTGeo well objects
            tops (bool): Get the tops if True (default), otherwise zone
            incl_limit (float): Inclination limit for zones (thickness points)
            top_prefix (str): Prefix used for Tops
            zonelist (list-like): Which zone numbers to apply.
            use_undef (bool): If True, then transition from UNDEF within zonelog
                is also used.

        Returns:
            None if well list is empty; otherwise the number of wells actually applied.

        Raises:
            Todo

        .. deprecated:: 2.16
           Use classmethod :py:func:`points_from_wells()` instead
        """
        kwargs = _wells_importer(
            wells, tops, incl_limit, top_prefix, zonelist, None, use_undef
        )
        self._reset(**kwargs)
        return self.get_nwells()

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_wells_dfrac() instead.",
    )
    def dfrac_from_wells(
        self,
        wells,
        dlogname,
        dcodes,
        incl_limit=90,
        count_limit=3,
        zonelist=None,
        zonelogname=None,
    ):

        """Get fraction of discrete code(s) (e.g. facies) per zone.

        Args:
            wells (list): List of XTGeo well objects.
            dlogname (str): Name of discrete log (e.g. Facies).
            dcodes (list of int): Code(s) to get fraction for, e.g. [3].
            incl_limit (float): Inclination limit for zones (thickness points).
            count_limit (int): Min. no of counts per segment for valid result.
            zonelist (list of int): Which zones to compute for (default None
                means that all zones will be evaluated).
            zonelogname (str): Name of zonelog; if None than the
                well.zonelogname property will be applied.

        Returns:
            None if well list is empty; otherwise the number of wells.

        .. deprecated:: 2.16
           Use classmethod :py:func:`points_from_wells_dfrac()` instead.
        """

        kwargs = _wells_dfrac_importer(
            wells, dlogname, dcodes, incl_limit, count_limit, zonelist, zonelogname
        )
        self._reset(**kwargs)

    # to ... ---------------------------------------------------------------------------

    def to_file(
        self,
        pfile,
        fformat="xyz",
        attributes=True,
        pfilter=None,
        wcolumn=None,
        hcolumn=None,
        mdcolumn="M_MDEPTH",
        **kwargs,
    ):  # pylint: disable=redefined-builtin
        """Export Points to file.

        Args:
            pfile (str): Name of file
            fformat (str): File format xyz/poi/pol or rms_attr
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

        return _xyz_io.to_generic_file(
            self,
            pfile,
            fformat=fformat,
            attributes=attributes,
            pfilter=pfilter,
            wcolumn=wcolumn,
            hcolumn=hcolumn,
            mdcolumn=mdcolumn,
            **kwargs,
        )

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
        """Export (store) a Points item to a Roxar RMS project.

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
            raise ValueError(f"Invalid stype, only {valid_stypes} stypes is supported.")

        _xyz_roxapi.export_xyz_roxapi(
            self,
            project,
            name,
            category,
            stype.lower(),
            pfilter,
            realisation,
            attributes,
            False,
        )

    # ==================================================================================
    # Other methods
    # ==================================================================================

    def copy(self):
        """Returns a deep copy of an instance."""
        mycopy = self.__class__()
        mycopy._df = self._df.apply(deepcopy)
        mycopy._xname = self._xname
        mycopy._yname = self._yname
        mycopy._zname = self._zname
        mycopy._name = self._name

        mycopy._filesrc = "Derived from: copied instance"

        return mycopy

    @inherit_docstring(inherit_from=XYZ.delete_columns)
    def delete_columns(self, clist, strict=False):
        _xyz_oper.delete_columns(self, clist, strict, False)

    # ==================================================================================
    # Operations vs surfaces and possibly other
    # ==================================================================================

    def snap_surface(self, surf, activeonly=True):
        """Snap (transfer) the points Z values to a RegularSurface

        Args:
            surf (~xtgeo.surface.regular_surface.RegularSurface): Surface to snap to.
            activeonly (bool): If True (default), the points outside the defined surface
                will be removed. If False, these points will keep the original values.

        Returns:
            None (instance is updated inplace)

        Raises:
            ValueError: Input object of wrong data type, must be RegularSurface
            RuntimeError: Error code from C routine surf_get_zv_from_xyv is ...

        .. versionadded:: 2.1

        """
        _xyz_oper.snap_surface(self, surf, activeonly=activeonly)

    # ==================================================================================
    # Operations restricted to inside/outside polygons!
    # See base class!
    # ==================================================================================
    ...
