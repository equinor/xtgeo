# -*- coding: utf-8 -*-
"""The XTGeo xyz.points module, which contains the Points class."""
import functools
import io
import pathlib
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, TypeVar, Union

import deprecation
import numpy as np
import pandas as pd
import xtgeo
import xtgeo.common.sys as xtgeosys
from xtgeo.common import inherit_docstring
from xtgeo.xyz import _xyz_io, _xyz_roxapi

from . import _xyz_oper
from ._xyz import XYZ

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


def _data_reader_factory(file_format):
    if file_format == "xyz":
        return _xyz_io.import_xyz
    if file_format == "zmap_ascii":
        return _xyz_io.import_zmap
    if file_format == "rms_attr":
        return _xyz_io.import_rms_attr
    raise ValueError(f"Unknown file format {file_format}")


def _read_file_worker(
    pfile: Union[str, pathlib.Path, io.BytesIO],
    fformat: Optional[str] = None,
):
    """General function for classmethod _read_file and (deprecated) method from_file.

    See _read_file in class for args.
    """
    pfile = xtgeo._XTGeoFile(pfile)
    if fformat is None or fformat == "guess":
        fformat = pfile.detect_fformat()
    else:
        fformat = pfile.generic_format_by_proposal(fformat)  # default
    kwargs = _data_reader_factory(fformat)(pfile, is_polygons=False)

    kwargs["filesrc"] = pfile.name

    return kwargs


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
                    "poi = xtgeo.points_from_file('some_file.xx') instead!",
                    DeprecationWarning,
                )
                pfile = args[0]
                fformat = kwargs.get("fformat", "guess")
                pfile = xtgeosys._XTGeoFile(pfile)
                if fformat is None or fformat == "guess":
                    fformat = pfile.detect_fformat()
                else:
                    fformat = pfile.generic_format_by_proposal(fformat)  # default
                kwargs = _data_reader_factory(fformat)(pfile, is_polygons=False)

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


# ======================================================================================
# FUNCTIONS as wrappers to class init + import
# ======================================================================================


def points_from_file(
    pfile: Union[str, pathlib.Path], fformat: Optional[str] = "guess"
) -> "Points":
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
    kwargs = Points._read_file(pfile, fformat=fformat)
    return Points(**kwargs)


def points_from_roxar(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
    attributes: Optional[bool] = False,
) -> "Points":
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
    args = Points._read_roxar(
        project,
        name,
        category,
        stype,
        realisation,
        attributes,
    )

    return Points(**args)


RegularSurface = TypeVar("RegularSurface")


def points_from_surface(
    regsurf: RegularSurface,
    zname: Optional[str] = "Z_TVDSS",
    name: Optional[str] = "unknown",
) -> "Points":
    """This makes an instance of a Points directly from a RegularSurface object.

    Each surface node will be stored as a X Y Z point.

    Args:
        regsurf: XTGeo RegularSurface() instance
        zname: Name of third column
        name: Name of instance

    .. versionadded:: 2.16
       Replaces the from_surface() method.
    """

    return Points._read_surface(regsurf, zname=zname, name=name)


def points_from_dataframe(
    dfr: pd.DataFrame,
    east: Optional[str] = "X",
    north: Optional[str] = "Y",
    zvalues: Optional[str] = "Z",
    zname: Optional[str] = "Z_TVDSS",
    attributes: Optional[dict] = None,
    **kwargs,
) -> "Points":
    """This makes an instance of a Points directly from a DataFrame object.

    Args:
        dfr: Pandas dataframe.
        east: Name of easting column in input dataframe.
        north: Name of northing column in input dataframe.
        zvalues: Name of depth or other values column in input dataframe.
        zname: Name of column in resulting poinset.
        attributes: Additional metadata columns, on form {"IX": "I", ...};
            "IX" here is the name of the target column, and "I" is the name in the
            input dataframe (source).

    Note:
        One can also use dataframe input directly to instance creating, e.g.::

            points = xtgeo.Points(dfr)

        In that case the order of columns is assumed to be correct in input as X Y Z. In
        contrast, this method uses the explicit names of the columns as to establish
        ordering. This means that a dataframe with 'unconventional ordering' may be
        applied as input here!

    Note:
        For backward compatibility with the deprecated ``from_dataframe()``, using the
        key ``tvdmsl`` instead of ``zvalues`` is possible.

    .. versionadded:: 2.16
       Replaces the :meth:`from_dataframe()` method.
    """

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

    newdfr = pd.DataFrame(
        {"X_UTME": dfr[east], "Y_UTMN": dfr[north], zname: dfr[zvalues]}
    )
    newattrs = {}
    if attributes:
        for target, source in attributes.items():
            newdfr[target] = dfr[source].copy()
            newattrs[target] = pd.to_numeric(newdfr[target])  # try to get correct dtype
            newattrs[target] = str(newattrs[target].dtype)

    newdfr.dropna(inplace=True)

    return Points(dataframe=newdfr, attributes=newattrs)


def points_from_wells(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    tops: Optional[bool] = True,
    incl_limit: Optional[float] = None,
    top_prefix: Optional[str] = "Top",
    zonelogname: Optional[str] = None,
    zonelist: Optional[list] = None,  # TODO: list like?
    use_undef: Optional[bool] = False,
) -> "Points":

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
        property `nwells`. Hence the following code::

            nwells_applied = poi.from_wells(...)  # deprecated method
            # vs
            poi = xtgeo.points_from_wells(...)
            nwells_applied = poi.nwells

    .. versionadded: 2.16


    """

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

    poi = Points(dataframe=dfr, attributes=attrs)
    poi._nwells = len(dflist)
    return poi


def points_from_wells_dfrac(
    wells: Union[xtgeo.Well, List[xtgeo.Well], List[Union[str, pathlib.Path]]],
    dlogname: str,
    dcodes: List[int],
    incl_limit: Optional[float] = 90,
    count_limit: Optional[int] = 3,
    zonelist: Optional[list] = None,  # TODO: list like?
    zonelogname: Optional[str] = None,
) -> "Points":

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
        property `nwells`. Hence the following code::

            nwells_applied = poi.dfrac_from_wells(...)  # deprecated method
            # vs
            poi = xtgeo.points_from_wells_dfrac(...)
            nwells_applied = poi.nwells

    .. versionadded:: 2.16 Replaces :meth:`~Points.dfrac_from_wells`
    """

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

    poi = Points(dataframe=dfr, attributes=attrs)
    poi.zname = "DFRAC"  # name of third column
    poi._nwells = len(dflist)
    return poi


class Points(XYZ):  # pylint: disable=too-many-public-methods
    """Class for a Points data.

    The Points class is a subclass of the :py:class:`~xtgeo.xyz._xyz.XYZ` class,
    and the point set itself is a `pandas <http://pandas.pydata.org>`_
    dataframe object.

    For points, 3 float columns (X Y Z) are mandatory. In addition it is possible to
    have points attribute columns, and such attributes may be integer, strings
    or floats.

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
        attrs = {"Wellname": "str", "ID", "int"}
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


    Args:
        *args: See :py:class:`~xtgeo.xyz._xyz.XYZ` class.
        **kargs: See :py:class:`~xtgeo.xyz._xyz.XYZ` class.
    """

    @inherit_docstring(inherit_from=XYZ.__init__)
    def __init__(self, *args, **kwargs):
        """Initialisation for points for Points()."""
        super().__init__(*args, **kwargs)

        logger.info("Initiated Points")

    def __repr__(self):
        # should be able to newobject = eval(repr(thisobject))
        myrp = "{0.__class__.__name__} (filesrc={0._filesrc!r}, " "ID={1})".format(
            self, id(self)
        )
        return myrp

    def __str__(self):
        """User friendly print."""
        return self.describe(flush=False)

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

    # ----------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------
    @classmethod
    def _read_file(
        cls,
        pfile: Union[str, pathlib.Path, io.BytesIO],
        fformat: Optional[str] = None,
    ):
        """Import Points or Polygons from a file, see _read_file_worker.

        Supported import formats (fformat):
        * 'xyz' or 'poi' or 'pol': Simple XYZ format
        * 'zmap': ZMAP line format as exported from RMS (e.g. fault lines)
        * 'rms_attr': RMS points formats with attributes (extra columns)
        * 'guess': Try to choose file format based on extension

        Args:
            pfile: File-like or memory stream instance.
            fformat (str): File format, None/guess/xyz/pol/poi...

        Example::
            >>> myxyz = _XYZ._read_file('myfile.x')

        Raises:
            OSError: if file is not present or wrong permissions.
        """
        kwargs = _read_file_worker(pfile, fformat)
        return cls(**kwargs)

    def _reset(self, **kwargs):
        self._df = kwargs.get("_dataframe", self._df)
        self._attrs = kwargs.get("attributes", self._attrs)
        self.xname = kwargs.get("xname", self._xname)
        self.yname = kwargs.get("yname", self._yname)
        self.zname = kwargs.get("zname", self._zname)
        self.pname = kwargs.get("pname", self._pname)

    @classmethod
    def _read_roxar(
        cls,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
        attributes=False,
    ):
        """Classmethod to load a points item from a Roxar RMS project.

        # The import from the RMS project can be done either within the project
        # or outside the project.

        # Use::

        #   import xtgeo
        #   mysurf = xtgeo.points_from_roxar(project, 'TopAare', 'DepthPolys')

        # Note also that horizon/zone/faults name and category must exists
        # in advance, otherwise an Exception will be raised.

        # Args:
        #     project (str or special): Name of project (as folder) if
        #         outside RMS, og just use the magic project word if within RMS.
        #     name (str): Name of polygons item
        #     category (str): For horizons/zones/faults: for example 'DL_depth'
        #         or use a folder notation on clipboard.

        #     stype (str): RMS folder type, 'horizons' (default) or 'zones' etc!
        #     realisation (int): Realisation number, default is 0
        #     attributes (bool): If True, attributes will be preserved (from RMS 11)
        #     is_polygons (bool): True if Polygons

        # Returns:
        #     Object instance updated

        # Raises:
        #     ValueError: Various types of invalid inputs.

        """
        stype = stype.lower()
        valid_stypes = ["horizons", "zones", "faults", "clipboard"]

        if stype not in valid_stypes:
            raise ValueError(f"Invalid stype, only {valid_stypes} stypes is supported.")

        kwargs = _xyz_roxapi.import_xyz_roxapi(
            project, name, category, stype, realisation, attributes, False
        )

        kwargs["filesrc"] = f"RMS: {name} ({category})"
        return cls(**kwargs)

    def _random(self, nrandom=10):
        """Generate nrandom random points within the range 0..1.

        Args:
            nrandom (int): Number of random points (default 10)

        .. versionadded:: 2.3
        """
        self._df = pd.DataFrame(
            np.random.rand(nrandom, 3), columns=[self._xname, self._yname, self._zname]
        )

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

    @inherit_docstring(inherit_from=XYZ.from_file)
    def from_file(self, pfile, fformat="xyz", **kwargs):
        super().from_file(pfile, fformat=fformat, **kwargs)

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_file() or xtgeo.polygons_from_file() instead",
    )
    def from_file(self, pfile, fformat="xyz"):
        newkwargs = _read_file_worker(pfile, fformat, False)
        self._reset(**newkwargs)

    @inherit_docstring(inherit_from=XYZ.from_roxar)
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
        super().from_roxar(
            project,
            name,
            category,
            stype="horizons",
            realisation=0,
            attributes=False,
        )

    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.points_from_dataframe() or a direct xtgeo.Points() "
        "initialisation instead, where 'values' a dataframe",
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
        if not all(item in dfr.columns for item in (east, north, tvdmsl)):
            raise ValueError("One or more column names are not correct")

        if attributes and not all(item in dfr.columns for item in attributes.values()):
            raise ValueError("One or more attribute column names are not correct")

        input_ = OrderedDict()
        input_["X_UTME"] = dfr[east]
        input_["Y_UTMN"] = dfr[north]
        input_["Z_TVDSS"] = dfr[tvdmsl]

        if attributes:
            for target, source in attributes.items():
                input_[target] = dfr[source]

        self._df = pd.DataFrame(input_)
        self._filesrc = "DataFrame input"

        self._df.dropna(inplace=True)

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
        return super().to_file(
            pfile,
            fformat=fformat,
            attributes=attributes,
            pfilter=pfilter,
            wcolumn=wcolumn,
            hcolumn=hcolumn,
            mdcolumn=mdcolumn,
            **kwargs,
        )

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
            use_undef (bool): If True, then transition from UNDEF is also
                used.

        Returns:
            None if well list is empty; otherwise the number of wells.

        Raises:
            Todo

        .. deprecated:: 2.16
           Use classmethod :py:func:`points_from_wells()` instead
        """

        if not wells:
            return None

        dflist = []
        for well in wells:
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
            self._df = pd.concat(dflist, ignore_index=True)
        else:
            return None

        for col in self._df.columns:
            if col == "Zone":
                self._attrs[col] = "int"
            elif col == "ZoneName":
                self._attrs[col] = "str"
            elif col == "WellName":
                self._attrs[col] = "str"
            else:
                self._attrs[col] = "float"

        return len(dflist)

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

        if not wells:
            return None

        dflist = []  # will be a list of pandas dataframes
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
            self._df = pd.concat(dflist, ignore_index=True)
            self.zname = "DFRAC"  # name of third column
        else:
            return None

        for col in self._df.columns[3:]:
            if col.lower() == "zone":
                self._attrs[col] = "int"
            elif col.lower() == "zonename":
                self._attrs[col] = "str"
            elif col.lower() == "wellname":
                self._attrs[col] = "str"
            else:
                self._attrs[col] = "float"

        return len(dflist)

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
    # Operations restricted to inside/outside polygons
    # ==================================================================================

    def operation_polygons(self, poly, value, opname="add", inside=True, where=True):
        """A generic function for doing points operations restricted to inside
        or outside polygon(s).

        Args:
            poly (Points): A XTGeo Points instance
            value(float or str): Value to add, subtract etc. If 'poly'
                then use the avg Z value from each polygon.
            opname (str): Name of operation... 'add', 'sub', etc
            inside (bool): If True do operation inside Points; else outside.
            where (bool): A logical filter (current not implemented)

        Examples::

            # assume a point set where you want eliminate number inside
            # Points, given that the points Z value inside this polygon is
            # larger than 1700:
            poi = Points(POINTSET2)
            pol = Points(POLSET2)

            poi.operation_polygons(pol, 0, opname='eli', inside=True)



        """

        _xyz_oper.operation_polygons(
            self, poly, value, opname=opname, inside=inside, where=where
        )

    # shortforms
    def add_inside(self, poly, value, where=True):
        """Add a value (scalar) inside Points (see `operation_polygons`)"""
        self.operation_polygons(poly, value, opname="add", inside=True, where=where)

    def add_outside(self, poly, value, where=True):
        """Add a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="add", inside=False, where=where)

    def sub_inside(self, poly, value, where=True):
        """Subtract a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="sub", inside=True, where=where)

    def sub_outside(self, poly, value, where=True):
        """Subtract a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="sub", inside=False, where=where)

    def mul_inside(self, poly, value, where=True):
        """Multiply a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="mul", inside=True, where=where)

    def mul_outside(self, poly, value, where=True):
        """Multiply a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="mul", inside=False, where=where)

    def div_inside(self, poly, value, where=True):
        """Divide a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="div", inside=True, where=where)

    def div_outside(self, poly, value, where=True):
        """Divide a value (scalar) outside Points (value 0.0 will give
        result 0)"""
        self.operation_polygons(poly, value, opname="div", inside=False, where=where)

    def set_inside(self, poly, value, where=True):
        """Set a value (scalar) inside Points"""
        self.operation_polygons(poly, value, opname="set", inside=True, where=where)

    def set_outside(self, poly, value, where=True):
        """Set a value (scalar) outside Points"""
        self.operation_polygons(poly, value, opname="set", inside=False, where=where)

    def eli_inside(self, poly, where=True):
        """Eliminate current map values inside Points"""
        self.operation_polygons(poly, 0, opname="eli", inside=True, where=where)

    def eli_outside(self, poly, where=True):
        """Eliminate current map values outside Points"""
        self.operation_polygons(poly, 0, opname="eli", inside=False, where=where)
