"""The XTGeo xyz.points module, which contains the Points class."""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo.common.exceptions import InvalidFileFormatError
from xtgeo.common.log import null_logger
from xtgeo.common.sys import inherit_docstring
from xtgeo.io._file import FileFormat, FileWrapper
from xtgeo.xyz import XYZ, _xyz_io, _xyz_oper, _xyz_roxapi

if TYPE_CHECKING:
    import io
    import pathlib

    from xtgeo.well import Well

logger = null_logger(__name__)


def _data_reader_factory(file_format: FileFormat):
    if file_format == FileFormat.XYZ:
        return _xyz_io.import_xyz
    if file_format == FileFormat.ZMAP_ASCII:
        return _xyz_io.import_zmap
    if file_format == FileFormat.RMS_ATTR:
        return _xyz_io.import_rms_attr

    extensions = FileFormat.extensions_string(
        [FileFormat.XYZ, FileFormat.ZMAP_ASCII, FileFormat.RMS_ATTR]
    )
    raise InvalidFileFormatError(
        f"File format {file_format} is invalid for type Points. "
        f"Supported formats are {extensions}."
    )


def _file_importer(
    points_file: str | pathlib.Path | io.BytesIO,
    fformat: str | None = None,
):
    """General function for points_from_file"""
    pfile = FileWrapper(points_file)
    fmt = pfile.fileformat(fformat)
    kwargs = _data_reader_factory(fmt)(pfile)
    kwargs["values"].dropna(inplace=True)
    kwargs["filesrc"] = pfile.name
    return kwargs


def _surface_importer(surf, zname="Z_TVDSS"):
    """General function for _read_surface()"""
    val = surf.values
    xc, yc = surf.get_xy_values()

    coord = []
    for vv in [xc, yc, val]:
        vv = np.ma.filled(vv.flatten(order="C"), fill_value=np.nan)
        vv = vv[~np.isnan(vv)]
        coord.append(vv)

    return {
        "values": pd.DataFrame(
            {
                "X_UTME": coord[0],
                "Y_UTMN": coord[1],
                zname: coord[2],
            }
        ),
        "zname": zname,
    }


def _roxar_importer(
    project,
    name: str,
    category: str,
    stype: str = "horizons",
    realisation: int = 0,
    attributes: bool | list[str] = False,
):
    return _xyz_roxapi.import_xyz_roxapi(
        project, name, category, stype, realisation, attributes, False
    )


def _wells_importer(
    wells: list[Well],
    tops: bool = True,
    incl_limit: float | None = None,
    top_prefix: str = "Top",
    zonelist: list | None = None,
    use_undef: bool = False,
) -> dict:
    """General function importing from wells"""
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

    dfr = pd.concat(dflist, ignore_index=True)

    attrs = {}
    for col in dfr.columns:
        col_lower = col.lower()
        if "float" in dfr[col].dtype.name:
            attrs[col] = "float"
        elif "int" in dfr[col].dtype.name:
            attrs[col] = "int"
        else:  # usually 'object'
            if col_lower == "zone":
                attrs[col] = "int"
            elif "name" in col_lower:
                attrs[col] = "str"
            else:
                attrs[col] = "float"  # fall-back

    return {"values": dfr, "attributes": attrs}


def _wells_dfrac_importer(
    wells: list[Well],
    dlogname: str,
    dcodes: list[int],
    incl_limit: float = 90,
    count_limit: int = 3,
    zonelist: list = None,
    zonelogname: str = None,
) -> dict:
    """General function, get fraction of discrete code(s) e.g. facies per zone."""

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

    dfr = pd.concat(dflist, ignore_index=True)

    attrs = {}
    for col in dfr.columns[3:]:
        col_lower = col.lower()
        if col_lower == "zone":
            attrs[col] = "int"
        elif col_lower == "zonename" or col_lower == "wellname":
            attrs[col] = "str"
        else:
            attrs[col] = "float"

    return {
        "values": dfr,
        "attributes": attrs,
        "zname": "DFRAC",
    }


def points_from_file(pfile: str | pathlib.Path, fformat: str | None = "guess"):
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
    project,
    name: str,
    category: str,
    stype: str = "horizons",
    realisation: int = 0,
    attributes: bool | list[str] = False,
):
    """Load a Points instance from Roxar RMS project.

    The import from the RMS project can be done either within the project
    or outside the project.

    Note also that horizon/zone/faults name and category must exists
    in advance, otherwise an Exception will be raised.

    Args:
        project: Name of project (as folder) if outside RMS, or just use the
            magic `project` word if within RMS.
        name (str): Name of points item, or name of well pick set if
            well picks.
        category: For horizons/zones/faults: for example 'DL_depth'
            or use a folder notation on clipboard/general2d_data.
            For well picks it is the well pick type: 'horizon' or 'fault'.
        stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard',
            'general2d_data', 'faults' or 'well_picks'
        realisation: Realisation number, default is 0
        attributes (bool): Bool or list with attribute names to collect.
            If True, all attributes are collected.

    Example::

        # inside RMS:
        import xtgeo
        mypoints = xtgeo.points_from_roxar(project, 'TopEtive', 'DP_seismic')

    .. versionadded:: 2.19 general2d_data support is added
    """

    return Points(
        **_roxar_importer(
            project,
            name,
            category,
            stype,
            realisation,
            attributes,
        )
    )


def points_from_surface(
    regular_surface,
    zname: str = "Z_TVDSS",
):
    """This makes an instance of a Points directly from a RegularSurface object.

    Each surface node will be stored as a X Y Z point.

    Args:
        regular_surface: XTGeo RegularSurface() instance
        zname: Name of third column

    .. versionadded:: 2.16
       Replaces the from_surface() method.
    """

    return Points(**_surface_importer(regular_surface, zname=zname))


def points_from_wells(
    wells: list[Well],
    tops: bool = True,
    incl_limit: float | None = None,
    top_prefix: str = "Top",
    zonelist: list | None = None,
    use_undef: bool = False,
):
    """Get tops or zone points data from a list of wells.

    Args:
        wells: List of XTGeo well objects.
            If a list of well files, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        tops: Get the tops if True (default), otherwise zone.
        incl_limit: Inclination limit for zones (thickness points)
        top_prefix: Prefix used for Tops.
        zonelist: Which zone numbers to apply, None means all.
        use_undef: If True, then transition from UNDEF is also used.

    Returns:
        None if empty data, otherwise a Points() instance.

    Example::

            wells = [xtgeo.well_from_file("w1.w"), xtgeo.well_from_file("w2.w")]
            points = xtgeo.points_from_wells(wells)
    """
    return Points(
        **_wells_importer(wells, tops, incl_limit, top_prefix, zonelist, use_undef)
    )


def points_from_wells_dfrac(
    wells: list[Well],
    dlogname: str,
    dcodes: list[int],
    incl_limit: float = 90,
    count_limit: int = 3,
    zonelist: list | None = None,
    zonelogname: str | None = None,
):
    """Get fraction of discrete code(s) e.g. facies per zone.

    Args:
        wells: List of XTGeo well objects.
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

            wells = [xtgeo.well_from_file("w1.w"), xtgeo.well_from_file("w2.w")]
            points = xtgeo.points_from_wells_dfrac(
                    wells, dlogname="Facies", dcodes=[4], zonelogname="ZONELOG"
                )
    """
    return Points(
        **_wells_dfrac_importer(
            wells, dlogname, dcodes, incl_limit, count_limit, zonelist, zonelogname
        )
    )


class Points(XYZ):
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

        >>> mypoints1 = Points(values=[(1,1,1), (2,2,2), (3,3,3)])
        >>> mypoints2 = Points(
        ...     values=pd.DataFrame(
        ...          [[1, 2, 3], [1, 2, 3], [1, 2, 3]],
        ...          columns=["X_UTME", "Y_UTMN", "Z_TVDSS"]
        ...     )
        ... )


    Similar as for lists, attributes are alse possible for numpy and dataframes.

    Default column names in the dataframe:

    * X_UTME: UTM X coordinate  as self._xname
    * Y_UTMN: UTM Y coordinate  as self._yname
    * Z_TVDSS: Z coordinate, often depth below TVD SS, but may also be
      something else! Use zname attribute to change name.

    Note:
        Attributes may have undefined entries. Pandas version 0.21 (which is applied
        for RMS version up to 12.0.x) do not support NaN values for Integers. The
        solution is store undefined values as large numbers, xtgeo.UNDEF_INT
        (2000000000) for integers and xtgeo.UNDEF (10e32) for float values.
        This will change from xtgeo version 3.x where Pandas version 1 and
        above will be required, which in turn support will pandas.NA
        entries.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column, default is X_UTME.
        yname: Name of second (Y) mandatory column, default is Y_UTMN.
        zname: Name of third (Z) mandatory column, default is Z_TVDSS.
        attributes: A dictionary for attribute columns as 'name: type', e.g.
            {"WellName": "str", "IX": "int"}. This is applied when values are input
            and is to name and type the extra attribute columns in a point set.
    """

    def __init__(
        self,
        values: list | np.ndarray | pd.DataFrame = None,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
        attributes: dict | None = None,
        filesrc: str = None,
    ):
        """Initialisation of Points()."""
        super().__init__(xname, yname, zname)
        if values is None:
            values = []

        self._attrs = attributes if attributes is not None else {}
        self._filesrc = filesrc

        if not isinstance(values, pd.DataFrame):
            self._df = _xyz_io._from_list_like(values, self._zname, attributes, False)
        else:
            self._df: pd.DataFrame = values
            self._dataframe_consistency_check()

    def _dataframe_consistency_check(self):
        dataframe = self.get_dataframe(copy=False)
        if self.xname not in dataframe:
            raise ValueError(
                f"xname={self.xname} is not a column "
                f"of dataframe {dataframe.columns}"
            )
        if self.yname not in dataframe:
            raise ValueError(
                f"yname={self.yname} is not a column "
                f"of dataframe {dataframe.columns}"
            )
        if self.zname not in dataframe:
            raise ValueError(
                f"zname={self.zname} is not a column "
                f"of dataframe {dataframe.columns}"
            )
        for attr in self._attrs:
            if attr not in dataframe:
                raise ValueError(
                    f"Attribute {attr} is not a column "
                    f"of dataframe {dataframe.columns}"
                )

    def __repr__(self):
        # should be able to newobject = eval(repr(thisobject))
        return f"{self.__class__.__name__} (filesrc={self._filesrc!r}, ID={id(self)})"

    def __str__(self):
        """User friendly print."""
        return self.describe(flush=False)

    def __eq__(self, value):
        """Magic method for ==."""
        return self.get_dataframe(copy=False)[self.zname] == value

    def __gt__(self, value):
        return self.get_dataframe(copy=False)[self.zname] > value

    def __ge__(self, value):
        return self.get_dataframe(copy=False)[self.zname] >= value

    def __lt__(self, value):
        return self.get_dataframe(copy=False)[self.zname] < value

    def __le__(self, value):
        return self.get_dataframe(copy=False)[self.zname] <= value

    # ----------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        warnings.warn(
            "Direct access to the dataframe property in Points class will be "
            "deprecated in xtgeo 5.0. Use `get_dataframe()` instead.",
            PendingDeprecationWarning,
        )
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        warnings.warn(
            "Direct access to the dataframe property in Points class will be "
            "deprecated in xtgeo 5.0. Use `set_dataframe(df)` instead.",
            PendingDeprecationWarning,
        )
        self.set_dataframe(df)

    def get_dataframe(self, copy: bool = True) -> pd.DataFrame:
        """Returns the Pandas dataframe object.

        Args:
            copy: If True (default) the a deep copy is returned; otherwise a view
                which may be faster in some cases)

        .. versionchanged:: 3.7 Add keyword `copy`, defaulted to True

        """
        return self._df.copy() if copy else self._df

    def set_dataframe(self, df):
        self._df = df.apply(deepcopy)

    def _random(self, nrandom=10):
        """Generate nrandom random points within the range 0..1

        Args:
            nrandom (int): Number of random points (default 10)

        .. versionadded:: 2.3
        """

        # currently a non-published method

        self._df = pd.DataFrame(
            np.random.rand(nrandom, 3), columns=[self._xname, self._yname, self._zname]
        )

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
    ):
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
        return _xyz_io.to_file(
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
    ):  # pragma: no cover
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
            name (str): Name of points item, or name of well pick set if
                well picks.
            category (str): For horizons/zones/faults: for example 'DL_depth'
                or use a folder notation on clipboard/general2d_data.
                For well picks it is the well pick type: "horizon" or "fault".
            pfilter (dict): Filter on e.g. top name(s) with keys TopName
                or ZoneName as {'TopName': ['Top1', 'Top2']}
            stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard',
                'general2d_data', 'faults' or 'well_picks'
            realisation (int): Realisation number, default is 0
            attributes (bool): If True, attributes will be preserved (from RMS 11)

        Returns:
            Object instance updated

        Raises:
            ValueError: Various types of invalid inputs.
            NotImplementedError: Not supported in this ROXAPI version

        .. versionadded:: 2.19 general2d_data support is added
        """

        _xyz_roxapi.export_xyz_roxapi(
            self,
            project,
            name,
            category,
            stype,
            pfilter,
            realisation,
            attributes,
        )

    def copy(self):
        """Returns a deep copy of an instance."""
        mycopy = self.__class__()
        mycopy._df = self._df.apply(deepcopy)
        mycopy._xname = self._xname
        mycopy._yname = self._yname
        mycopy._zname = self._zname

        return mycopy

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

    @inherit_docstring(inherit_from=XYZ.get_boundary)
    def get_boundary(self):
        return super().get_boundary()
