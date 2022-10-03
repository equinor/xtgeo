"""XTGeo xyz.polygons module, which contains the Polygons class."""

# For polygons, the order of the points sequence is important. In
# addition, a Polygons dataframe _must_ have a INT column called 'POLY_ID'
# which identifies each polygon piece.
import functools
import io
import pathlib
import warnings
from copy import deepcopy
from typing import Any, List, Optional, Union

import deprecation
import numpy as np
import pandas as pd
import shapely.geometry as sg
import xtgeo
from xtgeo.common import inherit_docstring
from xtgeo.xyz import _xyz_io, _xyz_roxapi

from . import _xyz_oper
from ._xyz import XYZ
from ._xyz_io import _convert_idbased_xyz

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.functionlogger(__name__)


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
    kwargs = _data_reader_factory(fformat)(pfile)

    if "POLY_ID" not in kwargs["values"].columns:
        kwargs["values"]["POLY_ID"] = (
            kwargs["values"].isnull().all(axis=1).cumsum().dropna()
        )
        kwargs["values"].dropna(axis=0, inplace=True)
        kwargs["values"].reset_index(inplace=True, drop=True)
    kwargs["name"] = "poly"
    return kwargs


def _roxar_importer(
    project: Union[str, Any],
    name: str,
    category: str,
    stype: Optional[str] = "horizons",
    realisation: Optional[int] = 0,
):  # pragma: no cover
    kwargs = _xyz_roxapi.import_xyz_roxapi(
        project, name, category, stype, realisation, None, True
    )

    kwargs["name"] = "poly"
    return kwargs


def _wells_importer(
    wells: List[xtgeo.Well],
    zone: Optional[int] = None,
    resample: Optional[int] = 1,
):
    """Get line segments from a list of wells and a single zone number.

    A future extension is that zone could be a list of zone numbers and/or mechanisms
    to retrieve well segments by other measures, e.g. >= depth.
    """

    dflist = []
    maxid = 0
    for well in wells:
        wp = well.get_zone_interval(zone, resample=resample)
        if wp is not None:
            wp["WellName"] = well.name
            # as well segments may have overlapping POLY_ID:
            wp["POLY_ID"] += maxid
            maxid = wp["POLY_ID"].max() + 1
            dflist.append(wp)

    if not dflist:
        return {}
    dfr = pd.concat(dflist, ignore_index=True)
    dfr.reset_index(inplace=True, drop=True)
    return {
        "values": dfr,
        "attributes": {"WellName": "str"},
    }


def polygons_from_file(
    pfile: Union[str, pathlib.Path], fformat: Optional[str] = "guess"
):
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
):  # pragma: no cover
    """Load a Polygons instance from Roxar RMS project.

    Note also that horizon/zone/faults name and category must exists
    in advance, otherwise an Exception will be raised.

    Args:
        project: Name of project (as folder) if outside RMS, or just use the magic
            `project` word if within RMS.
        name: Name of polygons item
        category: For horizons/zones/faults: for example 'DL_depth'
            or use a folder notation on clipboard/general2d_data.
        stype: RMS folder type, 'horizons' (default), 'zones', 'clipboard',
            'faults', 'general2d_data'
        realisation: Realisation number, default is 0

    Example::

        import xtgeo
        mysurf = xtgeo.polygons_from_roxar(project, 'TopAare', 'DepthPolys')

    .. versionadded:: 2.19 general2d_data support is added
    """

    return Polygons(
        **_roxar_importer(
            project,
            name,
            category,
            stype,
            realisation,
        )
    )


def polygons_from_wells(
    wells: List[xtgeo.Well],
    zone: Optional[int] = 1,
    resample: Optional[int] = 1,
):

    """Get polygons from wells and a single zone number.

    Args:
        wells: List of XTGeo well objects, a single XTGeo well or a list of well files.
            If a list of well files, the routine will try to load well based on file
            signature and/or extension, but only default settings are applied. Hence
            this is less flexible and more fragile.
        zone: The zone number to extract the linepiece from
        resample: If given, resample every N'th sample to make
            polylines smaller in terms of bits and bytes.
            1 = No resampling, which means just use well sampling (which can be rather
            dense; typically 15 cm).


    Returns:
        None if empty data, otherwise a Polygons() instance.

    Example::

        wells = ["w1.w", "w2.w"]
        points = xtgeo.polygons_from_wells(wells, zone=2)

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
    return Polygons(**_wells_importer(wells, zone, resample))


def _allow_deprecated_init(func):
    # This decorator is here to maintain backwards compatibility in the construction
    # of Polygons and should be deleted once the deprecation period has expired,
    # the construction will then follow the new pattern.
    # Introduced post xtgeo version 2.15
    @functools.wraps(func)
    def wrapper(cls, *args, **kwargs):
        # Checking if we are doing an initialization from file or surface and raise a
        # deprecation warning if we are.
        if len(args) == 1 and isinstance(args[0], (str, pathlib.Path)):
            warnings.warn(
                "Initializing directly from file name is deprecated and will be "
                "removed in xtgeo version 4.0. Use: "
                "pol = xtgeo.polygons_from_file('some_file.xx') instead!",
                DeprecationWarning,
            )
            fformat = kwargs.get("fformat", "guess")
            return func(cls, **_file_importer(args[0], fformat))

        return func(cls, *args, **kwargs)

    return wrapper


class Polygons(XYZ):  # pylint: disable=too-many-public-methods
    """Class for a Polygons object (connected points) in the XTGeo framework.

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

    Each Polygons instance can also a name (through the name attribute).
    Default is 'poly'. E.g. if a well fence, it is logical to name the
    instance to be the same as the well name.

    Args:
        values: Provide input values on various forms (list-like or dataframe).
        xname: Name of first (X) mandatory column, default is X_UTME.
        yname: Name of second (Y) mandatory column, default is Y_UTMN.
        zname: Name of third (Z) mandatory column, default is Z_TVDSS.
        pname: Name of forth (P) mandatory enumerating column, default is POLY_ID.
        hname: Name of cumulative horizontal length, defaults to "H_CUMLEN" if
            in dataframe otherwise None.
        dhname: Name of delta horizontal length, defaults to "H_DELTALEN" if in
            dataframe otherwise None.
        tname: Name of cumulative total length, defaults to "T_CUMLEN" if in
            dataframe otherwise None.
        dtname: Name of delta total length, defaults to "T_DELTALEN" if in
            dataframe otherwise None.
        attributes: A dictionary for attribute columns as 'name: type', e.g.
            {"WellName": "str", "IX": "int"}. This is applied when values are input
            and is to name and type the extra attribute columns in a point set.
    """

    @_allow_deprecated_init
    def __init__(
        self,
        values: Union[list, np.ndarray, pd.DataFrame] = None,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
        pname: str = "POLY_ID",
        hname: str = "H_CUMLEN",
        dhname: str = "H_DELTALEN",
        tname: str = "T_CUMLEN",
        dtname: str = "T_DELTALEN",
        name: str = "poly",
        attributes: Optional[dict] = None,
        # from legacy initialization, remove in 4.0, undocumented by purpose:
        fformat: str = "guess",
    ):
        super().__init__(xname, yname, zname)

        if values is None:
            values = []

        logger.info("Legacy fformat key with value %s shall be removed in 4.0", fformat)

        self._reset(
            values=values,
            xname=xname,
            yname=yname,
            zname=zname,
            pname=pname,
            hname=hname,
            dhname=dhname,
            tname=tname,
            dtname=dtname,
            name=name,
            attributes=attributes,
        )

    def _reset(
        self,
        values: Union[list, np.ndarray, pd.DataFrame],
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
        pname: str = "POLY_ID",
        hname: str = "H_CUMLEN",
        dhname: str = "H_DELTALEN",
        tname: str = "T_CUMLEN",
        dtname: str = "T_DELTALEN",
        name: str = "poly",
        attributes: Optional[dict] = None,
    ):  # pylint: disable=arguments-differ
        """Used in deprecated methods."""

        super()._reset(xname, yname, zname)
        # additonal state properties for Polygons
        self._pname = pname

        self._hname = hname
        self._dhname = dhname
        self._tname = tname
        self._dtname = dtname
        self._name = name

        if not isinstance(values, pd.DataFrame):
            self._df = _xyz_io._from_list_like(values, self._zname, attributes, True)
        else:
            self._df = values

    @property
    def name(self):
        """Returns or sets the name of the instance."""
        return self._name

    @name.setter
    def name(self, newname):
        self._name = newname

    @property
    def pname(self):
        return self._pname

    @pname.setter
    def pname(self, value):
        super()._check_name(value)
        self._pname = value

    @property
    def hname(self):
        """Returns or set the name of the cumulative horizontal length.

        If the column does not exist, None is returned. Default name is H_CUMLEN.

        .. versionadded:: 2.1
        """
        return self._hname

    @hname.setter
    def hname(self, value):
        super()._check_name(value)
        self._hname = value

    @property
    def dhname(self):
        """Returns or set the name of the delta horizontal length column if it exists.

        If the column does not exist, None is returned. Default name is H_DELTALEN.

        .. versionadded:: 2.1
        """
        return self._dhname

    @dhname.setter
    def dhname(self, value):
        super()._check_name(value)
        self._dhname = value

    @property
    def tname(self):
        """Returns or set the name of the cumulative total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._tname

    @tname.setter
    def tname(self, value):
        super()._check_name(value)
        self._tname = value

    @property
    def dtname(self):
        """Returns or set the name of the delta total length column if it exists.

        .. versionadded:: 2.1
        """
        return self._dtname

    @dtname.setter
    def dtname(self, value):
        super()._check_name(value)
        self._dtname = value

    @property
    def dataframe(self) -> pd.DataFrame:
        """Returns or set the Pandas dataframe object."""
        return self._df

    @dataframe.setter
    def dataframe(self, df):
        self._df = df.apply(deepcopy)
        self._name_to_none_if_missing()

    def _name_to_none_if_missing(self):
        if self._dtname not in self._df.columns:
            self._dtname = None
        if self._dhname not in self._df.columns:
            self._dhname = None
        if self._tname not in self._df.columns:
            self._tname = None
        if self._hname not in self._df.columns:
            self._hname = None

    # ----------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------
    @inherit_docstring(inherit_from=XYZ.protected_columns)
    def protected_columns(self):
        return super().protected_columns() + [self.pname]

    @inherit_docstring(inherit_from=XYZ.from_file)
    @deprecation.deprecated(
        deprecated_in="2.21",  # should have been 2.16, but was forgotten until 2.21
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use xtgeo.polygons_from_file() instead",
    )
    def from_file(self, pfile, fformat="xyz"):
        self._reset(**_file_importer(pfile, fformat))

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

        return _xyz_io.to_file(self, pfile, fformat=fformat, ispolygons=True)

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
        if not wells:
            return None

        self._reset(**_wells_importer(wells, zone, resample))

        nwells = self.dataframe["WellName"].nunique()
        # as the previous versions did not have the WellName column, this is dropped
        # here for backward compatibility:
        self.dataframe = self.dataframe.drop("WellName", axis=1)

        if nwells == 0:
            return None
        else:
            return nwells

    def to_roxar(
        self,
        project,
        name,
        category,
        stype="horizons",
        realisation=0,
    ):  # pragma: no cover
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
            category (str): For horizons/zones/faults: for example 'DL_depth' and use
                a folder notation for clipboard/general2d_data
            stype (str): RMS folder type, 'horizons' (default), 'zones'
                or 'faults' or 'clipboard'  (in prep: well picks)
            realisation (int): Realisation number, default is 0


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
            None,
            realisation,
            None,
        )

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

        return mycopy

    @inherit_docstring(inherit_from=XYZ.from_list)
    @deprecation.deprecated(
        deprecated_in="2.16",
        removed_in="4.0",
        current_version=xtgeo.version,
        details="Use direct Polygons() initialisation instead",
    )
    def from_list(self, plist):

        kwargs = {}
        kwargs["values"] = _xyz_io._from_list_like(plist, "Z_TVDSS", None, True)
        self._reset(**kwargs)

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
